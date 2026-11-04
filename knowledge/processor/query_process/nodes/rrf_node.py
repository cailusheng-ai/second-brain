"""
RRF (Reciprocal Rank Fusion) 融合排序节点

职责：
  将三路检索结果（dense 稠密 / sparse 稀疏 / web 联网）通过加权 RRF
  算法融合去重，输出单一排序列表到 state["rrf_chunks"]。

三路召回各自的定位：
  - dense:  query → bge-base-zh-v1.5 稠密检索 → 语义匹配
  - sparse: query → BM25 稀疏检索 → 关键词精确匹配
  - web:    query → MCP 联网搜索 → 外部信息

RRF 原理：
  RRF 是一种经典的零-shot 排序融合算法，核心思想是"只信排名，不信分数"。
  每条文档的最终得分 = Σ w_s / (k_s + rank_s(d))
  - w_s:      检索通路 s 的权重
  - k_s:      通路 s 的平滑常数，k 越小则头部排名贡献越大
  - rank_s(d): 文档 d 在第 s 路检索中的排名（从 1 开始）

为什么用 RRF 而不是简单拼接：
  1. 各路检索的相似度分数尺度不可比（dense/sparse 是向量分数，web 是引擎内部排序）
  2. RRF 只关心相对排名，天然免疫分数尺度差异
  3. 计算量近乎为零（纯算术运算），不增加 GPU 开销

与下游 Rerank 节点的关系：
  RRF 做"粗排" — 两路合并、去重、按融合分截断（默认 Top-10）
  Rerank 做"精排" — 用 Cross-Encoder 对粗排结果逐条精细打分

去重策略：
  - dense/sparse 的 chunk 以 chunk_id 为主键去重
  - Web 搜索结果（无 chunk_id）以 URL / snippet 的 MD5 为主键去重
  - 去重时保留首次出现的版本，后续重复只追加来源标记

参考：
  - Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion outperforms Condorcet
    and individual rank learning methods." SIGIR 2009.
"""

import hashlib
import logging
from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)


class RRFNode(BaseNode):
    """RRF 融合排序节点。

    将 dense 稠密、sparse 稀疏、web 联网搜索三路召回结果
    通过加权 Reciprocal Rank Fusion 算法合并去重排序。

    三路召回的特点：
      - dense:  从内部知识库做 bge-base-zh-v1.5 稠密检索，语义匹配
      - sparse: 从内部知识库做 BM25 稀疏检索，关键词精确匹配
      - web:    从阿里云百炼平台联网搜索，补充时效性信息和外部知识
      Web 结果的结构（snippet/title/url）与 chunk 结构（chunk_id/content/title）不同，
      本节点在去重和融合时统一处理两种结构。

    Attributes:
        name: 节点名称，用于 LangGraph 注册和日志前缀。

    Usage:
        node = RRFNode()
        state = node(state)
        # state["rrf_chunks"] 已填充融合排序后的 Top-N 文档
    """

    name = "rrf_node"

    # ========================================================================
    # 文档唯一标识 — 兼容 chunk 和 web 两种结构
    # ========================================================================

    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """提取文档的唯一标识符，用于跨通路去重。

        不同通路返回的文档结构不同：
          - dense/sparse 返回 chunk 结构：含 chunk_id（Milvus 主键）
          - Web 返回搜索结果结构：含 snippet，无 chunk_id

        优先级：
          1. chunk_id  — Milvus 内部主键，最可靠（dense/sparse 通路）
          2. url       — 网页 URL，对 web 搜索结果有较好的唯一性
          3. snippet / content 的 MD5 — 最终降级方案

        Args:
            doc: 单条文档，结构取决于来源通路。

        Returns:
            文档唯一标识字符串。
        """
        # 优先用 Milvus 主键
        chunk_id = doc.get("chunk_id")
        if chunk_id is not None:
            return f"chunk_{chunk_id}"

        # web 结果优先用 URL（比 MD5 更语义化、可追溯）
        url = doc.get("url")
        if url:
            return f"web_{url}"

        # 最终降级：对文本内容取 MD5
        text = doc.get("snippet") or doc.get("content") or ""
        return f"hash_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"

    # ========================================================================
    # 去重与排名索引构建
    # ========================================================================

    def _collect_and_dedup(
        self,
        sources: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, int]]]:
        """收集所有通路的文档，去重，并记录每条文档在各通路中的排名。

        去重规则：
          - 以 _get_doc_id() 返回值为主键
          - 首次出现的文档版本被保留（content/title 等字段以首次为准）
          - 后续出现的同名文档只追加 _rrf_sources 标记，不覆盖内容
          - 同一通路对同一文档多次命中时，只保留最高排名（最小 rank 值）

        这样做的原因：
          如果同一文档同时被多路召回，说明其综合信号更强，
          RRF 分数会因多通路命中而更高。

        Args:
            sources: {来源名称: [文档列表]}，每个列表已按该通路的相关性降序排列。

        Returns:
            (doc_map, rank_map) 二元组：
            - doc_map:  {doc_id: 文档对象}，含内部字段 _rrf_sources
            - rank_map: {doc_id: {来源名称: 排名(从1开始)}}
        """
        doc_map: Dict[str, Dict[str, Any]] = {}
        rank_map: Dict[str, Dict[str, int]] = {}

        for source_name, docs in sources.items():
            if not docs:
                self.logger.debug(f"来源 '{source_name}' 无结果，跳过")
                continue

            self.logger.debug(f"处理来源 '{source_name}': {len(docs)} 条")

            for rank_0based, doc in enumerate(docs):
                if not isinstance(doc, dict):
                    self.logger.warning(
                        f"来源 '{source_name}' 中存在非 dict 类型的文档: "
                        f"{type(doc).__name__}，跳过"
                    )
                    continue

                doc_id = self._get_doc_id(doc)
                rank_1based = rank_0based + 1  # RRF 排名从 1 开始

                # --- 记录文档内容（首次出现为准）---
                if doc_id not in doc_map:
                    doc_copy = dict(doc)
                    doc_copy["_rrf_sources"] = [source_name]
                    doc_map[doc_id] = doc_copy
                else:
                    # 已存在：只追加来源标记（去重）
                    if source_name not in doc_map[doc_id]["_rrf_sources"]:
                        doc_map[doc_id]["_rrf_sources"].append(source_name)

                # --- 记录排名（同一来源只保留最高排名）---
                if doc_id not in rank_map:
                    rank_map[doc_id] = {}
                if source_name not in rank_map[doc_id]:
                    rank_map[doc_id][source_name] = rank_1based
                else:
                    # 保留更小的 rank 值 = 更高排名
                    if rank_1based < rank_map[doc_id][source_name]:
                        rank_map[doc_id][source_name] = rank_1based

        # 统计跨通路召回（同一文档被多路同时命中 → 信号更强）
        multi_source_count = sum(
            1 for doc in doc_map.values()
            if len(doc.get("_rrf_sources", [])) > 1
        )
        if multi_source_count > 0:
            self.logger.info(
                f"跨通路重复文档: {multi_source_count} 条 "
                f"({multi_source_count / max(len(doc_map), 1) * 100:.1f}%)"
            )

        return doc_map, rank_map

    # ========================================================================
    # RRF 分数计算
    # ========================================================================

    def _compute_rrf_scores(
        self,
        rank_map: Dict[str, Dict[str, int]],
        source_weights: Dict[str, float],
        source_k_values: Dict[str, int],
    ) -> Dict[str, float]:
        """计算每条文档的加权 RRF 分数。

        公式：
          RRF_score(d) = Σ_s w_s / (k_s + rank_s(d))

        其中：
          - w_s  = source_weights[s]，通路 s 的权重
          - k_s  = source_k_values[s]，通路 s 的平滑常数
          - rank_s(d) = 文档 d 在通路 s 中的排名，从 1 开始

        Args:
            rank_map:       {doc_id: {来源名称: 排名}}
            source_weights:  {来源名称: 权重系数}
            source_k_values: {来源名称: k 值}

        Returns:
            {doc_id: rrf_score}，分数越高越相关。
        """
        rrf_scores: Dict[str, float] = {}

        for doc_id, source_ranks in rank_map.items():
            score = 0.0
            detail_parts: List[str] = []  # 调试用：记录每路贡献

            for source_name, rank in source_ranks.items():
                weight = source_weights.get(source_name, 1.0)
                k = source_k_values.get(source_name, self.config.rrf_k)
                contribution = weight / (k + rank)
                score += contribution
                detail_parts.append(
                    f"{source_name}(r={rank},w={weight},k={k})={contribution:.6f}"
                )

            rrf_scores[doc_id] = score
            self.logger.debug(
                f"doc_id={doc_id} RRF={score:.6f} = {' + '.join(detail_parts)}"
            )

        return rrf_scores

    # ========================================================================
    # 主流程
    # ========================================================================

    async def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 RRF 融合排序。

        处理步骤：
          1. 从 state 中收集三路检索结果（dense / sparse / web）
          2. 文档去重 + 排名索引构建
          3. 计算加权 RRF 分数
          4. 按分数降序 → 截断 Top-N
          5. 注入元数据（rrf_score / rrf_sources / rrf_ranks）
          6. 写入 state["rrf_chunks"]

        Args:
            state: 查询图状态，需包含至少一路非空检索结果。

        Returns:
            更新后的状态，rrf_chunks 字段已填充。
        """
        # ================================================================
        # Step 1: 收集三路检索结果
        # ================================================================
        self.log_step("step1", "收集三路检索结果（dense / sparse / web）")

        # 从 state 中安全获取各路结果，确保不为 None
        sources: Dict[str, List[Dict[str, Any]]] = {
            "dense":  state.get("dense_chunks") or [],
            "sparse": state.get("sparse_chunks") or [],
            "web":    state.get("web_search_docs") or [],
        }

        # 统计各路文档数量
        dense_count = len(sources["dense"])
        sparse_count = len(sources["sparse"])
        web_count = len(sources["web"])
        total_before_dedup = dense_count + sparse_count + web_count

        self.logger.info(
            f"输入统计: dense={dense_count}, sparse={sparse_count}, web={web_count} → "
            f"总计 {total_before_dedup} 条（去重前）"
        )

        # 快速路径：所有通路均无结果
        if total_before_dedup == 0:
            self.logger.warning(
                "三路检索通路均无结果，RRF 返回空列表。"
                "请检查上游 dense / sparse / web 检索节点是否正常执行。"
            )
            state["rrf_chunks"] = []
            return state

        # ================================================================
        # Step 2: 去重并建立排名索引
        # ================================================================
        self.log_step("step2", "文档去重并建立各通路排名索引")

        doc_map, rank_map = self._collect_and_dedup(sources)
        unique_count = len(doc_map)
        dedup_ratio = (
            (total_before_dedup - unique_count) / total_before_dedup * 100
            if total_before_dedup > 0 else 0
        )
        self.logger.info(
            f"去重完成: {total_before_dedup} → {unique_count} 条唯一文档 "
            f"(去重率 {dedup_ratio:.1f}%)"
        )

        # ================================================================
        # Step 3: 计算加权 RRF 分数（每条通路独立的 k 值）
        # ================================================================
        self.log_step(
            "step3",
            f"计算加权 RRF 分数 (k: dense/sparse/web={self.config.rrf_k})",
        )

        # 三路权重设计：web 加权，避免被本地双路（dense+sparse 重叠）翻倍分数挤掉
        source_weights = {
            "dense":  1.0,
            "sparse": 1.0,
            "web":    1.5,
        }

        # 三路统一 k 值：结果较多，k 大→平滑噪声、防头部垄断
        source_k_values = {
            "dense":  self.config.rrf_k,
            "sparse": self.config.rrf_k,
            "web":    self.config.rrf_k,
        }
        self.logger.info(
            f"通路参数: weights={source_weights}, k_values={source_k_values}"
        )

        rrf_scores = self._compute_rrf_scores(
            rank_map, source_weights, source_k_values
        )

        # 统计分数分布
        if rrf_scores:
            all_scores = list(rrf_scores.values())
            self.logger.info(
                f"RRF 分数范围: [{min(all_scores):.6f}, {max(all_scores):.6f}], "
                f"均值: {sum(all_scores) / len(all_scores):.6f}"
            )

        # ================================================================
        # Step 4: 按 RRF 分数降序排列
        # ================================================================
        self.log_step("step4", "按 RRF 分数降序排列")

        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: rrf_scores[did],
            reverse=True,
        )

        # ================================================================
        # Step 5: 截断 Top-N，注入元数据
        # ================================================================
        max_results = self.config.rrf_max_results
        self.log_step("step5", f"截断 Top-{max_results} 并注入元数据")

        rrf_chunks: List[Dict[str, Any]] = []
        for rank_idx, doc_id in enumerate(sorted_doc_ids[:max_results]):
            doc = doc_map[doc_id]

            # 注入 RRF 元数据字段
            doc["rrf_score"] = round(rrf_scores[doc_id], 6)
            # _rrf_sources 是内部标记字段，提升为最终输出字段
            doc["rrf_sources"] = doc.pop("_rrf_sources", [])
            doc["rrf_ranks"] = rank_map.get(doc_id, {})
            doc["rrf_final_rank"] = rank_idx + 1

            rrf_chunks.append(doc)

        # ================================================================
        # Step 6: 详细日志 — 输出 Top-3 的融合细节，便于调试
        # ================================================================
        if rrf_chunks:
            self.logger.info("── RRF Top-3 融合详情 ──")
            for idx, doc in enumerate(rrf_chunks[:3]):
                # 兼容两种结构：chunk 用 title，web 用 snippet[:50]
                label = (
                    doc.get("title")
                    or (doc.get("snippet") or "")[:50]
                    or "N/A"
                )
                sources_str = "+".join(doc.get("rrf_sources", []))
                ranks_str = ", ".join(
                    f"{src}:#{rank}"
                    for src, rank in doc.get("rrf_ranks", {}).items()
                )
                self.logger.info(
                    f"  #{idx + 1} | score={doc['rrf_score']:.6f} | "
                    f"通路=[{sources_str}] | 各通路排名=({ranks_str}) | "
                    f"label={label}"
                )

        # 统计被截断丢弃的文档数
        dropped = max(0, len(sorted_doc_ids) - max_results)
        if dropped > 0:
            self.logger.info(
                f"截断丢弃 {dropped} 条低分文档（超出 Top-{max_results} 限制）"
            )

        self.logger.info(
            f"RRF 融合完成: {total_before_dedup} 条输入 → "
            f"{unique_count} 条去重 → {len(rrf_chunks)} 条输出"
        )

        # ================================================================
        # Step 7: 写入 state
        # ================================================================
        state["rrf_chunks"] = rrf_chunks
        return state
