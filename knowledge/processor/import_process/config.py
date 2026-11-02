"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional
import os

from dotenv import load_dotenv
from knowledge.topic_taxonomy import TOPIC_TAXONOMY

load_dotenv()

@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 1000  # 切片最大长度
    min_content_length: int = 100   # 合并短内容的最小长度
    overlap_sentences: int = 50      # 句子级切分时的重叠字符数

    # ==================== 主题标签配置 ====================
    topic_chunk_k: int = 3           # 主题识别时使用的切片数量
    topic_chunk_size: int = 500      # 主题识别时使用的切片总字符上限
    topic_max_tags: int = 3          # 单篇文档最多主题标签数
    topic_taxonomy: List[str] = field(
        default_factory=lambda: list(TOPIC_TAXONOMY)
    )

    # ==================== 文本清洗配置（5层架构） ====================
    # 每个步骤有独立开关 + 可调参数，方便按文档类型定制
    #
    # 五层架构：
    #   第1层 编码层 — 统一换行符、删控制字符和零宽字符（基本不会误杀）
    #   第2层 空白层 — 压缩连续空行、去行首尾空白、合并错误断行（保护结构化内容）
    #   第3层 结构噪声层 — 去页眉页脚/页码/导航文本
    #   第4层 去重层 — 精确去重(MD5) + 模糊去重(默认关闭，只标记不删，rerank降权)
    #   第5层 术语归一化 — 繁简统一、全角半角统一

    # ============ 第1层：编码层 ============
    clean_layer1_enabled: bool = True           # 编码层总开关
    clean_normalize_newlines: bool = True       # 统一换行符 \r\n → \n
    clean_remove_control_chars: bool = True     # 去除不可打印控制字符 (\x00-\x08, \x0b-\x1f, \x7f)
    clean_remove_zero_width_chars: bool = True  # 去除零宽字符 (​, ‌, ‍, ﻿, ­ 等)

    # ============ 第2层：空白层 ============
    clean_layer2_enabled: bool = True           # 空白层总开关
    clean_strip_lines: bool = True              # 去除行首尾空白
    clean_merge_blank_lines: bool = True        # 合并连续多余空行
    clean_merge_broken_lines: bool = True       # 合并段落内部错误断行
    clean_max_blank_lines: int = 2              # 连续空行最多保留行数
    # 句末标点集：以这些字符结尾的行不合并下一行
    clean_sentence_end_chars: str = "。！？…」》）)\"'""'：；、"
    # 断行合并保护规则：这些结构不参与合并（表格行、列表项、代码块、引用块、分隔线）
    clean_merge_protected_patterns: bool = True  # 启用结构化内容保护

    # ============ 第3层：结构噪声层 ============
    clean_layer3_enabled: bool = True           # 结构噪声层总开关
    clean_remove_headers_footers: bool = True   # 去除重复页眉/页脚/页码
    clean_header_footer_max_length: int = 80    # 候选页眉行最大字符数
    clean_header_footer_min_occurrences: int = 2 # 最小出现次数判定为噪声（2-4可调）
    # 安全关键词：包含这些词的行即使重复也不删除（安全警告类重要内容保护）
    clean_hf_safety_keywords: str = "安全,警告,注意,危险,严禁,必须,重要,须知,警示,紧急"
    # 页眉页脚噪声关键词：含这些词的低信息量重复行优先判定为噪声
    clean_hf_noise_keywords: str = "页码,页眉,页脚,版权所有,版权,http,www,第,共,转第"

    # ============ 第5层：术语归一化 ============
    clean_layer5_enabled: bool = True           # 术语归一化总开关
    clean_trad_to_simp: bool = True             # 繁体中文 → 简体中文
    clean_fullwidth_to_halfwidth: bool = True   # 全角字符 → 半角字符（字母数字符号）

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    """
    简单默认值（name: str = "default"）用于不可变类型；
    field(default_factory=list)用于可变类型，避免多个实例共享同一个对象；
    field(default_factory=lambda: os.getenv(...))用于从环境变量动态读取配置，确保每次创建实例时都获取最新的环境变量值。
    使用 default_factory的关键原因是：简单默认值在类定义时计算一次，而 default_factory在每次实例化时计算，这对于可变类型和环境变量读取至关重要。"
    """
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 5

    # ==================== 速率限制 ====================
    requests_per_minute: int = 12  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()



# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config     # 告诉 Python：我要用的是外面的那个全局变量
    if _config is None:
        _config = ImportConfig.from_env()  # 修改的是全局变量
    return _config



