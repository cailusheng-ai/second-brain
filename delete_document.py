"""删除已导入文档的独立脚本

用法（在项目根目录运行）：
    # 列出所有已导入文档
    python delete_document.py --list

    # 删除指定文档（按文档标题 file_title）
    python delete_document.py --title "使用LangChain+LangGraph开发AI应用（9） - LangGraph 进阶"

删除逻辑：删 Milvus chunk → 删 MinIO 图片 → 重建 BM25 IDF。
"""

import argparse
import os
import re
from collections import Counter

from dotenv import load_dotenv

# 加载 knowledge/.env（含 Milvus / MinIO / SPARSE_IDF 等配置）
load_dotenv("knowledge/.env")

from knowledge.utils.milvus_client_utils import get_milvus_client, fetch_all_contents
from knowledge.utils.MinioClient import minio_client
from knowledge.utils.sparse_embedder import SparseEmbedder


COLLECTION_NAME = os.getenv("CHUNK_COLLECTION_NAME")
BUCKET = os.getenv("MINIO_BUCKET_NAME")


def list_documents():
    """列出所有已导入的文档标题 + chunk 数。"""
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        print("向量库 collection 不存在，无文档")
        return
    chunks = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["file_title"],
        limit=10000,
    )
    if not chunks:
        print("向量库为空，无文档")
        return
    counter = Counter((c.get("file_title") or "无标题") for c in chunks)
    print(f"共 {len(chunks)} 个 chunk，{len(counter)} 个文档：")
    for title, count in counter.most_common():
        print(f"  {count:>4} 个 chunk | {title}")


def _extract_image_urls(chunks):
    """从 chunk 的 content 里提取 MinIO 图片 URL。"""
    urls = set()
    for c in chunks:
        content = c.get("content") or ""
        urls.update(
            re.findall(
                r"https?://[^\s\)\]]+?\.(?:png|jpe?g|gif|webp)",
                content,
                re.IGNORECASE,
            )
        )
    return urls


def _delete_minio_images(urls):
    """删除 MinIO 里的图片对象。"""
    for url in urls:
        marker = f"/{BUCKET}/"
        if marker not in url:
            continue
        obj_name = url.split(marker, 1)[-1]
        try:
            minio_client.remove_object(BUCKET, obj_name)
            print(f"  已删图片: {obj_name}")
        except Exception as e:
            print(f"  图片删除失败 {obj_name}: {e}")


def _rebuild_idf():
    """用剩余语料重新 fit BM25 IDF 并保存。"""
    remaining = fetch_all_contents(COLLECTION_NAME)
    embedder = SparseEmbedder()
    embedder.fit(remaining)
    idf_path = os.getenv("SPARSE_IDF_PATH")
    if idf_path:
        embedder.save(idf_path)
    print(f"BM25 IDF 已重建（剩余 {len(remaining)} 条语料）")


def delete_document(title):
    """删除指定文档：Milvus chunk + MinIO 图片 + 重建 IDF。"""
    client = get_milvus_client()
    if not client.has_collection(COLLECTION_NAME):
        print("向量库 collection 不存在")
        return

    chunks = client.query(
        collection_name=COLLECTION_NAME,
        filter=f'file_title == "{title}"',
        output_fields=["chunk_id", "content"],
        limit=10000,
    )
    if not chunks:
        print(f"未找到文档「{title}」")
        return

    print(f"文档「{title}」共 {len(chunks)} 个 chunk")

    # 1. 提取图片 URL（在删 chunk 之前提取）
    image_urls = _extract_image_urls(chunks)

    # 2. 删 Milvus chunk
    client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'file_title == "{title}"',
    )
    print(f"已从 Milvus 删除 {len(chunks)} 个 chunk")

    # 3. 删 MinIO 图片
    if image_urls:
        _delete_minio_images(image_urls)
    else:
        print("该文档无图片")

    # 4. 重建 BM25 IDF
    _rebuild_idf()

    print("删除完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除已导入文档")
    parser.add_argument("--list", action="store_true", help="列出所有已导入文档")
    parser.add_argument("--title", type=str, help="要删除的文档标题（file_title）")
    args = parser.parse_args()

    if args.list:
        list_documents()
    elif args.title:
        delete_document(args.title)
    else:
        parser.print_help()
