import io
import logging
import os.path
import re
import shutil
import uuid
from datetime import datetime, timedelta

import httpx
from fastapi import UploadFile

from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.import_graph import create_import_graph
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.MinioClient import minio_client, ensure_bucket_public_read
from knowledge.utils.milvus_client_utils import get_milvus_client, fetch_all_contents
from knowledge.utils.sparse_embedder import SparseEmbedder
from knowledge.utils.web_fetcher import fetch_article_to_markdown


# 导入图节点后继关系（用于标记"进行中"节点；entry_node 为条件分支，动态判断）
IMPORT_NEXT_NODES = {
    "md_Img_node": ["text_clean_node"],
    "text_clean_node": ["document_split_node"],
    "document_split_node": ["topic_tagging_node"],
    "topic_tagging_node": ["chunks_embedding_node"],
    "chunks_embedding_node": ["chunks_save_to_milvus_node"],
    "chunks_save_to_milvus_node": [],
}


def _get_next_nodes(node_name: str, node_data: dict) -> list:
    """返回节点完成后的后继节点列表。entry_node 直接进入图片处理。"""
    if node_name == "entry_node":
        return ["md_Img_node"]
    return IMPORT_NEXT_NODES.get(node_name, [])


class ImportFileService:
    logger=logging.Logger(__name__)
    #初始化任务节点
    def __init__(self,task_service:TaskService):
        self.task_service=task_service


    #获取文件保存路径
    def get_pdf_save_path(self):
        return os.path.join(get_local_base_dir(),datetime.now().strftime("%Y%m%d"))

    #清理超过保留天数的临时文件目录（按日期目录名 YYYYMMDD 判断）
    def _clean_expired_temp_files(self):
        keep_days = int(os.getenv("TEMP_FILE_KEEP_DAYS", "7"))
        base_dir = get_local_base_dir()
        if not os.path.exists(base_dir):
            return
        cutoff = datetime.now() - timedelta(days=keep_days)
        for name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, name)
            if not os.path.isdir(dir_path):
                continue
            try:
                dir_date = datetime.strptime(name, "%Y%m%d")
            except ValueError:
                continue
            if dir_date < cutoff:
                shutil.rmtree(dir_path, ignore_errors=True)
                self.logger.info(f"清理过期临时目录: {dir_path}")

    #保存上传的文件到本地
    def save_upload_file_to_local(self,file:UploadFile,file_dir:str):
        #判断路径是否存在
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        #保存文件
        import_file_path=os.path.join(file_dir,file.filename)
        with open(import_file_path,"wb") as f:
            shutil.copyfileobj(file.file,f)
        return import_file_path


    #调用保存上传的文件到本地的方法，并记录
    def file_upload_process(self,file:UploadFile):
        #清理过期的临时文件
        self._clean_expired_temp_files()
        #获取文件保存路径
        date_dir=self.get_pdf_save_path()
        #生成UUID
        task_id=uuid.uuid4().hex[:12]
        #生成最终路径
        file_path=os.path.join(date_dir,task_id)


        #记录当前正在运行的节点，前端使用
        self.task_service.mark_node_running(task_id,"upload_file")
        #调用方法，将文件保存到本地
        import_file_path=self.save_upload_file_to_local(file,file_path)
        #如果是 zip 压缩包，解压并找到其中的 .md 文件
        if import_file_path.lower().endswith(".zip"):
            import_file_path=self._extract_zip_and_find_md(import_file_path,file_path)
        #记录当前完成节点，前端使用
        self.task_service.mark_node_done(task_id,"upload_file")
        return task_id,file_path,import_file_path

    def _extract_zip_and_find_md(self,zip_path:str,extract_dir:str)->str:
        """解压 zip 到指定目录，返回其中第一个 .md 文件的路径。"""
        import zipfile
        with zipfile.ZipFile(zip_path,"r") as z:
            z.extractall(extract_dir)
        for root,dirs,files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(".md"):
                    return os.path.join(root,f)
        raise ValueError("zip 压缩包中未找到 .md 文件")

    def run_upload_file_task(self,task_id,file_path,import_file_path):

        try:
            #将节点状态更新为运行中
            self.task_service.update_task_status(task_id,"processing")

            #按标题更新：已导入过相同标题则删旧数据后再导入（实现覆盖更新）
            title = os.path.splitext(os.path.basename(import_file_path))[0]
            if self._title_exists(title):
                self.logger.info(f"标题「{title}」已存在，删除旧数据后更新")
                self._delete_by_title(title)

            #标记图内第一个节点为运行中
            self.task_service.mark_node_running(task_id, "entry_node")

            #调用create_import_graph方法构建图对象，开始运行各个节点
            #构建初始数据
            initial_status:ImportGraphState={
                "task_id": task_id,
                "import_file_path":import_file_path,
                "file_dir":file_path
            }
            #构建图实例
            agent=create_import_graph()

            for event in agent.stream(initial_status):
                for node_name,node_data in event.items():
                    self.task_service.mark_node_done(task_id, node_name)
                    self.logger.info(f"{task_id}完成了节点{node_name}")

                    #标记后继节点为运行中
                    for next_name in _get_next_nodes(node_name, node_data):
                        self.task_service.mark_node_running(task_id, next_name)

            #标记任务处理完成
            self.task_service.update_task_status(task_id,"completed")
        except Exception as e:
            self.task_service.update_task_status(task_id, "failed")
            self.logger.exception(f"文件向量化任务{task_id}失败")
            raise e

    def run_url_import_task(self, task_id: str, url: str):
        """后台任务：抓取 URL → 保存为本地 md → 复用导入图流程。"""
        try:
            self.task_service.update_task_status(task_id, "processing")
            self.task_service.mark_node_running(task_id, "upload_file")

            # 清理过期的临时文件
            self._clean_expired_temp_files()

            # 抓取并转换为 Markdown
            article = fetch_article_to_markdown(url)
            if not article.get("markdown"):
                raise ValueError("抓取到的正文为空")

            # 下载图片并上传 MinIO，替换 md 里的图片 URL（避免微信 CDN 时效性）
            markdown = article["markdown"]
            images = article.get("images") or []
            if images:
                url_map = self._upload_images_to_minio(images)
                for wechat_url, minio_url in url_map.items():
                    markdown = markdown.replace(wechat_url, minio_url)

            # 保存为本地 md 文件（复用 temp_data/日期/task_id 结构）
            date_dir = self.get_pdf_save_path()
            file_path = os.path.join(date_dir, task_id)
            title = (article.get("title") or "").strip() or task_id
            safe_title = self._sanitize_filename(title)
            import_file_path = self._save_markdown(file_path, safe_title, markdown)

            self.task_service.mark_node_done(task_id, "upload_file")
            # 复用现有导入图流程（entry_node 会识别为 .md）
            self.run_upload_file_task(task_id, file_path, import_file_path)
        except Exception as e:
            self.task_service.mark_node_done(task_id, "upload_file")
            self.task_service.update_task_status(task_id, "failed")
            self.logger.exception(f"链接导入任务{task_id}失败")
            raise e

    def _title_exists(self, title: str) -> bool:
        """查 Milvus 是否已有相同 file_title 的 chunk（按标题去重）。"""
        collection_name = os.getenv("CHUNK_COLLECTION_NAME")
        if not collection_name:
            return False
        try:
            result = get_milvus_client().query(
                collection_name=collection_name,
                filter=f'file_title == "{title}"',
                output_fields=["chunk_id"],
                limit=1,
            )
            return bool(result)
        except Exception as e:
            self.logger.warning(f"标题查重失败（按不存在处理）：{e}")
            return False

    def _delete_by_title(self, title: str):
        """删除指定文档的旧数据：Milvus chunk + MinIO 图片 + 重建 BM25 IDF。"""
        collection_name = os.getenv("CHUNK_COLLECTION_NAME")
        bucket = os.getenv("MINIO_BUCKET_NAME")

        chunks = get_milvus_client().query(
            collection_name=collection_name,
            filter=f'file_title == "{title}"',
            output_fields=["chunk_id", "content"],
            limit=10000,
        )
        if not chunks:
            return

        # 提取图片 URL（在删 chunk 之前）
        image_urls = set()
        for c in chunks:
            content = c.get("content") or ""
            image_urls.update(
                re.findall(r"https?://[^\s\)\]]+?\.(?:png|jpe?g|gif|webp)", content, re.IGNORECASE)
            )

        # 删 Milvus chunk
        get_milvus_client().delete(
            collection_name=collection_name,
            filter=f'file_title == "{title}"',
        )
        self.logger.info(f"删除旧 chunk {len(chunks)} 条（标题「{title}」）")

        # 删 MinIO 图片
        for url in image_urls:
            marker = f"/{bucket}/"
            if marker not in url:
                continue
            obj_name = url.split(marker, 1)[-1]
            try:
                minio_client.remove_object(bucket, obj_name)
            except Exception as e:
                self.logger.warning(f"删除旧图片失败 {obj_name}: {e}")

        # 重建 BM25 IDF（用剩余语料）
        remaining = fetch_all_contents(collection_name)
        embedder = SparseEmbedder()
        embedder.fit(remaining)
        idf_path = os.getenv("SPARSE_IDF_PATH")
        if idf_path:
            embedder.save(idf_path)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """清理文件名中的 Windows 非法字符。"""
        for ch in r'\/:*?"<>|':
            name = name.replace(ch, "_")
        return name.strip() or "article"

    def _save_markdown(self, file_path: str, title: str, markdown: str) -> str:
        """把 Markdown 正文写入本地文件，返回文件路径。"""
        os.makedirs(file_path, exist_ok=True)
        md_path = os.path.join(file_path, f"{title}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        self.logger.info(f"链接内容已保存为 Markdown：{md_path}")
        return md_path

    def _upload_images_to_minio(self, image_urls: list) -> dict:
        """下载远程图片并上传 MinIO，返回 {原始URL: minio地址} 映射。"""
        cfg = get_config()
        # 确保 bucket 存在（首次使用时自动创建）
        if not minio_client.bucket_exists(cfg.minio_bucket):
            minio_client.make_bucket(cfg.minio_bucket)
        # 确保 bucket 匿名可读（答案里图片 URL 要能在浏览器直接打开）
        ensure_bucket_public_read(cfg.minio_bucket)
        mapping = {}
        for url in image_urls:
            try:
                data, content_type = self._download_image(url)
                if data is None:
                    continue
                ext = self._ext_from_content_type(content_type)
                object_name = f"article_images/{uuid.uuid4().hex}.{ext}"
                minio_client.put_object(
                    bucket_name=cfg.minio_bucket,
                    object_name=object_name,
                    data=io.BytesIO(data),
                    length=len(data),
                    content_type=content_type,
                )
                minio_url = (
                    f"http://{cfg.minio_endpoint}/"
                    f"{cfg.minio_bucket}/{object_name}"
                )
                mapping[url] = minio_url
            except Exception as e:
                self.logger.warning(f"图片下载/上传失败：{url[:60]}... {e}")
        return mapping

    @staticmethod
    def _download_image(url: str):
        """下载图片，返回 (bytes, content_type)，失败抛出异常。"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://mp.weixin.qq.com/",
        }
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.content, resp.headers.get("content-type", "image/jpeg")

    @staticmethod
    def _ext_from_content_type(content_type: str) -> str:
        """从 content-type 推断图片扩展名。"""
        mapping = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
            "image/svg+xml": "svg",
        }
        ctype = (content_type or "").split(";")[0].strip().lower()
        return mapping.get(ctype, "jpg")

