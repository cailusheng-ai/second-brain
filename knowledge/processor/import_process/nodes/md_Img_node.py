import logging
import os
import re
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.MinioClient import minio_client, ensure_bucket_public_read


class MDImgNode(BaseNode):
    name = "MDImg"
    config=get_config()

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.log_step("step1","获取MD文件内容、MD文件的Path对象、MD文件关联的图片目录的Path对象（定位图片储存目录）")
        MD_content,md_path_obj,img_path_obj=self._get_content_MDObi_ImgObi(state)

        if not img_path_obj.exists():
            self.logger.info("未找到images目录，跳过图片处理流程，直接进行下一步")
            state["md_content"] = MD_content
            return state

        self.log_step("step2",f"扫描images目录{img_path_obj}，筛选出在MD文件中被引用的图片列表")
        target_img_list=self._scan_and_filter_images(MD_content,img_path_obj,self.config.image_extensions)
        if not target_img_list:
            self.logger.info("未找到在MD文件中的图片引用")
            state["md_content"] = MD_content
            return state

        self.log_step("step3","根据图片路径上传minio服务器，得到minio路径，更新md_content内容，更新state")
        new_md_content=self._upload_images_and_replaces_md_content(minio_client,target_img_list,MD_content)
        state["md_content"]=new_md_content

        # 只备份处理后的 md，不覆盖 file_title（entry_node 已按原始文件名设置）。
        # 否则会给标题追加 _new，导致按标题去重时 key 对不上、旧数据永远删不掉。
        self._backup_new_md_file(md_path_obj,new_md_content)
        return state

    def _get_content_MDObi_ImgObi(self, state:ImportGraphState)->Tuple[str,Path,Path]:
        """
        :param state:
        :return: MD文件内容、MD文件Path对象、MD文件中关联的图片的Path对象（图片储存目录）
        """

        self.log_step("step1.1","获取MD文件的Path对象")
        md_path=state["md_path"]
        if not md_path:
            raise FileProcessingError("md文件路径不存在",node_name=self.name)
        md_path_obi=Path(md_path)

        self.log_step("step1.2", "获取MD文件内容")
        lines=[]
        try:
            with open(md_path,"r",encoding="utf-8") as f:
                for line in f:
                    lines.append(line.rstrip("\n"))
            md_content="\n".join(lines)
        except IOError as e:
            raise FileProcessingError(f"无法读取文件{md_path_obi},原因是{e}",node_name=self.name)

        self.log_step("step1.3", "获取MD文件内图片的Path对象")
        img_path_obj=md_path_obi.parent/f"{md_path_obi.stem}.assets"

        return md_content,md_path_obi,img_path_obj

    def _scan_and_filter_images(self,MD_content,img_path_obj,allowed_extensions)->list[Tuple[str,str]]:
        """
        筛选MD文件中被引用的图片列表
        :param MD_content:
        :param img_path_obj:
        :param allowed_extensions:
        :return:【“图片名称”，“图片路径”】
        """
        target_img_list=[]

        self.log_step("step2.1","获取图片文件名")
        #获取图片目录下所有的图片文件名
        img_name_list=os.listdir(img_path_obj)
        for img_name in img_name_list:
            #检查文件拓展名是否在允许列表中
            if os.path.splitext(img_name)[1].lower() not in allowed_extensions:
                continue

            self.log_step("step2.2", "检查图片是否在MD文件中被引用")
            if not self._is_image_referenced(img_name,MD_content):
                self.logger.debug(f"图片{img_name}未在文中引用,跳过")
                continue

            self.log_step("step2.3", "获取图片路径")
            img_full_path=str(img_path_obj/img_name)

            target_img_list.append((img_name,img_full_path))

        self.logger.info(f"共筛选出{len(target_img_list)}张需要处理的图片")
        return target_img_list

    def _is_image_referenced(self, img_name: str, MD_content: str) -> bool:
        """判断图片文件名是否在 MD 的图片语法中被引用。"""
        image_pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        )
        return image_pattern.search(MD_content) is not None

    def _upload_images_and_replaces_md_content(self, minio_client, target_img_list, MD_content):
        # 确保 bucket 匿名可读，否则浏览器无法直接打开图片 URL
        ensure_bucket_public_read(self.config.minio_bucket)

        minio_imgName_imgUrl={}
        for img_name,img_full_path in target_img_list:
            object_name=f"second-brain/{img_name}"
            ext = os.path.splitext(img_full_path)[1].lower()
            content_type = f"image/{ext[1:]}" if ext.startswith(".") else "application/octet-stream"

            if minio_client:

                try:
                    minio_client.fput_object(
                        bucket_name=self.config.minio_bucket,
                        object_name=object_name,
                        file_path=img_full_path,
                        content_type=content_type
                    )
                    minio_url=f"http://"+self.config.minio_endpoint+"/"+self.config.minio_bucket+"/"+object_name
                    minio_imgName_imgUrl[img_name]=minio_url
                    self.log_step("step3.1",f"图片上传成功,{img_name}->{minio_url}")
                except Exception as e:
                    self.logger.warning(f"图片{img_name}上传失败：{e}")

            else:
                self.logger.warning("客户端未初始化，跳过实际上传")

        new_md_content=MD_content
        for img_name,minio_url in minio_imgName_imgUrl.items():
            # 保留原 alt 文本，仅替换图片 URL（与链接导入行为一致，不再调用 VLM 生成摘要）
            replace_pattern=re.compile(r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",re.IGNORECASE)
            new_md_content=replace_pattern.sub(rf"![\1]({minio_url})",new_md_content)
        self.log_step("step3.2",f"成功替换MD文件中的{len(minio_imgName_imgUrl)}张图片")
        return new_md_content

    def _backup_new_md_file(self,original_path: Path,new_md_content: str) -> str:
        """
        将处理后的 Markdown 内容写入新文件。

        Args:
            original_md_path_str (str): 原始文件路径。
            new_md_content (str): 新的 Markdown 内容。

        Returns:
            str: 新文件的绝对路径。
        """
        self.log_step("step_5", "备份新文件")


        new_file_path = original_path.with_name(
            f"{original_path.stem}_new{original_path.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)
        return str(new_file_path)
