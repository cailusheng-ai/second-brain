from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState




class EntryNode(BaseNode):
    name = "entry_node"
    #实现process类
    def process(self,state:ImportGraphState):
        """
        处理入口逻辑
        1、获取页面文件的路径
        2、通过后缀名检测文件类型
        3、根据文件类型决定是否启用相应的文件读取
        4、提取文件名（不含拓展名）
        :param state:图状态
        :return:更新后的图状态
        """
        #1、获取页面文件路径
        self.log_step("step_1", "获取文件路径")
        import_file_path=state["import_file_path"]

        #判断文件路径是否存在
        if not import_file_path:
            raise ValidationError("import_file_path 不能为空",node_name=self.name)


        #2、通过文件路径获取后缀名
        #获取path对象
        file_path=Path(import_file_path)
        #通过path对象获取后缀名
        suffix=file_path.suffix.lower()
        self.log_step("step_2",f"已检测到文件类型为：{suffix}")

        #3、只支持 .md 文件
        if suffix != ".md":
            raise ValidationError(f"仅支持 .md 文件，当前文件类型：{suffix}", node_name=self.name)

        state["md_path"] = import_file_path
        self.logger.info("启用 md 文件读取")

        #4、读取文件标题
        state["file_title"]=file_path.stem
        self.logger.info(f"读取到文件标题：{state['file_title']}")

        return state
