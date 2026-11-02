"""URL 导入接口的请求/响应模型。"""

from pydantic import BaseModel


class UrlImportRequest(BaseModel):
    url: str


class UrlImportResponse(BaseModel):
    message: str
    task_id: str
