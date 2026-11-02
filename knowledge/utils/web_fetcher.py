"""网页/公众号文章抓取工具

抓取 URL（适配微信公众号 mp.weixin.qq.com），提取标题/作者/正文并转换为 Markdown。

公众号文章结构（已验证）：
  - 标题：<meta property="og:title">
  - 作者：<meta property="og:article:author">
  - 正文：<div id="js_content">
  - 图片：懒加载，实际 URL 在 <img data-src="..."> 里
"""

import logging
from typing import Dict

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md

logger = logging.getLogger(__name__)

# 公众号需要浏览器 UA + Referer，否则可能返回空页或"环境异常"
_WECHAT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_WECHAT_REFERER = "https://mp.weixin.qq.com/"


def fetch_article_to_markdown(url: str) -> Dict[str, str]:
    """抓取 URL 并转 Markdown，返回 {"title", "author", "markdown"}。"""
    html = _fetch_html(url)
    return _parse_article(html)


def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": _WECHAT_UA,
        "Referer": _WECHAT_REFERER,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    # 导入端为同步图，用同步客户端做单次抓取
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.text


def _parse_article(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "lxml")

    title = ""
    meta = soup.find("meta", property="og:title")
    if meta and meta.get("content"):
        title = meta["content"].strip()
    if not title:
        h1 = soup.find(id="activity-name")
        if h1:
            title = h1.get_text(strip=True)

    author = ""
    meta = soup.find("meta", property="og:article:author")
    if meta and meta.get("content"):
        author = meta["content"].strip()
    if not author:
        name = soup.find(id="js_name")
        if name:
            author = name.get_text(strip=True)

    markdown = ""
    images = []
    js_content = soup.find(id="js_content")
    if js_content:
        # 图片懒加载：把 data-src 复制到 src，markdownify 才能提取到图片 URL
        for img in js_content.find_all("img"):
            src = img.get("data-src") or img.get("data-original")
            if src:
                img["src"] = src
                images.append(src)
        markdown = md(
            str(js_content),
            heading_style="ATX",
            strip=["script", "style"],
        ).strip()

    return {"title": title, "author": author, "markdown": markdown, "images": images}
