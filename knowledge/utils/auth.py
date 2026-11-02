"""单用户 JWT 鉴权模块

- 单用户账号密码从环境变量读取（ADMIN_USERNAME / ADMIN_PASSWORD）
- 登录校验用 bcrypt 密码哈希
- 签发/验证 JWT（PyJWT）

用法：
    # 路由里加鉴权依赖
    @app.post("/query")
    async def query(..., username: str = Depends(get_current_user)):
        ...
"""

import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
JWT_SECRET = os.getenv("JWT_SECRET", "please-change-me-to-a-random-string")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7  # token 有效期 7 天

# 启动时对明文密码做一次 bcrypt 哈希（用于登录校验）
_PASSWORD_HASH = bcrypt.hashpw(_ADMIN_PASSWORD.encode("utf-8"), bcrypt.gensalt())

_bearer = HTTPBearer(auto_error=False)


def verify_password(username: str, password: str) -> bool:
    """校验账号密码。"""
    if username != ADMIN_USERNAME:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), _PASSWORD_HASH)
    except Exception:
        return False


def create_token(username: str) -> str:
    """签发 JWT。"""
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    """鉴权依赖：校验请求头里的 Bearer token，返回用户名。"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="未登录")
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="无效的 token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="token 无效或已过期")
