import json
import os

from dotenv import load_dotenv
from minio import Minio

load_dotenv()

minio_client=Minio(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)


def ensure_bucket_public_read(bucket: str) -> None:
    """确保 bucket 存在并设为匿名可读（幂等），让答案里的图片 URL 能被浏览器直接打开。"""
    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)
    policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"AWS": ["*"]},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::{bucket}/*"]
        }]
    }
    minio_client.set_bucket_policy(bucket, json.dumps(policy))