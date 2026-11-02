import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_client_utils import get_milvus_client

load_dotenv()

#主类，向milvus中存储数据
class ChunksSaveToMilvusNode(BaseNode):
    name="chunks_save_to_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        chunks=state.get("chunks")
        if not isinstance(chunks,list) or not chunks:
            raise ValidationError("chunks类型错误或为空", node_name=self.name)

        milvus_client=get_milvus_client()
        save_to_milvus=_SaveToMilvus(milvus_client,os.getenv("CHUNK_COLLECTION_NAME"))
        result=save_to_milvus.insert(chunks)
        self.logger.info(f"成功保存{result['insert_count']}条数据到milvus中")
        inserted_ids=result["ids"]

        for chunk,chunk_id in zip(chunks,inserted_ids):
            chunk["chunk_id"]=chunk_id

        self.logger.info(f"成功将{len(inserted_ids)}个ID写入到chunks中")
        # print(json.dumps(state,ensure_ascii=False,indent=2))
        return state



#创建Schema约束
class _CreateSchema:
    @staticmethod
    def create_schema(milvus_client:MilvusClient):
        schema=milvus_client.create_schema()
        schema.add_field(field_name="chunk_id",auto_id=True,is_primary=True,datatype=DataType.INT64)
        schema.add_field(field_name= "content",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name= "title",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name= "parent_title",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name= "chapter_path",datatype=DataType.VARCHAR,max_length=2000)
        schema.add_field(field_name= "file_title",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name= "topics",datatype=DataType.VARCHAR,max_length=1000)
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=int(os.getenv("EMBEDDING_DIM", "768")))
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)#稀疏浮点向量数据类型

        return schema

#创建索引
class _CreateIndexes:
    @staticmethod
    def create_indexes(milvus_client:MilvusClient):
        index_params=milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"
        )
        return index_params

#向Collection中保存数据
class _SaveToMilvus:
    MILVUS_SCHEMA_FIELDS = {
        "content", "title", "parent_title", "chapter_path",
        "file_title", "topics", "dense_vector", "sparse_vector"
    }

    def __init__(self,milvus_client:MilvusClient,collection_name:str):
        self.milvus_client=milvus_client
        self.collection_name=collection_name
        self.logger=logging.getLogger(__name__)

    def _filter_chunk(self, chunk: dict) -> dict:
        return {k: v for k, v in chunk.items() if k in self.MILVUS_SCHEMA_FIELDS}

    def insert(self,chunks:list[dict[str,Any]]):
        self._ensure_collection()

        filtered = [self._filter_chunk(c) for c in chunks]

        result=self.milvus_client.insert(
            collection_name=self.collection_name,
            data=filtered
        )
        self.logger.info(f"成功插入{len(filtered)}条数据到{self.collection_name}中")
        return result

    def _ensure_collection(self):
        """确保 collection 存在且 schema（维度 + 字段）与当前配置一致，否则 drop 重建。"""
        target_dim = int(os.getenv("EMBEDDING_DIM", "768"))
        expected_fields = set(_SaveToMilvus.MILVUS_SCHEMA_FIELDS)

        if self.milvus_client.has_collection(collection_name=self.collection_name):
            desc = self.milvus_client.describe_collection(collection_name=self.collection_name)
            actual_fields = {
                f.get("name") for f in desc.get("fields", []) if f.get("name") != "chunk_id"
            }
            actual_dim = None
            for field in desc.get("fields", []):
                if field.get("name") == "dense_vector":
                    actual_dim = field.get("params", {}).get("dim")
                    break
            if actual_dim == target_dim and actual_fields == expected_fields:
                return  # schema 一致，直接复用

            self.logger.info(
                f"collection {self.collection_name} schema 不匹配"
                f"（维度 {actual_dim} vs {target_dim}，字段 {actual_fields} vs {expected_fields}），drop 后重建"
            )
            self.milvus_client.drop_collection(collection_name=self.collection_name)

        self.milvus_client.create_collection(
            collection_name=self.collection_name,
            schema=_CreateSchema.create_schema(self.milvus_client),
            index_params=_CreateIndexes.create_indexes(self.milvus_client)
        )
