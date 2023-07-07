from hashashin.db import AbstractHashRepository
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from hashashin.classes import BinarySignature
from hashashin.classes import BinarySigModel
from hashashin.classes import FunctionFeatModel
from pathlib import Path
from typing import List, Optional, Union
from hashashin.classes import FunctionFeatures
import base64


class ElasticSearchHashRepository(AbstractHashRepository):
    @dataclass
    class ESHashRepoConfig:
        host: str
        port: int
        index: str
        user: str
        password: str

    def __init__(self, config: ESHashRepoConfig):
        self.config = config
        http, http_auth = ("http", None) if not (config.user and config.password) \
            else ("https", (config.user, config.password))
        self.client = Elasticsearch(hosts=f"{http}://{self.config.host}:{self.config.port}", http_auth=http_auth)

    @property
    def index_exists(self):
        return self.client.indices.exists(index=self.config.index)

    def create(self, shards=1, replicas=0):
        if self.index_exists:
            raise Exception(f"Index {self.config.index} already exists")
        self.client.indices.create(
            index=self.config.index,
            body={
                "settings": {
                    "number_of_shards": shards,
                    "number_of_replicas": replicas
                },
                "mappings": {
                    "properties": {
                        "filename": {"type": "text"},
                        "hash": {"type": "keyword", "index": True},
                        "signature": {
                            "type": "dense_vector",
                            "dims": BinarySignature.SIGNATURE_LEN,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "functions": {
                            "type": "nested",
                            "properties": {
                                "cyclomatic_complexity": {"type": "integer"},
                                "num_instructions": {"type": "integer"},
                                "num_strings": {"type": "integer"},
                                "max_string_length": {"type": "integer"},
                                "instruction_histogram":
                                    {
                                        "type": "dense_vector",
                                        "dims": FunctionFeatures.InstructionHistogram.length,
                                    },
                                "edge_histogram":
                                    {
                                        "type": "dense_vector",
                                        "dims": FunctionFeatures.EdgeHistogram.length,
                                    },
                                "dominator_signature": {"type": "unsigned_long"},
                                "constants": {"type": "keyword", "ignore_above": 256},
                                "strings": {"type": "keyword", "ignore_above": 256},
                            }
                        }
                    }
                }
            }
        )

    def drop_idx(self):
        if not self.index_exists:
            raise Exception(f"Index {self.config.index} does not exist")
        self.client.indices.delete(index=self.config.index)

    def save(self, signature: BinarySignature):
        self.client.index(
            index=self.config.index,
            body={
                "filename": str(signature.path),
                "hash": str(base64.b64encode(signature.binary_hash)),
                "signature": signature.signature,
                "functions": [
                    {
                        "cyclomatic_complexity": f.cyclomatic_complexity,
                        "num_instructions": f.num_instructions,
                        "num_strings": f.num_strings,
                        "max_string_length": f.max_string_length,
                        "instruction_histogram": f.instruction_histogram,
                        "edge_histogram": f.edge_histogram,
                        "dominator_signature": f.dominator_signature.asArray(),
                        "constants": f.constants.serialize(),
                        "strings": f.strings,
                    }
                    for f in signature.functionFeatureList
                ],
                "extraction_engine": str(signature.extraction_engine),
            },
        )

    @staticmethod
    def dict2model(source: dict) -> BinarySigModel:
        return BinarySigModel(
            hash=base64.b64decode(source["hash"]),
            sig=source["signature"],
            filename=Path(source["filename"]),
            functions=[
                FunctionFeatModel(
                    cyclomatic_complexity=f["cyclomatic_complexity"],
                    num_instructions=f["num_instructions"],
                    num_strings=f["num_strings"],
                    max_string_length=f["max_string_length"],
                    instruction_histogram=f["instruction_histogram"],
                    edge_histogram=f["edge_histogram"],
                    dominator_signature=f["dominator_signature"],
                    constants=FunctionFeatures.Constants.deserialize(f["constants"]),
                    strings=f["strings"],
                )
                for f in source["functions"]
            ],
        )

    def get(self, path: Path) -> Optional[BinarySignature]:
        raise NotImplementedError

    def getAll(self) -> List[BinarySignature]:
        raise NotImplementedError

    def match(
        self, signature: BinarySignature, threshold: int
    ) -> List[BinarySignature]:
        raise NotImplementedError

    def drop(self, option: Optional[Union[str, Path]] = None):
        raise NotImplementedError

    def summary(self, path_filter: str = ""):
        if path_filter != "":
            raise NotImplementedError
        return self.client.count(index=self.config.index)

    def find_closest_match(self, signature: BinarySignature) -> Optional[BinarySignature]:
        query = {
            "field": "signature",
            "query_vector": signature.signature,
            "k": 1,
            "num_candidates": 100,
        }
        resp = self.client.knn_search(
            index=self.config.index,
            knn=query,
        )
        if resp["hits"]["total"]["value"] == 0:
            return None
        return resp["hits"]["hits"][0]["_source"]

    def __len__(self):
        raise NotImplementedError
