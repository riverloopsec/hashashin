from hashashin.db import AbstractHashRepository
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from hashashin.classes import BinarySignature
from hashashin.classes import BinarySigModel
from hashashin.classes import FunctionFeatModel
from hashashin.classes import extractor_from_name
from pathlib import Path
from typing import List, Optional, Union
from hashashin.classes import FunctionFeatures
from hashashin.classes import AbstractFunction
from hashashin.classes import merge_uint32_to_int
import base64


class ElasticSearchHashRepository(AbstractHashRepository):
    @dataclass
    class ESHashRepoConfig:
        host: str
        port: int
        index: str
        user: str
        password: str

    @dataclass
    class QueryResult:
        score: float
        signature: BinarySignature

        @classmethod
        def from_dict(cls, source: dict):
            try:
                source["_score"]
            except Exception:
                breakpoint()
            return cls(
                score=source["_score"],
                signature=ElasticSearchHashRepository.dict2signature(source["_source"])
            )

    def __init__(self, config: ESHashRepoConfig):
        self.config = config
        http, http_auth = ("http", None) if not (config.user and config.password) \
            else ("https", (config.user, config.password))
        self.client = Elasticsearch(hosts=f"{http}://{self.config.host}:{self.config.port}", http_auth=http_auth)

    @classmethod
    def from_kwargs(cls, **kwargs):
        return cls(cls.ESHashRepoConfig(**kwargs))

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
                                "name": {"type": "keyword", "ignore_above": 256},
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

    def delete_index(self):
        if not self.index_exists:
            raise Exception(f"Index {self.config.index} does not exist")
        self.client.indices.delete(index=self.config.index)

    @staticmethod
    def signature2dict(signature: BinarySignature) -> dict:
        return {
            "filename": str(signature.path),
            "hash": str(base64.b64encode(signature.binary_hash)),
            "signature": signature.signature,
            "functions": [
                {
                    "name": f.function.name,
                    "cyclomatic_complexity": f.cyclomatic_complexity,
                    "num_instructions": f.num_instructions,
                    "num_strings": f.num_strings,
                    "max_string_length": f.max_string_length,
                    "instruction_histogram": f.instruction_histogram,
                    "edge_histogram": f.edge_histogram,
                    "vertex_histogram": f.vertex_histogram,
                    "dominator_signature": f.dominator_signature.asArray(),
                    "constants": f.constants.serialize(),
                    "strings": f.strings.serialize(),
                }
                for f in signature.functionFeatureList
            ],
            "extraction_engine": str(signature.extraction_engine),
        }

    @staticmethod
    def dict2signature(source: dict) -> BinarySignature:
        return BinarySignature(
            path=Path(source["filename"]),
            functionFeatureList=[
                FunctionFeatures.fromPrimitives(
                    cyclomatic_complexity=f["cyclomatic_complexity"],
                    num_instructions=f["num_instructions"],
                    num_strings=f["num_strings"],
                    max_string_length=f["max_string_length"],
                    instruction_histogram=f["instruction_histogram"],
                    edge_histogram=f["edge_histogram"],
                    vertex_histogram=f["vertex_histogram"],
                    dominator_signature=merge_uint32_to_int(f["dominator_signature"]),
                    constants=FunctionFeatures.Constants.deserialize(f["constants"]),
                    strings=FunctionFeatures.Strings.deserialize(f["strings"]),
                    extraction_engine=extractor_from_name(source["extraction_engine"]),
                    function=AbstractFunction(name=f["name"], _function=None)
                ) for f in source["functions"]
            ],
            extraction_engine=extractor_from_name(source["extraction_engine"])
        )

    def insert(self, signature: BinarySignature):
        self.client.index(
            index=self.config.index,
            body=self.signature2dict(signature)
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
                    strings=FunctionFeatures.Strings.deserialize(f["strings"])
                )
                for f in source["functions"]
            ],
        )

    def get(self, path: Path) -> Optional[BinarySignature]:
        raise NotImplementedError

    def getAll(self) -> List[QueryResult]:
        return [self.QueryResult.from_dict(hit) for hit in self.client.search(
            index=self.config.index,
            body={
                "query": {
                    "match_all": {}
                }
            },
            size=10000
        )["hits"]["hits"]]

    def match(
        self, signature: BinarySignature, threshold: int
    ) -> List[BinarySignature]:
        for func in signature.functionFeatureList:
            query = {
                "nested": {
                    "path": "functions",
                    "query": {
                        "knn": [
                            # TODO: Check if this auto-normalizes or if boost should account for size
                            # TODO: Perform actual analysis to determine boost values
                            {"field": "functions.cyclomatic_complexity",
                             "query_vector": func.cyclomatic_complexity,
                             "k": 5, "boost": 0.9},
                            {"field": "functions.num_instructions",
                             "query_vector": func.num_instructions,
                             "k": 5, "boost": 0.4},
                            {"field": "functions.num_strings",
                             "query_vector": func.num_strings,
                             "k": 5, "boost": 0.3},
                            {"field": "functions.max_string_length",
                             "query_vector": func.max_string_length,
                             "k": 5, "boost": 0.5},
                            {"field": "functions.instruction_histogram",
                             "query_vector": func.instruction_histogram,
                             "k": 5, "boost": 0.5},
                            {"field": "functions.edge_histogram",
                             "query_vector": func.edge_histogram,
                             "k": 5, "boost": 0.5},
                            {"field": "functions.dominator_signature",
                             "query_vector": func.dominator_signature.asArray(),
                             "k": 5, "boost": 0.7},
                            {"field": "functions.constants",
                             "query_vector": func.constants.serialize(),
                             "k": 5, "boost": 0.5},
                            {"field": "functions.strings",
                             "query_vector": func.strings.serialize(),
                             "k": 5, "boost": 0.9},
                        ]
                    },
                }
            }
            resp = self.client.search(
                index=self.config.index,
                body={
                    "query": query,
                    "min_score": threshold,
                },
            )

    def drop(self, option: Optional[Union[str, Path]] = None):
        raise NotImplementedError

    def summary(self, path_filter: str = ""):
        if path_filter != "":
            raise NotImplementedError
        return self.client.count(index=self.config.index)

    def find_closest_match(self, signature: BinarySignature) -> QueryResult:
        query = {
            "field": "signature",
            "query_vector": signature.signature,
            "k": 1,
            "num_candidates": 100,
        }
        resp = self.client.search(
            index=self.config.index,
            knn=query,
        )
        if resp["hits"]["total"]["value"] == 0:
            raise ValueError("Index is empty!")

        return self.QueryResult.from_dict(resp["hits"]["hits"][0])

    def __len__(self):
        raise NotImplementedError
