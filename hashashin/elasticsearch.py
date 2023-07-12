from hashashin.db import AbstractHashRepository
from dataclasses import dataclass
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from hashashin.classes import BinarySignature
from hashashin.classes import BinarySigModel
from hashashin.classes import FunctionFeatModel
from hashashin.classes import extractor_from_name
from hashashin.classes import BinjaFeatureExtractor
from pathlib import Path
from typing import List, Optional, Union
from hashashin.classes import FunctionFeatures
from hashashin.classes import AbstractFunction
from hashashin.classes import merge_uint32_to_int
from typing import Tuple
import base64
import logging
import numpy as np
import pprint

logger = logging.getLogger(__name__)


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
        signature: Optional[BinarySignature]
        function: Optional[FunctionFeatures]

        @classmethod
        def from_hit(cls, hit: dict):
            if "_source" not in hit or "_score" not in hit:
                raise Exception("Invalid hit")
            if hit["_source"]["bin_fn_relation"]["name"] == "binary":
                binary_dict = hit["_source"]
                binary_dict["functions"] = list()
                for func in hit.get("inner_hits", {}).get("function", {}).get("hits", {}).get("hits", []):
                    if "_source" not in func:
                        raise Exception("Invalid function hit")
                    source = func["_source"]
                    if source.get("bin_fn_relation", {}).get("parent") != hit.get("_id"):
                        raise Exception("Invalid function hit")
                    binary_dict["functions"].append(source)
                if len(binary_dict["functions"]) != 0 and (
                    hit["inner_hits"]["function"]["hits"]["total"]["value"] != len(binary_dict["functions"])
                ):
                    # TODO: just go get the rest of them you lazy bum
                    raise ValueError("Pagination problem with inner_hits query. Increase size.")
                return cls(
                    score=hit["_score"],
                    signature=ElasticSearchHashRepository.dict2signature(binary_dict),
                    function=None
                )
            elif hit["_source"]["bin_fn_relation"]["name"] == "function":
                return cls(
                    score=hit["_score"],
                    signature=None,
                    function=ElasticSearchHashRepository.dict2function(hit["_source"], str(BinjaFeatureExtractor()))
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
        # TODO: add extraction engine
        if self.index_exists:
            raise ValueError(f"Index {self.config.index} already exists")
        body = {
            "settings": {
                "number_of_shards": shards,
                "number_of_replicas": replicas
            },
            "mappings": {
                "properties": {
                    "binary": {
                        "properties": {
                            "filename": {"type": "text"},
                            "hash": {"type": "keyword", "index": True},
                            "signature": {
                                "type": "dense_vector",
                                "dims": BinarySignature.SIGNATURE_LEN,
                                "index": True,
                                "similarity": "cosine",
                            },
                            "extraction_engine": {"type": "keyword"},
                        }
                    },
                    "function": {
                        "properties": {
                            "name": {"type": "keyword"},
                            "cyclomatic_complexity":
                                {"type": "dense_vector", "dims": 1, "index": True, "similarity": "l2_norm"},
                            "num_instructions":
                                {"type": "dense_vector", "dims": 1, "index": True, "similarity": "l2_norm"},
                            "num_strings":
                                {"type": "dense_vector", "dims": 1, "index": True, "similarity": "l2_norm"},
                            "max_string_length":
                                {"type": "dense_vector", "dims": 1, "index": True, "similarity": "l2_norm"},
                            "instruction_histogram": {
                                "type": "dense_vector",
                                "dims": FunctionFeatures.InstructionHistogram.length,
                                "index": True,
                                "similarity": "l2_norm",
                            },
                            "edge_histogram": {
                                "type": "dense_vector",
                                "dims": FunctionFeatures.EdgeHistogram.length,
                                "index": True,
                                "similarity": "l2_norm",
                            },
                            "vertex_histogram": {
                                "type": "dense_vector",
                                "dims": FunctionFeatures.VertexHistogram.length,
                                "index": True,
                                "similarity": "l2_norm",
                            },
                            "dominator_signature": {
                                "type": "dense_vector",
                                "dims": FunctionFeatures.DominatorSignature.length,
                                "index": True,
                                "similarity": "l2_norm",
                            },
                            "constants": {
                                "type": "keyword", "ignore_above": 256,
                            },
                            "strings": {
                                "type": "keyword", "ignore_above": 256,
                            },
                        }
                    },
                    "bin_fn_relation": {
                        "type": "join",
                        "relations": {
                            "binary": "function",
                        }
                    }
                }
            }
        }
        # Add properties to flat hierarchy for searching
        search_properties = [
            "signature", "cyclomatic_complexity", "num_instructions", "num_strings", "max_string_length",
            "instruction_histogram", "edge_histogram", "vertex_histogram", "dominator_signature"
        ]
        for prop in search_properties:
            if prop in body["mappings"]["properties"]:
                logger.warning(f"Property {prop} already exists in mapping, not overwriting")
                continue
            mapping = body["mappings"]["properties"]["binary"]["properties"].get(prop, {}) | body["mappings"]["properties"]["function"]["properties"].get(prop, {})
            body["mappings"]["properties"][prop] = mapping

        self.client.indices.create(
            index=self.config.index,
            body=body,
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
                    "cyclomatic_complexity": [f.cyclomatic_complexity],
                    "num_instructions": [f.num_instructions],
                    "num_strings": [f.num_strings],
                    "max_string_length": [f.max_string_length],
                    "instruction_histogram": list(f.instruction_histogram.asArray()),
                    "edge_histogram": f.edge_histogram,
                    "vertex_histogram": f.vertex_histogram,
                    "dominator_signature": list(f.dominator_signature.asArray()),
                    "constants": f.constants.serialize(),
                    "strings": f.strings.serialize(),
                }
                for f in signature.functionFeatureList
            ],
            "extraction_engine": str(signature.extraction_engine),
        }

    @staticmethod
    def dict2function(f: dict, engine: str) -> FunctionFeatures:
        return FunctionFeatures.fromPrimitives(
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
            extraction_engine=extractor_from_name(engine),
            function=AbstractFunction(name=f["name"], _function=None)
        )

    @staticmethod
    def dict2signature(source: dict) -> BinarySignature:
        return BinarySignature(
            path=Path(source["filename"]),
            cached_signature=np.array(source["signature"]),
            functionFeatureList=[
                ElasticSearchHashRepository.dict2function(f, source["extraction_engine"])
                for f in source["functions"]],
            extraction_engine=extractor_from_name(source["extraction_engine"])
        )

    def insert(self, signature: BinarySignature):
        logger.debug(f"Inserting {len(signature.functionFeatureList)} functions from {signature.path}")
        body = self.signature2dict(signature)
        functions = body.pop("functions")
        # insert binary
        resp = self.client.index(
            index=self.config.index,
            body=body | {"bin_fn_relation": {"name": "binary"}}
        )
        bin_id = resp["_id"]

        # for fdict in functions:
        #     q_body = {
        #         "index": self.config.index,
        #         "body": fdict | {"bin_fn_relation": {"name": "function", "parent": bin_id}},
        #         "routing": bin_id,
        #     }
        #     try:
        #         self.client.index(**q_body)
        #     except Exception as e:
        #         logger.error(e)
        #         breakpoint()
        #         print(e)
        #         raise e
        # insert functions
        bulk_requests = [{
            "_index": self.config.index,
            "_source": fdict | {"bin_fn_relation": {"name": "function", "parent": bin_id}},
            "_routing": bin_id,
            "_op_type": "index",
        } for fdict in functions]
        resp = helpers.bulk(self.client, bulk_requests)
        logger.debug(resp)

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

    def getAll(self, max_hits=10000) -> List[QueryResult]:
        return [self.QueryResult.from_hit(hit) for hit in self.client.search(
            index=self.config.index,
            body={
                "query": {
                    "has_child": {
                        "type": "function",
                        "query": {"match_all": {}},
                        "inner_hits": {
                            "from": 0,
                            "size": 100
                        },
                    }
                }
            },
            size=max_hits
        )["hits"]["hits"]]

    def match(
        self, signature: BinarySignature, num_matches: int
    ) -> List[QueryResult]:
        match_counts = dict()
        for func in signature.functionFeatureList:
            query = {
                "field": "dominator_signature",
                "query_vector": func.dominator_signature.asArray(),
                "k": 5,
                "num_candidates": 1000
            }
            resp = self.client.search(
                index=self.config.index,
                knn=query,
            )

            scores = dict()
            for hit in resp["hits"]["hits"]:
                scores[hit["_routing"]] = scores.get(hit["_routing"], 0) + hit["_score"]
            best_match = max(scores, key=scores.get)
            match_counts[best_match] = match_counts.get(best_match, 0) + 1

        logger.debug(f"Matched: {match_counts}")
        return match_counts

    def drop(self, option: Optional[Union[str, Path]] = None):
        if option is None:
            self.delete_index()
        else:
            raise NotImplementedError

    def summary(self, path_filter: str = "") -> Tuple[int, int]:
        if path_filter != "":
            raise NotImplementedError
        bin_count = self.client.count(index=self.config.index, q="bin_fn_relation:binary")['count']
        func_count = self.client.count(index=self.config.index, q="bin_fn_relation:function")['count']
        # func_count = sum([len(s.signature.functionFeatureList) for s in self.getAll()])
        return bin_count, func_count

    def find_closest_by_signature(self, signature: BinarySignature) -> QueryResult:
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

        return self.QueryResult.from_hit(resp["hits"]["hits"][0])

    def __len__(self):
        raise NotImplementedError
