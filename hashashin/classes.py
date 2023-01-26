import logging
import os
from typing import Optional
from typing import Union
from typing import Annotated
import zlib
import json

from pathlib import Path
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from binaryninja import Function as BinaryNinjaFunction  # type: ignore
from binaryninja import BinaryView  # type: ignore
from abc import ABC
from hashashin.feature_extractors import FeatureExtractor

from hashashin.utils import func2str

logger = logging.getLogger(os.path.basename(__name__))


@dataclass
class AbstractFunction(ABC):
    name: str
    function: Union[BinaryNinjaFunction]


@dataclass
class BinjaFunction(AbstractFunction):
    name: str
    function: BinaryNinjaFunction

    @classmethod
    def fromFunctionRef(cls, function: BinaryNinjaFunction) -> "BinjaFunction":
        return cls(func2str(function), function)


@dataclass
class FunctionFeatures:
    extraction_engine: FeatureExtractor
    function: AbstractFunction
    cyclomatic_complexity: int
    num_instructions: int
    num_strings: int
    max_string_length: int
    constants: list[int]
    strings: list[str]
    instruction_histogram: list[int]
    dominator_signature: int
    vertex_histogram: list[int]
    edge_histogram: list[int]

    def asArray(self) -> npt.NDArray[np.uint32]:
        try:
            return np.array(
                [
                    self.cyclomatic_complexity,
                    self.num_instructions,
                    self.num_strings,
                    self.max_string_length,
                    self.dominator_signature,
                    *self.constants,
                    *self.strings,
                    *self.instruction_histogram,
                    *self.vertex_histogram,
                    *self.edge_histogram,
                ],
                dtype=np.uint32,
            )
        except ValueError as e:
            logger.error(f"Error while creating array for {self.function.name}")
            raise e

    def asBytes(self) -> bytes:
        return self.asArray().tobytes()

    @property
    def signature(self) -> bytes:
        return zlib.compress(self.asBytes())


@dataclass
class BinarySignature:
    SIGNATURE_LEN = 20
    path: Path
    functions: list[FunctionFeatures]
    extraction_engine: FeatureExtractor

    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist.")
        if not self.extraction_engine:
            self.extraction_engine = self.functions[0].extraction_engine
        if any(f.extraction_engine != self.extraction_engine for f in self.functions):
            raise ValueError("All functions must have the same extraction engine.")

    @classmethod
    def fromFile(cls, path: Path, extractor: FeatureExtractor) -> "BinarySignature":
        return extractor.extract_from_file(path)

    @staticmethod
    def min_hash(
        features: npt.NDArray[np.uint32], sig_len: int = SIGNATURE_LEN, seed: int = 2023
    ) -> npt.NDArray[np.uint32]:
        """
        Generate a minhash signature for a given set of features.

        :param features: a matrix of vectorized features
        :param sig_len: the length of the minhash signature
        :param seed: a seed for the random number generator
        :return: a string representing the minhash signature
        """
        random_state = np.random.RandomState(seed)
        a = random_state.randint(0, 2**32 - 1, size=sig_len)
        b = random_state.randint(0, 2**32 - 1, size=sig_len)
        c = 4297922131  # prime number above 2**32-1

        b = np.stack([np.stack([b] * features.shape[0])] * features.shape[1]).T
        # h(x) = (ax + b) % c
        hashed_features = (np.tensordot(a, features, axes=0) + b) % c
        minhash = hashed_features.min(axis=(1, 2))
        return minhash

    @property
    def np_signature(self) -> npt.NDArray[np.uint32]:
        return self.min_hash(np.array([f.asArray() for f in self.functions]))

    @property
    def signature(self) -> bytes:
        return zlib.compress(self.np_signature.tobytes())

    @staticmethod
    def jaccard_similarity(
        sig1: npt.NDArray[np.uint32], sig2: npt.NDArray[np.uint32]
    ) -> float:
        """
        Calculate the jaccard similarity between two minhash signatures.

        :param sig1: the first signature
        :param sig2: the second signature
        :return: the jaccard similarity between the two signatures
        """
        return np.sum(sig1 == sig2) / sig1.shape[0]

    def __sub__(self, other: "BinarySignature") -> float:
        return self.jaccard_similarity(self.np_signature, other.np_signature)
