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
from hashashin.utils import split_int_to_uint32

logger = logging.getLogger(os.path.basename(__name__))


@dataclass
class AbstractFunction(ABC):
    name: str
    function: BinaryNinjaFunction  # Union[BinaryNinjaFunction, GhidraFunction]


@dataclass
class BinjaFunction(AbstractFunction):
    # TODO: change from dataclass to regular class
    name: str
    function: BinaryNinjaFunction

    @staticmethod
    def binja2str(function: BinaryNinjaFunction) -> str:
        return f"{function} @ 0x{function.start:X}"

    @classmethod
    def fromFunctionRef(cls, function: BinaryNinjaFunction) -> "BinjaFunction":
        return cls(cls.binja2str(function), function)


class FeatureExtractor:
    version: str

    def extract(self, func: AbstractFunction) -> "FunctionFeatures":
        raise NotImplementedError

    def extract_from_file(self, path: Path) -> "BinarySignature":
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.version})"


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
    # Reference used to set the foreign key in the database
    binary_id: Optional[int] = None

    def _strings_to_array(self) -> np.ndarray:
        logger.debug(
            "Wasting space here, can shorten array by 256 bytes by using uint32"
        )
        strings = "\0".join(sorted(self.strings)).encode("utf-8")
        strings = strings[: min(len(strings), 512)]
        return np.pad(
            np.frombuffer(strings, dtype=np.byte), (0, 512 - len(strings)), "constant"
        )

    def asArray(self) -> npt.NDArray[np.uint32]:
        try:
            return np.array(
                [
                    self.cyclomatic_complexity,
                    self.num_instructions,
                    self.num_strings,
                    self.max_string_length,
                    *self.vertex_histogram,
                    *self.edge_histogram,
                    *self.instruction_histogram,
                    *split_int_to_uint32(self.dominator_signature, pad=32, wrap=True),
                    *(
                        sorted(self.constants)[: min(len(self.constants), 64)]
                        + [0] * max(0, 64 - len(self.constants))
                    ),
                    *self._strings_to_array(),
                ],
                dtype=np.uint32,
            )
        except Exception as e:
            breakpoint()
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
    functionFeatureList: list[FunctionFeatures]
    extraction_engine: FeatureExtractor

    def __post_init__(self):
        if not self.path.exists():
            raise FileNotFoundError(f"File {self.path} does not exist.")
        if not self.extraction_engine:
            self.extraction_engine = self.functionFeatureList[0].extraction_engine
        if any(
            f.extraction_engine != self.extraction_engine
            for f in self.functionFeatureList
        ):
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
        try:
            return self.min_hash(
                np.array([f.asArray() for f in self.functionFeatureList])
            )
        except Exception as e:
            breakpoint()
            logger.error(f"Error while creating signature for {self.path}")
            raise e

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
