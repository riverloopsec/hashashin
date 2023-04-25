import zlib
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Dict
from typing import Union
from typing import Callable

import numpy as np
import numpy.typing as npt
import xxhash
from binaryninja import BinaryView  # type: ignore
from binaryninja import Function as BinaryNinjaFunction  # type: ignore
from binaryninja import core_version
from binaryninja import enums
from binaryninja import open_view
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import LargeBinary
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.relationships import RelationshipProperty
from tqdm import tqdm

from hashashin.feature_extractors import EDGES
from hashashin.feature_extractors import VERTICES
from hashashin.feature_extractors import compute_constants
from hashashin.feature_extractors import compute_cyclomatic_complexity
from hashashin.feature_extractors import compute_dominator_signature
from hashashin.feature_extractors import compute_edge_taxonomy_histogram
from hashashin.feature_extractors import compute_instruction_histogram
from hashashin.feature_extractors import compute_vertex_taxonomy_histogram
from hashashin.feature_extractors import get_fn_strings
from hashashin.feature_extractors import BinjaTimeoutException
from hashashin.utils import merge_uint32_to_int
from hashashin.utils import split_int_to_uint32
import logging
import time

logger = logging.getLogger(__name__)
ORM_BASE: Any = declarative_base()
TLD = Path(__file__).parent.parent


class BinarySigModel(ORM_BASE):
    __tablename__ = "binaries"
    __allow_unmapped__ = True
    id = Column(Integer, primary_key=True)
    hash = Column(LargeBinary, unique=True, index=True)
    path = Column(String)
    sig = Column(LargeBinary, nullable=True)
    functions: RelationshipProperty = relationship("FunctionFeatModel", cascade="all, delete-orphan")
    extraction_engine = Column(String)

    @classmethod
    def fromBinarySignature(cls, sig: "BinarySignature") -> "BinarySigModel":
        try:
            path = str(sig.path.absolute().relative_to(Path(__file__).parent))
        except ValueError:
            path = str(sig.path)
        with open(sig.path, "rb") as f:
            return cls(
                hash=sig.binary_hash,
                path=path,
                sig=sig.signature,
                extraction_engine=str(sig.extraction_engine),
            )

    def __eq__(self, other) -> bool:
        if not isinstance(other, BinarySigModel):
            return False
        return (
                self.hash == other.hash
                and self.sig == other.sig
                and self.extraction_engine == other.extraction_engine
        )

    def __xor__(self, other):
        if isinstance(other, bytes):
            return bytes([a ^ b for a, b in zip(self.sig, other)])
        if isinstance(other, BinarySignature):
            return bytes([a ^ b for a, b in zip(self.sig, other.signature)])
        if isinstance(other, BinarySigModel):
            return bytes([a ^ b for a, b in zip(self.sig, other.sig)])
        raise TypeError(f"Cannot xor BinarySigModel with {type(other)}")

    def __repr__(self) -> str:
        return f"BinarySigModel({self.path}, {self.hash}, {self.sig}, {self.extraction_engine})"


class FunctionFeatModel(ORM_BASE):
    __tablename__ = "functions"
    __allow_unmapped__ = True
    id = Column(Integer, primary_key=True)
    bin_id = Column(Integer, ForeignKey("binaries.id", ondelete="CASCADE"))
    binary: RelationshipProperty = relationship(
        BinarySigModel, back_populates="functions"
    )
    name = Column(String)  # function name & address using func2str
    sig = Column(LargeBinary)  # zlib compression has variable size
    extraction_engine = Column(String)

    @classmethod
    def fromFunctionFeatures(cls, features: "FunctionFeatures") -> "FunctionFeatModel":
        if features.binary_id is None:
            # TODO: query the database for the binary_id
            raise ValueError("binary_id must be set")
        return cls(
            bin_id=features.binary_id,
            name=features.function.name,
            sig=features.signature,
            extraction_engine=str(features.extraction_engine),
        )

    def __xor__(self, other):
        if isinstance(other, bytes):
            return bytes([a ^ b for a, b in zip(zlib.decompress(self.sig), zlib.decompress(other))]).count(b'\x00')
        if isinstance(other, FunctionFeatures):
            return bytes([a ^ b for a, b in zip(zlib.decompress(self.sig), zlib.decompress(other.signature))]).count(
                b'\x00') / (FunctionFeatures.length * 4)
        if isinstance(other, FunctionFeatModel):
            return bytes([a ^ b for a, b in zip(zlib.decompress(self.sig), zlib.decompress(other.sig))]).count(
                b'\x00') / (FunctionFeatures.length * 4)
        raise TypeError(f"Cannot xor FunctionFeatModel with {type(other)}")


@dataclass
class AbstractFunction(ABC):
    name: str
    _function: Optional[BinaryNinjaFunction]
    path: Optional[Path] = None

    @property
    def function(self) -> BinaryNinjaFunction:
        if self._function is None:
            if self.path is None:
                raise ValueError("Path must be set")
            logger.debug(f"Loading {self.name} from bv")
            bv = open_view(self.path)
            self._function = bv.get_function_at(BinjaFunction.str2addr(self.name))
            if BinjaFunction.binja2str(self._function) != self.name:
                raise ValueError(f"Function {self.name} not found in {self.path}")
        return self._function


@dataclass
class BinjaFunction(AbstractFunction):
    # TODO: change from dataclass to regular class?
    @staticmethod
    def binja2str(function: BinaryNinjaFunction) -> str:
        return f"{function} @ 0x{function.start:X}"

    @staticmethod
    def str2addr(name: str) -> int:
        # TODO: handle function name without address
        return int(name.split("@")[1].strip(), 16)

    @classmethod
    def fromFunctionRef(cls, function: BinaryNinjaFunction) -> "BinjaFunction":
        return cls(name=cls.binja2str(function), _function=function)

    @classmethod
    def fromFile(cls, path: Path, name: str) -> "BinjaFunction":
        return cls(name=name, _function=None, path=path)


class FeatureExtractor:
    version: str

    def extract(self, func: AbstractFunction) -> "FunctionFeatures":
        raise NotImplementedError

    def extract_from_bv(self, bv: BinaryView, progress_kwargs=None) -> "BinarySignature":
        raise NotImplementedError

    def extract_from_file(
            self, path: Path, progress_kwargs: Optional[dict] = None
    ) -> "BinarySignature":
        raise NotImplementedError

    def extract_function(self, path_or_bv: Union[Path, BinaryView], function: Union[str, int]) -> "FunctionFeatures":
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.version})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, FeatureExtractor):
            return False
        return self.version == other.version

    def get_abstract_function(self, path: Path, name: str) -> AbstractFunction:
        raise NotImplementedError

    class NotABinaryError(Exception):
        pass


class BinjaFeatureExtractor(FeatureExtractor):
    version = core_version()

    def _extract(self, func: BinaryNinjaFunction) -> "FunctionFeatures":
        return FunctionFeatures(
            extraction_engine=self,
            function=BinjaFunction.fromFunctionRef(func),
            cyclomatic_complexity=compute_cyclomatic_complexity(func),
            num_instructions=len(list(func.instructions)),
            num_strings=len(get_fn_strings(func)),
            max_string_length=len(max(get_fn_strings(func), key=len, default="")),
            constants=sorted(compute_constants(func)),
            strings=sorted(get_fn_strings(func)),
            instruction_histogram=compute_instruction_histogram(func),
            dominator_signature=compute_dominator_signature(func),
            vertex_histogram=compute_vertex_taxonomy_histogram(func),
            edge_histogram=compute_edge_taxonomy_histogram(func),
        )

    def extract(self, function: AbstractFunction) -> "FunctionFeatures":
        """
        Extracts features from a function.
        :param function: function to extract features from
        :return: features
        """
        if not isinstance(function, (AbstractFunction, BinaryNinjaFunction)):
            raise ValueError(
                f"Expected Abstract or Binary Ninja function, got {type(function)}"
            )
        if isinstance(function, BinaryNinjaFunction):
            func: BinaryNinjaFunction = function
        else:
            func = function.function
        return self._extract(func)

    def _extract_functions(self, bv: BinaryView, progress_kwargs: Dict) -> Iterable["FunctionFeatures"]:
        disable = progress_kwargs.pop("disable", False)
        pbar = tqdm(bv.functions, disable=disable)  # type: ignore
        for func in pbar:
            if "desc" in progress_kwargs and callable(progress_kwargs["desc"]):
                pbar.set_description(progress_kwargs["desc"](func))  # type: ignore
            elif "desc" in progress_kwargs:
                pbar.set_description(progress_kwargs["desc"])  # type: ignore
            else:
                pbar.set_description(f"Extracting features from {func}")  # type: ignore
            yield self._extract(func)

    def extract_from_bv(self, bv: BinaryView, progress_kwargs: Optional[Dict] = None) -> "BinarySignature":
        """
        Extracts features from all functions in a binary.
        :param bv: binary view
        :param progress_kwargs: optionally pass tqdm kwargs to show progress.
                If progress_kwargs["desc"] is callable, it will be called with the current function
        :return: list of features
        """
        progress_kwargs = (
            {"disable": "True"} if progress_kwargs is None else progress_kwargs
        )
        if Path(bv.file.filename).suffix == ".bndb":
            path = Path(bv.file.filename).with_suffix("")
        else:
            path = Path(bv.file.filename)
        return BinarySignature(
            path=path,
            functionFeatureList=list(self._extract_functions(bv, progress_kwargs)),
            extraction_engine=self,
        )

    @staticmethod
    def progress(cur, total):
        if cur < total:
            print(f"{cur}/{total}", end="\r")
        else:
            print(f"{cur}/{total}")
        return True

    @staticmethod
    def open_view(path: Path, **kwargs) -> BinaryView:
        if path.with_suffix(path.suffix + ".bndb").exists():
            bv_path = path.with_suffix(path.suffix + ".bndb")
            logger.debug(f"Found bndb for {path}: {bv_path.absolute()}")
        else:
            bv_path = path
        logger.debug(f"Opening {bv_path.absolute()} with Binary Ninja")
        update_analysis = kwargs.pop("update_analysis", False)
        bv = open_view(bv_path, update_analysis=update_analysis, **kwargs)
        if not update_analysis:
            timeout = kwargs.pop("timeout", 300)
            bv.update_analysis()
            start = time.time()
            while (
                    time.time() - start < timeout and
                    bv.analysis_progress.state != enums.AnalysisState.IdleState
            ):
                if logger.getEffectiveLevel() <= logging.DEBUG:
                    print(f"{(str(bv.analysis_progress) + '...').ljust(40)}\t{time.time() - start:.1f}/{timeout}s", end="\r")
                time.sleep(0.1)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                print()
            if bv.analysis_progress.state != enums.AnalysisState.IdleState:
                raise TimeoutError(f"Analysis timed out after {timeout}s")
            if time.time() - start > 30:
                logger.debug(f"Analysis took {time.time() - start:.1f}s, caching bndb")
                bv.create_database(f"{bv.file.filename}.bndb")
        return bv

    def extract_from_file(self, path: Path, progress_kwargs=None) -> "BinarySignature":
        """
        Extracts features from all functions in a binary.
        :param path: path to binary
        :param progress_kwargs: optionally pass tqdm kwargs to show progress
        :return: list of features
        """
        if not path.is_file():
            raise FileNotFoundError(f"File {path} does not exist")
        progress_kwargs = (
            {"disable": "True"} if progress_kwargs is None else progress_kwargs
        )
        with self.open_view(
                path,
                # progress_func=self.progress,
        ) as bv:
            bs = self.extract_from_bv(bv, progress_kwargs)
            if not bs.functionFeatureList:
                raise self.NotABinaryError(f"No functions found in {path}")
            return bs

    def extract_function(self, path_or_bv: Union[Path, BinaryView], function: Union[str, int]) -> "FunctionFeatures":
        """
        Extracts features from a single function in a binary.
        :param path_or_bv: path to binary or loaded bv
        :param function: name of function or address to function start
        :return: FunctionFeatures
        """
        if isinstance(path_or_bv, Path):
            with self.open_view(path_or_bv) as _bv:
                return self.extract_function(_bv, function)
        elif not isinstance(path_or_bv, BinaryView):
            raise ValueError(f"Expected path or BinaryView, got {type(path_or_bv)}")
        bv: BinaryView = path_or_bv
        if isinstance(function, str):
            if function.upper().startswith("0X"):
                func = bv.get_function_at(int(function, 16))
            else:
                fns = bv.get_functions_by_name(function)
                if len(fns) != 1:
                    raise ValueError(f"Found {len(fns)} functions for {function}")
                func = fns[0]
        else:
            func = bv.get_function_at(function)
        if func is None:
            raise ValueError(f"Could not find function {function} in {path_or_bv}")
        logger.debug(f"Extracting features from {func} in {bv.file.filename}...")
        return self._extract(func)

    def get_abstract_function(self, path: Path, name: str) -> AbstractFunction:
        return BinjaFunction.fromFile(path, name)


def extractor_from_name(name: str) -> FeatureExtractor:
    if name.startswith(BinjaFeatureExtractor.__name__):
        extractor = BinjaFeatureExtractor()
        if name == str(extractor):
            return extractor
        raise ValueError(f"Version mismatch, expected {extractor.version}, got {name}")
    raise ValueError(f"Unknown extractor {name}")


@dataclass
class FunctionFeatures:
    class VertexHistogram(list):
        length = VERTICES

    class EdgeHistogram(list):
        length = EDGES

    class InstructionHistogram(list):
        length = len(enums.MediumLevelILOperation.__members__)

        def __repr__(self):
            return "|".join(str(x) for x in self)

    class DominatorSignature(int):
        length = 32

        def __new__(cls, *args, **kwargs):
            if type(args[0]) == np.ndarray:
                return super().__new__(cls, merge_uint32_to_int(args[0]))
            return super().__new__(cls, *args, **kwargs)

        def __repr__(self):
            return hex(self)

        def asArray(self):
            x = split_int_to_uint32(self, pad=self.length, wrap=True)
            # if int(np.ceil(len(bin(self)) / 32)) > self.length:
            if self > 2 ** (self.length * 32):
                logger.debug(
                    f"Dominator signature too long, truncating {hex(self)} -> {hex(merge_uint32_to_int(x))}"
                )
                logger.dominator_warning = True
            return x

    class Constants(list):
        length = 64

        def asArray(self):
            return sorted(self[: len(self)])[: min(len(self), self.length)] + [0] * max(
                0, self.length - len(self)
            )

    class Strings(list):
        length = 512

        def __init__(self, strings):
            if type(strings) == np.ndarray:
                return super().__init__(self.fromArray(strings))
            return super().__init__(strings)

        @staticmethod
        def fromArray(array: np.ndarray) -> Iterable[str]:
            if 'strings_warning' not in dir(logger):
                logger.debug("There is a bug here, strings are not displayed properly")
                logger.strings_warning = True  # type: ignore
            if len(array) != FunctionFeatures.Strings.length:
                raise ValueError(
                    f"Expected array of length {FunctionFeatures.Strings.length}, got {len(array)}"
                )
            r = array.tobytes().decode("utf-8").rstrip("\0").split("\0")
            if len(r) > 1:
                pass
            return r

        def asArray(self) -> Iterable[int]:
            if len(self) == 0:
                return [0] * self.length
            if "warned" not in dir(logger):
                logger.warning(
                    "Wasting space here, can shorten array by 256 bytes by using uint32"
                )
                logger.warned = True  # type: ignore
            strings = "\0".join(sorted(self[: len(self)])).encode("utf-8")
            strings = strings[: min(len(strings), self.length)]
            return np.pad(
                np.frombuffer(strings, dtype=np.byte),
                (0, self.length - len(strings)),
                "constant",
            )

    extraction_engine: FeatureExtractor
    function: AbstractFunction
    cyclomatic_complexity: int
    num_instructions: int
    num_strings: int
    max_string_length: int
    constants: Constants
    strings: Strings
    instruction_histogram: InstructionHistogram
    dominator_signature: DominatorSignature
    vertex_histogram: VertexHistogram
    edge_histogram: EdgeHistogram
    length = 4 + sum(
        [
            x.length
            for x in [
            Constants,
            Strings,
            InstructionHistogram,
            DominatorSignature,
            VertexHistogram,
            EdgeHistogram,
        ]
        ]
    )
    # Reference used to set the foreign key in the database
    binary_id: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.constants, self.Constants):
            self.constants = self.Constants(self.constants)
        if not isinstance(self.strings, self.Strings):
            self.strings = self.Strings(self.strings)
        if not isinstance(self.instruction_histogram, self.InstructionHistogram):
            self.instruction_histogram = self.InstructionHistogram(
                self.instruction_histogram
            )
        if not isinstance(self.dominator_signature, self.DominatorSignature):
            self.dominator_signature = self.DominatorSignature(self.dominator_signature)
        if not isinstance(self.vertex_histogram, self.VertexHistogram):
            self.vertex_histogram = self.VertexHistogram(self.vertex_histogram)
        if not isinstance(self.edge_histogram, self.EdgeHistogram):
            self.edge_histogram = self.EdgeHistogram(self.edge_histogram)

    def asArray(self) -> npt.NDArray[np.uint32]:
        try:
            logger.dominator_warning = False
            array = np.array(
                [
                    self.cyclomatic_complexity,
                    self.num_instructions,
                    self.num_strings,
                    self.max_string_length,
                    *self.vertex_histogram,
                    *self.edge_histogram,
                    *self.instruction_histogram,
                    *self.dominator_signature.asArray(),
                    *self.constants.asArray(),
                    *self.strings.asArray(),
                ],
                dtype=np.uint32,
            )
            if logger.dominator_warning:
                logger.debug(
                    f"Truncated dominator signature for {str(self.function.path)}: {self.function.name}))"
                )
            logger.dominator_warning = False
            if len(array) != self.length:
                for field in (
                        self.vertex_histogram,
                        self.edge_histogram,
                        self.instruction_histogram,
                        self.dominator_signature,
                        self.constants,
                        self.strings,
                ):
                    arr = field if issubclass(type(field), list) else field.asArray()
                    logger.error(
                        f"{type(field)} expected {field.length} got {len(arr)}"
                    )
                raise ValueError(
                    f"Something went wrong, expected {self.length} got {len(array)}"
                )
            return array
        except Exception as e:
            breakpoint()
            logger.error(f"Error while creating array for {self.function.name}")
            raise e

    @classmethod
    def fromArray(
            cls,
            array: npt.NDArray[np.uint32],
            name: str,
            path: Path,
            extraction_engine: FeatureExtractor,
    ) -> "FunctionFeatures":
        if not len(array) == cls.length:
            raise ValueError(
                f"Wrong array length {len(array)} != {cls.length} for {function.name}"
            )
        return cls(
            extraction_engine=extraction_engine,
            function=extraction_engine.get_abstract_function(path, name),
            cyclomatic_complexity=array[0],
            num_instructions=array[1],
            num_strings=array[2],
            max_string_length=array[3],
            vertex_histogram=FunctionFeatures.VertexHistogram(
                array[4: 4 + FunctionFeatures.VertexHistogram.length]
            ),
            edge_histogram=FunctionFeatures.EdgeHistogram(
                array[
                4
                + FunctionFeatures.VertexHistogram.length: 4
                                                           + FunctionFeatures.VertexHistogram.length
                                                           + FunctionFeatures.EdgeHistogram.length
                ]
            ),
            instruction_histogram=FunctionFeatures.InstructionHistogram(
                array[
                4
                + FunctionFeatures.VertexHistogram.length
                + FunctionFeatures.EdgeHistogram.length: 4
                                                         + FunctionFeatures.VertexHistogram.length
                                                         + FunctionFeatures.EdgeHistogram.length
                                                         + FunctionFeatures.InstructionHistogram.length
                ]
            ),
            dominator_signature=FunctionFeatures.DominatorSignature(
                array[
                4
                + FunctionFeatures.VertexHistogram.length
                + FunctionFeatures.EdgeHistogram.length
                + FunctionFeatures.InstructionHistogram.length: 4
                                                                + FunctionFeatures.VertexHistogram.length
                                                                + FunctionFeatures.EdgeHistogram.length
                                                                + FunctionFeatures.InstructionHistogram.length
                                                                + FunctionFeatures.DominatorSignature.length
                ]
            ),
            constants=FunctionFeatures.Constants(
                array[
                4
                + FunctionFeatures.VertexHistogram.length
                + FunctionFeatures.EdgeHistogram.length
                + FunctionFeatures.InstructionHistogram.length
                + FunctionFeatures.DominatorSignature.length: 4
                                                              + FunctionFeatures.VertexHistogram.length
                                                              + FunctionFeatures.EdgeHistogram.length
                                                              + FunctionFeatures.InstructionHistogram.length
                                                              + FunctionFeatures.DominatorSignature.length
                                                              + FunctionFeatures.Constants.length
                ]
            ),
            strings=FunctionFeatures.Strings(
                array[
                4
                + FunctionFeatures.VertexHistogram.length
                + FunctionFeatures.EdgeHistogram.length
                + FunctionFeatures.InstructionHistogram.length
                + FunctionFeatures.DominatorSignature.length
                + FunctionFeatures.Constants.length: 4
                                                     + FunctionFeatures.VertexHistogram.length
                                                     + FunctionFeatures.EdgeHistogram.length
                                                     + FunctionFeatures.InstructionHistogram.length
                                                     + FunctionFeatures.DominatorSignature.length
                                                     + FunctionFeatures.Constants.length
                                                     + FunctionFeatures.Strings.length
                ]
            ),
        )

    @classmethod
    def fromModel(cls, model: FunctionFeatModel) -> "FunctionFeatures":
        return cls.fromArray(
            np.frombuffer(zlib.decompress(model.sig), dtype=np.uint32),
            model.name,
            model.binary.path,
            extractor_from_name(model.extraction_engine),
        )

    def asBytes(self) -> bytes:
        return self.asArray().tobytes()

    @property
    def signature(self) -> bytes:
        return zlib.compress(self.asBytes())

    def get_feature_dict(self) -> Dict[str, Any]:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "num_instructions": self.num_instructions,
            "num_strings": self.num_strings,
            "max_string_length": self.max_string_length,
            "vertex_histogram": self.vertex_histogram,
            "edge_histogram": self.edge_histogram,
            "instruction_histogram": self.instruction_histogram,
            "dominator_signature": self.dominator_signature,
            "constants": self.constants,
            "strings": self.strings,
        }

    def __hash__(self) -> int:
        return hash(self.signature)

    def __sub__(self, other):
        if not isinstance(other, FunctionFeatures):
            raise TypeError(f"Cannot subtract {type(self)} with {type(other)}")
        return np.linalg.norm(self.asArray() - other.asArray())

    def __xor__(self, other):
        if not isinstance(other, FunctionFeatures):
            raise TypeError(f"Cannot xor {type(self)} with {type(other)}")
        return bytes([a ^ b for a, b in zip(self.asBytes(), other.asBytes())]).count(b'\x00') / (self.length * 4)


@dataclass
class BinarySignature:
    SIGNATURE_LEN = 20
    path: Path
    functionFeatureList: List[FunctionFeatures]
    extraction_engine: FeatureExtractor
    cached_signature: Optional[bytes] = None
    cached_array: Optional[np.ndarray] = None

    def __post_init__(self):
        for i in range(len(self.path.parts)):
            if Path(*self.path.parts[i:]).exists():
                self.path = Path(*self.path.parts[i:])
                break
        if not self.path.exists() and not self.path.with_suffix(".bndb").exists():
            globsearch = list(TLD.glob(f"**/{self.path}"))
            logger.debug(f"Can't find path {self.path}, globbed:\n{globsearch}")
            if len(globsearch) == 1 and globsearch[0].exists():
                self.path = globsearch[0]
            else:
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

    @classmethod
    def fromModel(cls, model: BinarySigModel) -> "BinarySignature":
        return cls(
            Path(model.path),
            [FunctionFeatures.fromModel(f) for f in model.functions],
            extractor_from_name(model.extraction_engine),
        )

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
        a = random_state.randint(0, 2 ** 32 - 1, size=sig_len)
        b = random_state.randint(0, 2 ** 32 - 1, size=sig_len)
        c = 4297922131  # prime number above 2**32-1

        b = np.stack([np.stack([b] * features.shape[0])] * features.shape[1]).T
        # h(x) = (ax + b) % c
        hashed_features = (np.tensordot(a, features, axes=0) + b) % c
        minhash = hashed_features.min(axis=(1, 2))
        return minhash

    @property
    def np_signature(self) -> npt.NDArray[np.uint32]:
        if self.cached_array is None:
            self.cached_array = self.min_hash(
                np.array([f.asArray() for f in self.functionFeatureList])
            )
        else:
            logger.debug("Using cached np signature.")
        return self.cached_array

    @property
    def signature(self) -> bytes:
        if self.cached_signature is None:
            self.cached_signature = zlib.compress(self.np_signature.tobytes())
        else:
            logger.debug("Using cached signature.")
        return self.cached_signature

    @property
    def function_matrix(self) -> npt.NDArray[np.uint32]:
        return np.array([f.asArray() for f in self.functionFeatureList])

    @staticmethod
    def hash_file(path: Path) -> bytes:
        with open(path, "rb") as f:
            return xxhash.xxh64(f.read()).digest()

    @property
    def binary_hash(self) -> bytes:
        return self.hash_file(self.path)

    def get_function_features(self, func_name: Optional[str], func_addr: Optional[int]) -> Optional[FunctionFeatures]:
        if func_name:
            for f in self.functionFeatureList:
                if func_name in f.function.name:
                    return f
        elif func_addr:
            for f in self.functionFeatureList:
                if f.function.function.start == func_addr:
                    return f
        return None

    @staticmethod
    def minhash_similarity(
            sig1: npt.NDArray[np.uint32], sig2: npt.NDArray[np.uint32]
    ) -> float:
        """
        Calculate the similarity between two minhash signatures.

        :param sig1: the first signature
        :param sig2: the second signature
        :return: the hamming distance between the two signatures
        """
        return np.sum(sig1 == sig2) / sig1.shape[0]

    @staticmethod
    def jaccard_similarity(
            sig1: Set[FunctionFeatures], sig2: Set[FunctionFeatures]
    ) -> float:
        """
        Calculate the feature distance between two signatures.

        :param sig1: the first signature
        :param sig2: the second signature
        :return: the distance between the two signatures
        """
        if isinstance(sig1, list):
            sig1 = set(sig1)
        if isinstance(sig2, list):
            sig2 = set(sig2)
        return len(sig1.intersection(sig2)) / len(sig1.union(sig2))

    @staticmethod
    def jaccard_estimate(sig1: Set[bytes], sig2: Set[bytes]) -> float:
        """
        Calculate the feature distance between two signatures.

        :param sig1: the first signature
        :param sig2: the second signature
        :return: the distance between the two signatures
        """
        if isinstance(sig1, list):
            sig1 = set(sig1)
        if isinstance(sig2, list):
            sig2 = set(sig2)
        return len(sig1.intersection(sig2)) / len(sig1.union(sig2))

    def __floordiv__(self, other: "BinarySignature") -> float:
        if not isinstance(other, BinarySignature):
            raise TypeError(
                f"Cannot divide {type(other)} from {type(self)} (expected BinarySignature)"
            )
        return self.minhash_similarity(self.np_signature, other.np_signature)

    def __sub__(self, other: "BinarySignature") -> float:
        if not isinstance(other, BinarySignature):
            raise TypeError(
                f"Cannot subtract {type(other)} from {type(self)} (expected BinarySignature)"
            )
        return self.jaccard_similarity(
            set(self.functionFeatureList), set(other.functionFeatureList)
        )

    def __xor__(self, other) -> float:
        if isinstance(other, bytes):
            return bytes([a ^ b for a, b in zip(self.signature, other)]).count(b'\x00') / len(self.signature)
        if isinstance(other, BinarySignature):
            return bytes([a ^ b for a, b in zip(self.signature, other.signature)]).count(b'\x00') / len(self.signature)
        if isinstance(other, BinarySigModel):
            return bytes([a ^ b for a, b in zip(self.signature, other.sig)]).count(b'\x00') / len(self.signature)
        raise TypeError(f"Cannot xor BinarySigModel with {type(other)}")
