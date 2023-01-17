import logging
import os
from typing import Optional
from typing import Union
from typing import Annotated
import zlib
import json

import numpy as np
import numpy.typing as npt
from binaryninja import Function  # type: ignore

from hashashin.feature_extractors import FEATURES
from hashashin.utils import func2str

logger = logging.getLogger(os.path.basename(__name__))


class BinSig:
    """Class to represent the signature of an entire binary."""
    SIGNATURE_LEN = 20
    TYPE = Annotated[npt.NDArray[np.int64], SIGNATURE_LEN]

    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, BinSig):
                return {
                    o.sig: o.serialized
                }
            return json.JSONEncoder.default(self, o)

    @staticmethod
    def deserialize(serialized: str) -> TYPE:
        compressed_data = bytes.fromhex(serialized)
        data = zlib.decompress(compressed_data)
        return np.frombuffer(data, dtype=BinSig.TYPE)

    def __init__(self, sig: Union[TYPE, str]):
        self.sig: BinSig.TYPE
        if isinstance(sig, str):
            self.sig = BinSig.deserialize(sig)
        elif isinstance(sig, np.ndarray):
            self.sig = sig
        else:
            raise ValueError(f"Signature must be {str} or {BinSig.TYPE}, got {type(sig)}.")
        self._check_sig()

    def _check_sig(self):
        if (
                self.sig.shape != FuncSig.TYPE.shape
                or self.sig.dtype != FuncSig.TYPE.dtype
        ):
            raise ValueError(
                f"Signature must be a {FuncSig.TYPE.dtype} array of shape {FuncSig.TYPE.shape}, "
                f"not {self.sig.dtype} array of shape {self.sig.shape}."
            )


    @property
    def serialized(self) -> str:
        return zlib.compress(self.sig.tobytes()).hex()

    def __eq__(self, other):
        return all(self.sig == other.sig)

    def distance(self, other):
        return np.linalg.norm(self.sig - other.sig)

    def __sub__(self, other):
        return self.distance(other)


class FuncSig:
    """Class to represent the function signature generated from feature extraction."""
    TYPE = Annotated[npt.NDArray[np.uint32], (sum(f.size for f in FEATURES),)]

    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, FuncSig):
                return {
                    o.func: o.serialized
                }
            return json.JSONEncoder.default(self, o)

    @staticmethod
    def deserialize(serialized: str) -> TYPE:
        try:
            hex_str = zlib.decompress(bytes.fromhex(serialized)).decode()
        except zlib.error as e:
            raise ValueError(f"Could not decompress {serialized}: {e}")
        return np.array([int(hex_str[i: i + 8], 16) for i in range(0, len(hex_str), 8)], dtype=FuncSig.TYPE)

    def __init__(
        self, func: Union[Function, str], features: Optional[Union[TYPE, str]]
    ):
        """Initialize a function signature.

        Args:
            func: The function to generate a signature for.
            features: The feature vector representing the signature.
                If str, it must be a zlib-compressed hex string.
        """
        if isinstance(func, Function):
            self.func = func2str(func)
        elif isinstance(func, str):
            self.func = func
        else:
            raise ValueError(f"func must be a Function or str, not {type(func)}")
        self.features: Optional[FuncSig.TYPE]
        if isinstance(features, str):
            features = self.deserialize(features)
        self.features = features
        # TODO: remove check
        assert self.features is None or self.features.dtype == np.uint32 and self.features.size == sum(f.size for f in FEATURES)
        self._check_features()

    def _check_features(self):
        if self.features is None:
            logger.debug("Creating an empty feature vector.")
            self.features = np.zeros_like(FuncSig.TYPE)
            # TODO: remove check
            assert self.features.dtype == np.uint32 and self.features.size == sum(f.size for f in FEATURES)
            return
        if (
            self.features.shape != FuncSig.TYPE.shape
            or self.features.dtype != FuncSig.TYPE.dtype
        ):
            raise ValueError(
                f"Features must be a {FuncSig.TYPE.dtype} array of shape {FuncSig.TYPE.shape}, "
                f"not {self.features.dtype} array of shape {self.features.shape}."
            )

    @property
    def serialized(self):
        return zlib.compress(self.features.tobytes()).hex()

    def distance(self, other):
        return np.linalg.norm(self.features - other.features)

    def __sub__(self, other):
        return self.distance(other)

    def __iter__(self):
        i = 0
        for f in FEATURES:
            yield f.repr(self.features[i: i + f.size])
            i += f.size
    
    def __str__(self):
        return json.dumps(self, cls=FuncSig.CustomEncoder)
    
    def __repr__(self):
        return f"{self.func}: {self}"
