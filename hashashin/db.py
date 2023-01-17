import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

from binaryninja import Function  # type: ignore

from hashashin.types import BinSig
from hashashin.types import FuncSig
from hashashin.utils import decode_feature
from hashashin.utils import func2str
from hashashin.utils import vec2hex

logger = logging.getLogger(os.path.basename(__name__))

HASH_EXT = ".hash.json"
SIG_DB_PATH = Path(__file__).parent / "sig_db.csv"
SIG_DB_HEADER = ("sig", "path")


class SigDB:
    """Flat DB for storing binary signatures and their respective function hashes.
    If a signature is added to the DB, by convention all function signatures are also added.
    These function signatures (features) are stored at {binary_path}{HASH_EXT}.
    You can choose to either load all function signatures from disk at once or lazily.
    """

    def __init__(self, db_path: Path = SIG_DB_PATH, db_header: tuple = SIG_DB_HEADER, lazy: bool = True):
        self.db_path = db_path
        self.db_header = db_header
        # {BinSig: {binary_path, ...}}
        self.sig_map: dict[BinSig, set[Path]] = {}
        # {binary_path: {func_str: FuncSig, ...}}
        self.func_map: dict[Path, dict[str, FuncSig]] = {}
        self._load(lazy)

    def _load_functions(self, bin_path: Path) -> BinSig:
        """Load function signatures from {bin_path}{HASH_EXT}, return saved binary signature."""
        if bin_path in self.func_map and len(self.func_map[bin_path]) > 0:
            sig = self.getBinSig(bin_path)
            if not sig:
                raise ValueError(f"Signature for {bin_path} not found.")
            return sig
        hash_path = bin_path.with_suffix(HASH_EXT)
        if not hash_path.exists():
            raise FileNotFoundError(f"Hash file {hash_path} does not exist.")
        with open(hash_path, "r") as f:
            binhash = json.load(f)
        self.func_map[bin_path] = {
            func: FuncSig(func, decode_feature(feature))
            for func, feature in binhash["features"].items()
        }
        return BinSig(binhash["signature"])

    def _load(self, lazy_load: bool = True):
        """Load the signature DB & associated function features from disk."""
        if not os.path.exists(self.db_path):
            logger.warning(f"Hash DB not found at {self.db_path}")
            with open(self.db_path, "w") as f:
                f.write(",".join(self.db_header))
            return
        with open(self.db_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sig, path = BinSig(row["sig"]), Path(row["path"])
                if not lazy_load and self._load_functions(path) != sig:
                    raise ValueError(
                        f"Signature mismatch for {path}. DB is likely stale."
                    )
                self.sig_map[sig] = self.sig_map.get(sig, set()).union({path})

    @staticmethod
    def _check_path(filepath: Union[Path, str]) -> Path:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Binary {filepath} does not exist.")
        return filepath.resolve()

    def add(
        self,
        filepath: Union[Path, str],
        sig: BinSig,
        features: dict[Union[Function, str], FuncSig],
    ):
        """Add a binary hash (signature & function features) to the DB."""
        filepath = self._check_path(filepath)
        # Convert function keys to strings
        features = {func2str(k) if isinstance(k, Function) else k: v for k, v in features.items()}
        if sig in self.sig_map and filepath in self.sig_map[sig]:
            if self.func_map[filepath] != features:
                raise ValueError(f"Features mismatch for {filepath}.")
            logger.warning(f"Signature & filepath already in DB.")
            return
        self.sig_map[sig] = self.sig_map.get(sig, set()).union({filepath})
        with open(self.db_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.db_header)
            writer.writerow({"sig": sig, "path": filepath})
        self.func_map[filepath] = features
        with open(filepath.with_suffix(HASH_EXT), "w") as f:
            json.dump(
                {"signature": sig, "features": features},
                f,
                indent=4,
                sort_keys=True,
                cls=CustomEncoder,
            )

    def delete(self, filepath: Union[Path, str]):
        if all(filepath not in x for x in self.sig_map.values()):
            logger.warning(f"Signature for {filepath} does not exist in DB.")
        raise NotImplementedError

    def getBinPath(self, sig: BinSig) -> set[Path]:
        """Return the set of binaries that match the given signature."""
        return self.sig_map.get(sig, set())

    def getBinSig(self, bin_path: Union[Path, str]) -> Optional[BinSig]:
        """Return the signature of the given binary."""
        bin_path = self._check_path(bin_path)
        ret = None
        for sig, paths in self.sig_map.items():
            if bin_path in paths:
                if ret is None:
                    ret = sig
                else:
                    raise ValueError(f"Multiple signatures found for {bin_path}.")
        return ret

    def getFuncSig(self, func: Function) -> Optional[FuncSig]:
        """Return the function signature of the given function."""
        raise NotImplementedError

    def getClosestSig(self, sig: BinSig) -> BinSig:
        """Return the closest saved signature to the given signature."""
        return min(self.sig_map.keys(), key=lambda x: x.distance(sig))


# class Hasher:
#     def __init__(self, bv_or_file: Union[str, BinaryView]):
#         if isinstance(bv_or_file, str):
#             if not os.path.exists(bv_or_file):
#                 raise FileNotFoundError(f"File {bv_or_file} not found.")
#             self.bv = open_view(bv_or_file)
#         elif isinstance(bv_or_file, BinaryView):
#             self.bv = bv_or_file
#         else:
#             raise ValueError("bv_or_file must be a string or a BinaryView")
#
#     def hash_function(function: Function) -> FuncSig:
#         """
#         Hash a given function by "bucketing" basic blocks to capture a high level overview of their functionality, then
#         performing a variation of the Weisfeiler Lehman graph similarity test on the labeled CFG.
#         For more information on this process, see
#         https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23.
#
#         :param function: the function to hash
#         :return: a deterministic hash of the function
#         """
#         return np.concatenate(
#             np.pad(res, (0, size - len(res)), "constant", constant_values=0) for res, size in
#             ((f.extract(function), f.size) for f in FEATURES)
#         )
#
#     def hash_binary(bv: BinaryView,)
