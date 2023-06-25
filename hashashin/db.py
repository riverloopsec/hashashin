from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple
from typing import Generator
from typing import Iterable
import git
import re
import subprocess
import shutil
import os
from tqdm import tqdm
from multiprocessing import Pool
import pickle

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func

from hashashin.classes import BinarySigModel
from hashashin.classes import BinarySignature
from hashashin.classes import FunctionFeatModel
from hashashin.classes import FunctionFeatures
from hashashin.classes import ORM_BASE
from hashashin.utils import build_net_snmp_from_tag
from hashashin.utils import get_binaries
from hashashin.utils import resolve_relative_path
from hashashin.classes import BinjaFeatureExtractor
from hashashin.metrics import matrix_norms
from hashashin.metrics import stacked_norms
import logging

logger = logging.getLogger(__name__)
SIG_DB_PATH = Path(__file__).parent / "hashashin.db"
NET_SNMP_BINARY_CACHE = Path(__file__).parent / "binary_data/net-snmp-binaries"
BINARY_DATA_SUMMARY_PATH = Path(__file__).parent / "binary_data"
# TODO: Figure out why library only has 576 binaries not 611. Likely due to duplicate binary hashes.
LIBRARY_PATHS = {
    "net-snmp": (Path(__file__).parent / "binary_data/libraries/net-snmp-binaries", 576)
}
LIBRARY_TRIAGE = {
    "net-snmp": lambda filename: "snmp" in str(filename),
}


class RepositoryType(Enum):
    NONE = 0
    SQLALCHEMY = 1


@dataclass
class RepositoryConfig(ABC):
    db_type: Tuple[RepositoryType, RepositoryType] = (
        RepositoryType.NONE,
        RepositoryType.NONE,
    )


class BinarySignatureRepository:
    config: RepositoryConfig
    name: str

    def store_signature(self, binary: BinarySignature):
        raise NotImplementedError

    def store_signatures(self, binaries: List[BinarySignature]):
        for binary in binaries:
            self.store_signature(binary)

    def match_signature(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        """Return all matching signatures with a similarity above the threshold."""
        raise NotImplementedError

    def fast_match(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        """Return all matching signatures with a similarity above the threshold."""
        raise NotImplementedError

    def get(self, path: Path):
        raise NotImplementedError

    def hashAll(self, path: Path):
        raise NotImplementedError

    def getAll(self):
        raise NotImplementedError

    def get_snmp_signatures(self, version: Optional[str]):
        raise NotImplementedError

    def get_id_from_path(self, path: Path):
        raise NotImplementedError

    def get_hashed_binary_classes(self):
        raise NotImplementedError

    def get_binaries_in_path(self, path: Path):
        raise NotImplementedError

    @staticmethod
    def get_repo_names() -> List[str]:
        return [
            subclass.name for subclass in BinarySignatureRepository.__subclasses__()
        ]

    def from_name(self, name: str):
        for subclass in self.__class__.__subclasses__():
            if subclass.__name__ == name:
                return subclass()
        raise ValueError(f"Unknown repository type {name}")
    
    def summary(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class FunctionFeatureRepository:
    config: RepositoryConfig

    def store_feature(self, features: FunctionFeatures):
        raise NotImplementedError

    def store_features(self, features: List[FunctionFeatures]):
        for feat in features:
            self.store_feature(feat)

    def get(self, binary_id: int, function_name: Optional[str] = None):
        raise NotImplementedError

    def get_all(self):
        raise NotImplementedError

    def get_bin_count(self, binary_id: int):
        raise NotImplementedError

    def get_feature_matrix(
        self, binary: Optional[str]
    ) -> Tuple[np.array, List[FunctionFeatModel]]:
        raise NotImplementedError

    def match(self, fn_feats: FunctionFeatures, topn: int = 10) -> List[FunctionFeatModel]:
        raise NotImplementedError


@dataclass
class SQLAlchemyConfig(RepositoryConfig):
    db_type = (RepositoryType.SQLALCHEMY, RepositoryType.SQLALCHEMY)
    db_path: str = f"sqlite:///{SIG_DB_PATH}"

    # def __eq__(self, other) -> bool:
    #     if not isinstance(other, SQLAlchemyConfig):
    #         return False
    #     return self.db_type == other.db_type and self.db_path == other.db_path


class SQLAlchemyFunctionFeatureRepository(FunctionFeatureRepository):
    def __init__(self, db_config: SQLAlchemyConfig = SQLAlchemyConfig()):
        self.config = db_config
        self.engine = create_engine(self.config.db_path)
        ORM_BASE.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def store_feature(self, features: FunctionFeatures) -> FunctionFeatModel:
        func = FunctionFeatModel.fromFunctionFeatures(features)
        with self.session() as session:
            session.add(func)
            session.commit()
            session.refresh(func)
        return func

    def get(self, binary_id: int, function_name: Optional[str] = None):
        with self.session() as session:
            if function_name is None:
                return (
                    session.query(FunctionFeatModel).filter_by(bin_id=binary_id).all()
                )
            else:
                return (
                    session.query(FunctionFeatModel)
                    .filter_by(bin_id=binary_id, name=function_name)
                    .first()
                )

    def get_all(self):
        with self.session() as session:
            return session.query(FunctionFeatModel).all()

    def get_feature_matrix(
        self, binary_path: Optional[Path]
    ) -> Tuple[np.array, List[FunctionFeatModel]]:
        with self.session() as session:
            features = session.query(FunctionFeatModel)
            if binary_path is not None:
                bin_id = (
                    session.query(BinarySigModel)
                    .filter_by(path=str(binary_path))
                    .first()
                )
                if bin_id is None:
                    if (
                        session.query(BinarySigModel)
                        .filter_by(path=str("hashashin" / binary_path))
                        .first()
                        is not None
                    ):
                        bin_id = (
                            session.query(BinarySigModel)
                            .filter_by(path=str("hashashin" / binary_path))
                            .first()
                        )
                    else:
                        raise ValueError(f"Binary {binary_path} not found in database")
                features = features.filter_by(bin_id=bin_id.id)
            features = features.all()
            return (
                np.stack([FunctionFeatures.fromModel(x).asArray() for x in features]),
                features,
            )

    # def match(self, fn: FunctionFeatures, threshold: float = 0.95) -> list[FunctionFeatures]:
    #     with self.session() as session:
    #         features = session.query(FunctionFeatModel).all()
    #         logger.info(f"Matching {len(features)} features against {fn}")
    #         features = [(x, x ^ fn) for x in tqdm(features, desc=f"Computing similarity scores...") if (x ^ fn) > threshold]
    #         logger.info(f"Found {len(features)} features above threshold")
    #         sorted_features = sorted(
    #             features,
    #             key=lambda x: x[1],
    #             reverse=True,
    #         )
    #         sorted_features = [FunctionFeatures.fromModel(x[0]) for x in sorted_features]
    #         return sorted_features

    def get_bin_count(self, bin_id: int) -> int:
        with self.session() as session:
            return session.query(FunctionFeatModel).filter_by(bin_id=bin_id).count()

    def __len__(self) -> int:
        with self.session() as session:
            return session.query(FunctionFeatModel).count()

    def drop(self):
        ORM_BASE.metadata.drop_all(self.engine)

    def match(self, fn_feats: FunctionFeatures, topn: int = 10) -> List[FunctionFeatModel]:
        with self.session() as session:
            features = session.query(FunctionFeatModel).all()
            logger.info(f"Matching {len(features)} features against {fn_feats}")
            features = [(x, x ^ fn_feats) for x in tqdm(features, desc=f"Computing similarity scores...")]
            logger.info(f"Found {len(features)} features above threshold")
            sorted_features = sorted(
                features,
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_features = [x[0] for x in sorted_features[:topn]]
            return sorted_features


class SQLAlchemyBinarySignatureRepository(BinarySignatureRepository):
    name = "sqlalchemy"

    def __init__(self, db_config: SQLAlchemyConfig = SQLAlchemyConfig()):
        self.config = db_config
        self.engine = create_engine(self.config.db_path)
        ORM_BASE.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def store_signature(self, signature: BinarySignature) -> BinarySigModel:
        binary = BinarySigModel.fromBinarySignature(signature)
        # Check if the binary is already in the database
        with self.session() as session:
            dup = session.query(BinarySigModel).filter_by(hash=binary.hash).first()
            if dup:
                logger.warning("Binary already in database")
                if dup != binary:
                    raise ValueError(
                        f"Binary stored in database differs, {dup} != {binary}"
                    )
                return dup
            session.add(binary)
            session.commit()
            session.refresh(binary)
        return binary

    def fast_match(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        with self.session() as session:
            if threshold == 0:
                return (
                    session.query(BinarySigModel)
                    .filter(BinarySigModel.sig == signature.signature)
                    .all()
                )
            signatures = session.query(BinarySigModel).all()
            signatures = list(
                filter(
                    lambda s: (s ^ signature).count(b"\x00")
                    > threshold * len(signature.signature),
                    signatures,
                )
            )
            sorted_signatures = sorted(
                signatures,
                key=lambda sig: BinarySignature.fromModel(sig) // signature,
                reverse=True,
            )
            return sorted_signatures

    def match_signature(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        if threshold == 0:
            raise ValueError("Threshold must be > 0")
        logger.warning("This is a very slow operation. (O(n))")
        with self.session() as session:
            if threshold == 1:
                return (
                    session.query(BinarySigModel)
                    .filter(BinarySigModel.sig == signature.signature)
                    .all()
                )
            signatures = session.query(BinarySigModel).all()
            for sig in signatures:
                try:
                    BinarySignature.fromModel(sig).np_signature
                except Exception as e:
                    breakpoint()
                    pass
            sorted_signatures = sorted(
                signatures,
                key=lambda sig: BinarySignature.fromModel(sig) // signature,
                reverse=True,
            )
            return [
                sig
                for sig in sorted_signatures
                if BinarySignature.fromModel(sig) // signature > threshold
            ]

    def get(self, path: Path) -> Optional[BinarySignature]:
        with self.session() as session:
            logger.debug(f"xxhashing {path}")
            binhash = BinarySignature.hash_file(path)
            logger.debug("Querying database for cached binary")
            cached = session.query(BinarySigModel).filter_by(hash=binhash)
            if cached.count() > 1:
                logger.debug("Found cached binary.")
                if cached.filter_by(path=str(path)).count() == 1:
                    cached = cached.filter_by(path=str(path))
                elif cached.filter_by(path=str(path)).count() > 1:
                    raise ValueError("Duplicate entries in database")
                else:
                    raise ValueError(
                        "Duplicate entries without matching path in database"
                    )
            cached = cached.one() if cached.count() == 1 else None
            try:
                return BinarySignature.fromModel(cached) if cached else None
            except FileNotFoundError:
                logger.warning(
                    f"Cached binary {path} not found, it may have been moved. Updating database"
                )
                cached.path = str(path)
                session.commit()
                return BinarySignature.fromModel(cached)

    def getAll(self) -> List[BinarySignature]:
        with self.session() as session:
            return [
                BinarySignature.fromModel(x)
                for x in session.query(BinarySigModel).all()
            ]

    def get_snmp_signatures(self, version: Optional[str]) -> List[BinarySignature]:
        with self.session() as session:
            if version:
                raise NotImplementedError("SNMP version filtering not implemented")
            # return all signatures with NET_SNMP_BINARY_CACHE.name in path
            return [
                BinarySignature.fromModel(x)
                for x in session.query(BinarySigModel).filter(
                    BinarySigModel.path.like(f"%{NET_SNMP_BINARY_CACHE.name}%")
                )
            ]

    def get_id_from_path(self, path: Path):
        with self.session() as session:
            saved = session.query(BinarySigModel).filter_by(path=str(path)).first()
            if saved:
                return saved.id

    def get_hashed_binary_classes(self):
        # get all paths from database
        with self.session() as session:
            paths = [x.path for x in session.query(BinarySigModel).all()]
        breakpoint()

    def __len__(self) -> int:
        with self.session() as session:
            return session.query(BinarySigModel).count()

    def drop(self, option: Optional[Union[str, Path]] = None):
        if option == "all":
            ORM_BASE.metadata.drop_all(self.engine)
        elif isinstance(option, Path) and option.is_file():
            with self.session() as session:
                cached = session.query(BinarySigModel).filter_by(path=str(option))
                cached.delete()
                session.commit()

    def summary(self):
        """Print summary of database.
        Print the number of binaries and functions in each path relative to binary_data directory"""
        with self.session() as session:
            paths = [
                resolve_relative_path(x.path, BINARY_DATA_SUMMARY_PATH)
                for x in tqdm(
                    session.query(BinarySigModel).all(),
                    desc="Resolving paths",
                )
            ]
            paths = [x.relative_to(BINARY_DATA_SUMMARY_PATH) for x in paths]
            for path in paths:
                print(f"{path}: {paths.count(path)}")
                # get number of functions relative to path
                functions = session.query(FunctionFeatModel).filter(
                    FunctionFeatModel.path.like(f"%{path}%")
                )
                print(f"  Functions: {functions.count()}")

    def get_binaries_in_path(self, path: Path):
        """Get all BinarySignatures with a path that is a subpath of the given path"""
        logger.debug(f"Getting binaries in path {path}")
        with self.session() as session:
            p = path.relative_to(Path(__file__).parent / "binary_data")
            return [
                BinarySignature.fromModel(x)
                for x in tqdm(
                    session.query(BinarySigModel)
                    .filter(BinarySigModel.path.contains(str(p)))
                    .all(),
                    disable=not logger.isEnabledFor(logging.DEBUG),
                )
            ]


class HashRepository:
    @staticmethod
    def _bin_repo_type_to_class(
        repo_type: RepositoryType, config: Optional[RepositoryConfig]
    ) -> BinarySignatureRepository:
        if repo_type == RepositoryType.SQLALCHEMY:
            if config is None:
                return SQLAlchemyBinarySignatureRepository(SQLAlchemyConfig())
            if isinstance(config, SQLAlchemyConfig):
                return SQLAlchemyBinarySignatureRepository(config)
            raise ValueError(f"Invalid config type {type(config)}")
        raise NotImplementedError(f"Repository type {repo_type} not implemented")

    @staticmethod
    def _func_repo_type_to_class(
        repo_type: RepositoryType, config: Optional[RepositoryConfig]
    ) -> FunctionFeatureRepository:
        if repo_type == RepositoryType.SQLALCHEMY:
            if config is None:
                return SQLAlchemyFunctionFeatureRepository(SQLAlchemyConfig())
            if isinstance(config, SQLAlchemyConfig):
                return SQLAlchemyFunctionFeatureRepository(config)
            raise ValueError(f"Invalid config type {type(config)}")
        raise NotImplementedError(f"Repository type {repo_type} not implemented")

    def __init__(
        self,
        repo_type: RepositoryType = RepositoryType.SQLALCHEMY,
        db_config: Optional[RepositoryConfig] = None,
    ):
        self.binary_repo: BinarySignatureRepository = self._bin_repo_type_to_class(
            repo_type, db_config
        )
        self.function_repo: FunctionFeatureRepository = self._func_repo_type_to_class(
            repo_type, db_config
        )
        if self.binary_repo.config != self.function_repo.config:
            raise ValueError("Binary and Function repository must have the same config")
        self.db_config = self.binary_repo.config

    def save(self, signatures: Union[Collection[BinarySignature], BinarySignature]):
        if isinstance(signatures, BinarySignature):
            signatures = [signatures]
        for sig in signatures:
            if len(sig.functionFeatureList) == 0:
                raise ValueError("Signature has no function features")
            binary = self.binary_repo.store_signature(sig)
            for feat in sig.functionFeatureList:
                feat.binary_id = binary.id
                self.function_repo.store_feature(feat)

    def get(self, path: Path) -> Optional[BinarySignature]:
        return self.binary_repo.get(path)

    def match(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        return self.binary_repo.match_signature(signature, threshold)

    def get_snmp_signatures(
        self, version: Optional[str] = None
    ) -> List[BinarySignature]:
        return self.binary_repo.get_snmp_signatures(version)

    @staticmethod
    def _process_file(t):
        extractor = BinjaFeatureExtractor()
        try:
            target_signature = extractor.extract_from_file(t)
        except BinjaFeatureExtractor.NotABinaryError as e:
            logger.warning(f"Skipping {t}: {e}")
            return None
        breakpoint()
        return target_signature

    def hashAll(
        self, target_path: Union[Path, List[Path]]
    ) -> Iterable[BinarySignature]:
        # TODO: Write this better smh, hardcoding binja is bad :(
        if isinstance(target_path, list) and not all(p.exists() for p in target_path):
            raise ValueError(
                f"List of target binaries contains non-existent paths: {target_path}"
            )
        if isinstance(target_path, Path) and not target_path.exists():
            raise ValueError(f"Target binary does not exist: {target_path}")
        extractor = BinjaFeatureExtractor()
        targets = get_binaries(target_path, progress=True, recursive=True)
        pbar = tqdm(targets)  # type: ignore
        for t in pbar:
            pbar.set_description(f"Hashing {t}")  # type: ignore
            cached = self.get(t)
            if cached is not None:
                pbar.set_description(f"Retrieved {t} from db")  # type: ignore
                logger.debug(f"Binary {t} already hashed, skipping")
                yield cached
                continue
            else:
                logger.debug(f"{t} not found in cache")
            try:
                target_signature = extractor.extract_from_file(
                    t,
                    progress_kwargs={
                        "desc": lambda f: f"Extracting fn {f.name} @ {f.start:#x}",
                        "position": 1,
                        "leave": False,
                    },
                )
            except BinjaFeatureExtractor.NotABinaryError as e:
                logger.warning(f"Skipping {t}: {e}")
                continue
            self.save(target_signature)
            yield target_signature

    def __len__(self) -> int:
        return len(self.binary_repo)


def _cache_snmp_binary(path: Path, version: str) -> Path:
    """Save compiled net-snmp binary to disk"""
    if not NET_SNMP_BINARY_CACHE.is_dir():
        NET_SNMP_BINARY_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = NET_SNMP_BINARY_CACHE / f"net-snmp-{version}"
    cache_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, cache_path / path.name)
    return cache_path / path.name


def cache_snmp_binaries(
    binary_paths: list, version: str
) -> Generator[Path, None, None]:
    """Save compiled net-snmp binaries to disk"""
    for path in binary_paths:
        yield _cache_snmp_binary(path, version)


def get_latest_hashed_snmp_tag(tags: list) -> Optional[git.Tag]:
    for tag in reversed(tags):
        dirpath = NET_SNMP_BINARY_CACHE / f"net-snmp-{tag.name}"
        if dirpath.is_dir() and len(os.listdir(dirpath)) > 0:
            return tag
    return None


def compute_net_snmp_db(
    output_dir: Union[str, Path] = Path(__file__).parent / "binary_data/net-snmp",
    skip_failed_builds: bool = True,
):
    """Download the net-snmp database from GitHub and add signatures to db"""
    # TODO: get rid of lazy imports
    from hashashin.main import ApplicationFactory, HashashinApplicationContext

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_url = "git@github.com:net-snmp/net-snmp.git"
    # clone repo if not already present
    if not (output_dir / ".git").is_dir():
        logger.info(f"Cloning {repo_url} to {output_dir}")
        repo = git.Repo.clone_from(repo_url, output_dir)
    else:
        logger.info(f"Found existing repo in {output_dir}")
        repo = git.Repo(output_dir)
    # match all tags above v5.0
    tags = [t for t in repo.tags if re.match(r"^v[5-9]\.[0-9](?:\.[0-9]+)?$", t.name)]
    last_built_tag = get_latest_hashed_snmp_tag(tags)
    lbt_index = tags.index(last_built_tag) if last_built_tag is not None else -1
    hashApp = ApplicationFactory.getHasher(
        HashashinApplicationContext.from_args(["--save-to-db"])
    )
    successes = []
    for tag in tags:
        # check if tag has already been built and cached
        if (NET_SNMP_BINARY_CACHE / f"net-snmp-{tag.name}").is_dir() and len(
            cached_binaries := list(
                NET_SNMP_BINARY_CACHE.glob(f"net-snmp-{tag.name}/*")
            )
        ) > 0:
            logger.info(f"Found cached binaries for {tag.name}")
        else:
            # skip if tag is not cached but a higher version is cached
            if skip_failed_builds and tags.index(tag) <= lbt_index:
                logger.info(
                    f"Skipping {tag.name} because higher version has already been built meaning this tag likely will fail building."
                )
                continue
            try:
                build_net_snmp_from_tag(repo, tag, output_dir)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error building net-snmp {tag.name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error collecting net-snmp {tag.name}: {e}")
                breakpoint()
            binaries = get_binaries(output_dir / "apps")
            if len(binaries) == 0:
                logger.warning(f"No binaries found after building {tag.name}")
                continue
            cached_binaries = list(cache_snmp_binaries(binaries, tag.name))
        logger.info(
            f"Computing signatures for {len(cached_binaries)} binaries in {tag.name}"
        )
        signatures = list()
        for bin in cached_binaries:
            sig = hashApp.target(bin)
            if len(sig) != 1:
                logger.warning(f"Found {len(sig)} signatures for {bin.name}")
                breakpoint()
            signatures.append(sig[0])
        successes.append(tag.name)
    failures = [t.name for t in tags if t.name not in successes]
    logger.info(f"Successfully built {len(successes)} net-snmp versions: {successes}")
    logger.info(f"Failed to build {len(failures)} net-snmp versions: {failures}")
    return successes


def populate_db(output_dir: Union[str, Path] = Path(__file__).parent / "binary_data/"):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level="INFO",
    )
    success = compute_net_snmp_db(output_dir / "net-snmp")
    breakpoint()


def _load_library_matrix(
        library: str,
        generate: bool = False,
):
    # TODO: ok I know this is bad but I need a speedup fast so I'm pickling the BinarySignature triage
    pickle_path = SIG_DB_PATH.parent / f"{library}_triage.pickle"
    lib_bins = None
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            library_matrix = pickle.load(f)
        logger.info(f"Loaded {library} signature matrix from {pickle_path}")
    else:
        # Validate library binaries exist
        library_path, library_count = LIBRARY_PATHS[library]
        db = HashRepository()
        # TODO: This operation is very slow. Add db indexing to speed up.
        lib_bins = db.binary_repo.get_binaries_in_path(library_path)
        if len(lib_bins) != library_count:
            breakpoint()
            if generate:
                logger.info(f"Generating signatures for {library} binaries")
                lib_bins = list(db.hashAll(library_path))
            else:
                raise ValueError(
                    f"Library binaries not found in {library_path}. "
                    f"Pass --generate to generate signatures."
                )
        logger.info(f"Computing {library} signature matrix for caching.")
        library_matrix = np.stack(
            [
                bs.np_signature
                for bs in tqdm(lib_bins, disable=not logger.isEnabledFor(logging.DEBUG))
            ]
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(library_matrix, f)
    return library_matrix, lib_bins


def _load_library_np_signatures(
        library: str,
        lib_bins
):
    # TODO: get rid of pickle caching and make db operations faster
    pickle_path = SIG_DB_PATH.parent / f"{library}_signatures.pickle"
    np_signatures = None
    if pickle_path.exists():
        with open(pickle_path, "rb") as f:
            np_signatures, lib_bins_paths = pickle.load(f)
        logger.info(f"Loaded {library} signatures from {pickle_path}")
    else:
        if lib_bins is None:
            logger.info(f"Loading {library} binaries from database")
            library_path, library_count = LIBRARY_PATHS[library]
            db = HashRepository()
            lib_bins = db.binary_repo.get_binaries_in_path(library_path)
            # TODO: test assumption that binaries are pre-computed if pickle exists
        lib_bins_paths = [b.path for b in lib_bins]
        logger.info("Gathering function features from library binaries...")
        np_signatures = [b.function_matrix for b in tqdm(lib_bins, disable=not logger.isEnabledFor(logging.DEBUG))]
        with open(pickle_path, "wb") as f:
            pickle.dump((np_signatures, lib_bins_paths), f)
            logger.info(f"Saved {library} signatures to {pickle_path}")
    return np_signatures, lib_bins_paths


def get_closest_library_versions(
    library: str,
    binary_path: List[Union[str, Path]],
    generate: bool = False,
    triage_threshold: float = 0.1,
    match_threshold: float = 0.3,
    remove_nones: bool = False,
) -> Tuple[List[Optional[str]], List[Path]]:
    """Compute the closest library version for a given binary.
    Args:
        library (str): Library to match against
        binary_path (List[Union[str, Path]]): Path to binaries to match
        generate (bool): Generate library signatures if not present.
            If True and the library binaries are not found in the path
            listed in LIBRARY_PATHS, an error will be thrown.
        triage_threshold (float): Threshold for triage. If the closest
            library version is above this threshold, None will be returned.
        match_threshold (float): Threshold for matching. If the closest
            library version is above this threshold, None will be returned.
        remove_nones (bool): If True, remove None values from the return

    Returns:
        tuple: Tuple of (closest_library_version or None, filename)
    """
    # TODO: Add back bin_path type hint, mypy is being terrifically annoying
    if library not in LIBRARY_PATHS:
        raise ValueError(f"Invalid library {library} not in {LIBRARY_PATHS.keys()}")
    if not isinstance(binary_path, list):
        if isinstance(binary_path, str):
            binary_paths = [Path(binary_path)]
        elif isinstance(binary_path, Path):
            binary_paths = [binary_path]
        else:
            raise ValueError(f"Invalid binary_path {binary_path}")
    binary_paths = [Path(p) if isinstance(p, str) else p for p in binary_path]
    if not all(isinstance(p, Path) for p in binary_paths):
        raise ValueError(f"Invalid binary_paths {binary_paths}")

    # resolve directories and validate files
    filelist = list()
    for potential_dir in binary_paths:
        if not potential_dir.exists():
            raise ValueError(f"Invalid binary path {potential_dir} does not exist")
        if potential_dir.is_file():
            filelist.append(potential_dir)
        elif potential_dir.is_dir():
            filelist.extend(get_binaries(potential_dir))
        else:
            raise ValueError(f"Invalid binary path {potential_dir}")

    # Triage by name
    filename_triage = [f for f in filelist if LIBRARY_TRIAGE[library](f)]
    logger.info(f"Found {len(filename_triage)} files matching {library} filename")
    if len(filename_triage) == 0:
        return [None] * len(filelist), filelist if not remove_nones else []

    # Triage by signature
    library_matrix, lib_bins = _load_library_matrix(library, generate)
    extractor = BinjaFeatureExtractor()
    signature_triage = list()
    for bin_path in filename_triage:
        bs = BinarySignature.fromFile(bin_path, extractor)

        closest_sig = max(
            library_matrix, key=lambda x: bs.minhash_similarity(bs.np_signature, x)
        )
        closest_value = bs.minhash_similarity(bs.np_signature, closest_sig)
        if closest_value < triage_threshold:
            logger.info(
                f"Closest {library} signature to {bin_path} is {closest_value} below threshold {triage_threshold}. Skipping."
            )
            continue
        logger.info(
            f"Closest {library} signature to {bin_path} is {closest_value}. Continuing to robust matching."
        )
        signature_triage.append(bs)

    if len(signature_triage) == 0:
        logger.info(f"No binaries passed triage for {library}.")
        return [None] * len(filelist), filelist if not remove_nones else []

    # Triage by FunctionFeatures robust matching
    np_signatures, lib_bins_paths = _load_library_np_signatures(library, lib_bins)
    feature_triage: List[Tuple[str, Path]] = list()
    for sig in tqdm(signature_triage, disable=not logger.isEnabledFor(logging.INFO)):
        logger.debug(f"Computing norms for {sig.path}...")
        norms = stacked_norms(np_signatures, sig.function_matrix)
        if max(norms) < match_threshold:
            logger.info(
                f"Closest {library} function features is {max(norms)} below threshold {match_threshold}"
            )
            continue
        version = str(lib_bins_paths[np.argmax(norms)].parent).split("-v")[1]
        logger.info(
            f"Closest {library} function features is {max(norms)} to v{version}."
        )
        feature_triage.append(
            (version, sig.path)
        )
    if len(feature_triage) == 0:
        logger.info(f"No binaries passed robust matching for {library}.")
        return [None] * len(filelist), filelist if not remove_nones else []

    # Return closest library version
    if remove_nones:
        return [v for v, _ in feature_triage], [p for _, p in feature_triage]

    # Add back nones to match up with filelist
    versions, paths = [None] * len(filelist), [None] * len(filelist)
    for version, path in feature_triage:
        idx: int = int(filelist.index(path))
        versions[idx] = version  # type: ignore
        paths[idx] = path  # type: ignore
    return versions, paths  # type: ignore


def get_closest_library_version(
    library: str,
    binary_path: Union[Path, str],
    generate: bool = False,
    triage_threshold: float = 0.1,
    match_threshold: float = 0.3,
) -> Optional[str]:
    """Compute the closest library version for a given binary.
    Args:
        library (str): Library to match against
        binary_path (Union[Path, str]): Path to binary to match
        generate (bool): Generate library signatures if not present.
            If True and the library binaries are not found in the path
            listed in LIBRARY_PATHS, an error will be thrown.
        triage_threshold (float): Threshold for triage. If the closest
            library version is above this threshold, None will be returned.
        match_threshold (float): Threshold for matching. If the closest
            library version is above this threshold, None will be returned.

    Returns:
        str: Closest library version, or None if no version is close enough.
    """
    binary_path = Path(binary_path) if isinstance(binary_path, str) else binary_path
    if not binary_path.is_file():
        raise ValueError(f"binary path {binary_path} is not a file")
    ret = get_closest_library_versions(
        library,
        [binary_path],
        generate=generate,
        triage_threshold=triage_threshold,
        match_threshold=match_threshold,
        remove_nones=False,
    )
    if len(ret) == 0:
        return None
    return ret[0][0]


def get_closest_library_version_cli() -> Union[Tuple[List[Optional[str]], List[Path]],Optional[str]]:
    """CLI entrypoint for get_closest_library_version"""
    import argparse

    parser = argparse.ArgumentParser(description="Match a binary against a library")
    parser.add_argument(
        "library",
        type=str,
        choices=LIBRARY_PATHS.keys(),
        default="net-snmp",
        help="Library to match against",
    )
    parser.add_argument("bin_path", type=str, nargs="+", help="Path(s) to binary")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate library signatures if not present",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--threshold", type=float, default=0.1, help="Triage threshold")
    parser.add_argument("--match-threshold", type=float, default=0.3, help="Match threshold")
    args = parser.parse_args()
    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    versions, paths = get_closest_library_versions(
        args.library, args.bin_path, args.generate, args.threshold, args.match_threshold
    )
    if all(v is None for v in versions):
        logger.info("No matches found.")
    else:
        for v, p in zip(versions, paths):
            if v is not None:
                logger.info(f"{p}: {v}")
    if len(paths) == 1:
        return versions[0]
    return versions, paths


if __name__ == "__main__":
    # testing
    from hashashin.metrics import (
        compute_metrics,
        compute_matrices,
        hash_paths,
    )

    from hashashin.main import ApplicationFactory, HashashinApplicationContext
    from hashashin.classes import BinjaFeatureExtractor

    app_context = HashashinApplicationContext(
        extractor=BinjaFeatureExtractor(),
        hash_repo=HashRepository(),
        target_path=None,
        save_to_db=True,
    )
    hashApp = ApplicationFactory.getHasher(app_context)

    signatures = hash_paths("openssl", hashApp, paths="*[0-9][.][0-9]*")

    minhash_similarities, jaccard_similarities, binaries = compute_matrices(signatures)
    minhash_metrics = compute_metrics(minhash_similarities)
    print(
        f"Minhash precision: {minhash_metrics[0]}, recall: {minhash_metrics[1]}, f1: {minhash_metrics[2]}"
    )
    jaccard_metrics = compute_metrics(jaccard_similarities)
    print(
        f"Jaccard precision: {jaccard_metrics[0]}, recall: {jaccard_metrics[1]}, f1: {jaccard_metrics[2]}"
    )
