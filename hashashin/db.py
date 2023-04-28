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
import git
import re
import subprocess
import shutil
import os

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from hashashin.classes import BinarySigModel
from hashashin.classes import BinarySignature
from hashashin.classes import FunctionFeatModel
from hashashin.classes import FunctionFeatures
from hashashin.classes import ORM_BASE
from hashashin.utils import build_net_snmp_from_tag
from hashashin.utils import get_binaries
import logging

logger = logging.getLogger(__name__)
SIG_DB_PATH = Path(__file__).parent / "hashashin.db"
NET_SNMP_BINARY_CACHE = Path(__file__).parent / "binary_data/net-snmp-binaries"


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

    def fast_match(self, signature: BinarySignature, threshold: float = 0.5) -> List[BinarySignature]:
        """Return all matching signatures with a similarity above the threshold."""
        raise NotImplementedError

    def get(self, path: Path):
        raise NotImplementedError

    def getAll(self):
        raise NotImplementedError
    
    def get_snmp_signatures(self, version: Optional[str]):
        raise NotImplementedError

    def get_id_from_path(self, path: Path):
        raise NotImplementedError
    
    @staticmethod
    def get_repo_names() -> List[str]:
        return [subclass.name for subclass in BinarySignatureRepository.__subclasses__()]
    
    def from_name(self, name: str):
        for subclass in self.__class__.__subclasses__():
            if subclass.__name__ == name:
                return subclass()
        raise ValueError(f"Unknown repository type {name}")

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

    def get_feature_matrix(self, binary: Optional[str]) -> Tuple[np.array, List[FunctionFeatModel]]:
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

    def get_feature_matrix(self, binary_path: Optional[Path]) -> Tuple[np.array, List[FunctionFeatModel]]:
        with self.session() as session:
            features = session.query(FunctionFeatModel)
            if binary_path is not None:
                bin_id = session.query(BinarySigModel).filter_by(path=str(binary_path)).first()
                if bin_id is None:
                    if session.query(BinarySigModel).filter_by(path=str('hashashin' / binary_path)).first() is not None:
                        bin_id = session.query(BinarySigModel).filter_by(path=str('hashashin' / binary_path)).first()
                    else:
                        raise ValueError(f"Binary {binary_path} not found in database")
                features = features.filter_by(bin_id=bin_id.id)
            features = features.all()
            return np.stack([FunctionFeatures.fromModel(x).asArray() for x in features]), features


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

    def fast_match(self, signature: BinarySignature, threshold: float = 0.5) -> List[BinarySignature]:
        with self.session() as session:
            if threshold == 0:
                return (
                    session.query(BinarySigModel)
                    .filter(BinarySigModel.sig == signature.signature)
                    .all()
                )
            signatures = session.query(BinarySigModel).all()
            signatures = list(filter(lambda s: (s ^ signature).count(b'\x00') > threshold * len(signature.signature), signatures))
            sorted_signatures = sorted(
                signatures,
                key=lambda sig: BinarySignature.fromModel(sig) // signature,
                reverse=True,
            )
            return sorted_signatures

    def match_signature(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        logger.warning("This is a very slow operation. (O(n))")
        with self.session() as session:
            if threshold == 0:
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
            return [BinarySignature.fromModel(x) for x in session.query(BinarySigModel).all()]
    
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
    
    def get_snmp_signatures(self, version: Optional[str] = None) -> List[BinarySignature]:
        return self.binary_repo.get_snmp_signatures(version)

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


def cache_snmp_binaries(binary_paths: list, version: str) -> Generator[Path, None, None]:
    """Save compiled net-snmp binaries to disk"""
    for path in binary_paths:
        yield _cache_snmp_binary(path, version)


def get_latest_hashed_snmp_tag(tags: list) -> Optional[git.Tag]:
    for tag in reversed(tags):
        dirpath = NET_SNMP_BINARY_CACHE / f"net-snmp-{tag.name}"
        if dirpath.is_dir() and len(os.listdir(dirpath)) > 0:
            return tag
    return None


def compute_net_snmp_db(output_dir: Union[str, Path] = Path(__file__).parent / "binary_data/net-snmp", skip_failed_builds: bool = True):
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
    hashApp = ApplicationFactory.getHasher(HashashinApplicationContext.from_args(["--save-to-db"]))
    successes = []
    for tag in tags:
        # check if tag has already been built and cached
        if (NET_SNMP_BINARY_CACHE / f"net-snmp-{tag.name}").is_dir() and len(
            cached_binaries := list(NET_SNMP_BINARY_CACHE.glob(f"net-snmp-{tag.name}/*"))
        ) > 0:
            logger.info(f"Found cached binaries for {tag.name}")
        else:
            # skip if tag is not cached but a higher version is cached
            if skip_failed_builds and tags.index(tag) <= lbt_index:
                logger.info(f"Skipping {tag.name} because higher version has already been built meaning this tag likely will fail building.")
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
        logger.info(f"Computing signatures for {len(cached_binaries)} binaries in {tag.name}")
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
        format='%(asctime)s,%(msecs)03d %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level='INFO',
    )
    success = compute_net_snmp_db(output_dir / "net-snmp")
    breakpoint()


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
