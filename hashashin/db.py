from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Collection
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from hashashin.classes import BinarySigModel
from hashashin.classes import BinarySignature
from hashashin.classes import FunctionFeatModel
from hashashin.classes import FunctionFeatures
from hashashin.classes import ORM_BASE
from hashashin.utils import logger

logger = logger.getChild(Path(__file__).name)
SIG_DB_PATH = Path(__file__).parent / "hashashin.db"


class RepositoryType(Enum):
    NONE = 0
    SQLALCHEMY = 1


@dataclass
class RepositoryConfig(ABC):
    db_type: tuple[RepositoryType, RepositoryType] = (
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
    ) -> list[BinarySignature]:
        """Return all matching signatures with a similarity above the threshold."""
        raise NotImplementedError

    def fast_match(self, signature: BinarySignature, threshold: float = 0.5) -> list[BinarySignature]:
        """Return all matching signatures with a similarity above the threshold."""
        raise NotImplementedError

    def get(self, path: Path):
        raise NotImplementedError

    def getAll(self):
        raise NotImplementedError

    def get_id_from_path(self, path: Path):
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

    def get_feature_matrix(self, binary: Optional[str]) -> tuple[np.array, list[FunctionFeatModel]]:
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

    def get_feature_matrix(self, binary_path: Optional[Path]) -> tuple[np.array, list[FunctionFeatModel]]:
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

    def fast_match(self, signature: BinarySignature, threshold: float = 0.5) -> list[BinarySignature]:
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
    ) -> list[BinarySignature]:
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
            binhash = BinarySignature.hash_file(path)
            cached = session.query(BinarySigModel).filter_by(hash=binhash)
            if cached.count() > 1:
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

    def getAll(self) -> list[BinarySignature]:
        with self.session() as session:
            return [BinarySignature.fromModel(x) for x in session.query(BinarySigModel).all()]

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
    ) -> list[BinarySignature]:
        return self.binary_repo.match_signature(signature, threshold)

    def __len__(self) -> int:
        return len(self.binary_repo)


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
