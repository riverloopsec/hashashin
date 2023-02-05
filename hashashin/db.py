import csv
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union, Collection
from abc import ABC

import xxhash
from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String, create_engine

from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.orm.relationships import RelationshipProperty

from hashashin.classes import (
    BinarySigModel,
    FunctionFeatModel,
    BinarySignature,
    FunctionFeatures,
    ORM_BASE,
)

logger = logging.getLogger(os.path.basename(__name__))
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

    def get(self, path: Path):
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
            with open(path, "rb") as f:
                binhash = xxhash.xxh64(f.read()).digest()
            cached = session.query(BinarySigModel).filter_by(hash=binhash).first()
            return BinarySignature.fromModel(cached) if cached else None

    def __len__(self) -> int:
        with self.session() as session:
            return session.query(BinarySigModel).count()

    def drop(self):
        ORM_BASE.metadata.drop_all(self.engine)


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
