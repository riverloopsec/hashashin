import csv
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union

import xxhash
from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.orm.relationships import RelationshipProperty

from hashashin.classes import BinarySignature, FunctionFeatures

logger = logging.getLogger(os.path.basename(__name__))
ORM_BASE: Any = declarative_base()
SIG_DB_PATH = Path(__file__).parent / "hashashin.db"


class BinarySigModel(ORM_BASE):
    __tablename__ = "binaries"
    id = Column(Integer, primary_key=True)
    hash = Column(LargeBinary, unique=True, index=True)
    path = Column(String)
    sig = Column(LargeBinary, nullable=True)
    functions: RelationshipProperty = relationship("FunctionFeatModel")
    extraction_engine = Column(String)

    @classmethod
    def fromBinarySignature(cls, sig: BinarySignature) -> "BinarySigModel":
        with open(sig.path, "rb") as f:
            return cls(
                hash=xxhash.xxh64(f.read()).digest(),
                path=sig.path.name,
                sig=sig.signature,
                extraction_engine=str(sig.extraction_engine),
            )

    def __eq__(self, other: "BinarySigModel") -> bool:
        return (
            self.hash == other.hash
            and self.path == other.path
            and self.sig == other.sig
            and self.extraction_engine == other.extraction_engine
        )


class FunctionFeatModel(ORM_BASE):
    __tablename__ = "functions"
    id = Column(Integer, primary_key=True)
    bin_id = Column(Integer, ForeignKey("binaries.id"))
    binary: RelationshipProperty = relationship(
        "BinarySigModel", back_populates="functions"
    )
    name = Column(String)  # function name & address using func2str
    sig = Column(LargeBinary)  # zlib compression has variable size
    extraction_engine = Column(String)

    @classmethod
    def fromFunctionFeatures(cls, features: FunctionFeatures) -> "FunctionFeatModel":
        if features.binary_id is None:
            # TODO: query the database for the binary_id
            raise ValueError("binary_id must be set")
        return cls(
            bin_id=features.binary_id,
            name=features.function.name,
            sig=features.signature,
            extraction_engine=str(features.extraction_engine),
        )


class FunctionFeatureRepository:
    def store_feature(self, features: FunctionFeatures):
        raise NotImplementedError

    def store_features(self, features: List[FunctionFeatures]):
        for feat in features:
            self.store_feature(feat)


@dataclass
class SQLAlchemyConfig:
    db_path: str = f"sqlite:///{SIG_DB_PATH}"


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

    def __len__(self) -> int:
        with self.session() as session:
            return session.query(FunctionFeatModel).count()

    def drop(self):
        ORM_BASE.metadata.drop_all(self.engine)


class BinarySignatureRepository:
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
            if session.query(BinarySigModel).filter_by(hash=binary.hash).first():
                logger.debug("Binary already in database")
                return
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
            sorted_signatures = sorted(
                signatures,
                key=lambda sig: sig.sig.similarity(signature.signature),
                reverse=True,
            )
            return [
                sig
                for sig in sorted_signatures
                if sig.sig.similarity(signature.signature) > threshold
            ]

    def __len__(self) -> int:
        with self.session() as session:
            return session.query(BinarySigModel).count()

    def drop(self):
        ORM_BASE.metadata.drop_all(self.engine)
