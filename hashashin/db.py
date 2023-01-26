import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union
from typing import Any, List
from enum import Enum
from dataclasses import dataclass
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, relationship
from sqlalchemy.orm.relationships import RelationshipProperty
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, ForeignKey, LargeBinary
from hashashin.classes import BinarySignature, FunctionFeatures

import xxhash

logger = logging.getLogger(os.path.basename(__name__))
ORM_BASE: Any = declarative_base()
SIG_DB_PATH = Path(__file__).parent / "sig_db.sqlite3"


class BinarySigModel(ORM_BASE):
    __tablename__ = "binaries"
    id = Column(Integer, primary_key=True)
    hash = Column(Integer, unique=True, index=True)  # xxhash
    path = Column(String)
    sig = Column(LargeBinary, nullable=True)
    functions: RelationshipProperty = relationship("FunctionFeatModel")
    extraction_engine = Column(String)

    @classmethod
    def fromBinarySignature(cls, sig: BinarySignature) -> "BinarySigModel":
        with open(sig.path, "rb") as f:
            return cls(
                hash=xxhash.xxh64(f.read()).intdigest(),
                path=sig.path.name,
                sig=sig.signature,
                extraction_engine=sig.extraction_engine,
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
        logger.warning("Table entry may not have bin_id fk set correctly.")
        return cls(
            name=features.function.name,
            sig=features.signature,
            extraction_engine=features.extraction_engine,
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

    def store_feature(self, features: FunctionFeatures):
        func = FunctionFeatModel.fromFunctionFeatures(features)
        with self.session() as session:
            session.add(func)
            session.commit()


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

    def store_signature(self, signature: BinarySignature):
        binary = BinarySigModel.fromBinarySignature(signature)
        with self.session() as session:
            session.add(binary)
            session.commit()

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
