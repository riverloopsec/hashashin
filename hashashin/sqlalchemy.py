from hashashin.classes import BinarySignature
from hashashin.classes import FunctionFeatures
from hashashin.classes import FunctionFeatModel
from hashashin.classes import BinarySigModel
from hashashin.classes import ORM_BASE
from hashashin.db import AbstractHashRepository
from hashashin.utils import str2path
from hashashin.utils import get_binaries
from hashashin.db import RepositoryType
from hashashin.classes import BinjaFeatureExtractor

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Collection
from typing import Iterable
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


SIG_DB_PATH = Path(__file__).parent / "hashashin.db"
NET_SNMP_BINARY_CACHE = Path(__file__).parent / "binary_data/net-snmp-binaries"


class BinarySignatureRepository:
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

    def drop(self, option: Optional[Union[str, Path]] = None):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class FunctionFeatureRepository:
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

    def match(
        self, fn_feats: FunctionFeatures, topn: int = 10
    ) -> List[FunctionFeatModel]:
        raise NotImplementedError


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

    def get_bin_count(self, bin_id: int) -> int:
        with self.session() as session:
            return session.query(FunctionFeatModel).filter_by(bin_id=bin_id).count()

    def __len__(self) -> int:
        with self.session() as session:
            return session.query(FunctionFeatModel).count()

    def drop(self):
        ORM_BASE.metadata.drop_all(self.engine)

    def match(
        self, fn_feats: FunctionFeatures, topn: int = 10
    ) -> List[FunctionFeatModel]:
        with self.session() as session:
            features = session.query(FunctionFeatModel).all()
            logger.info(f"Matching {len(features)} features against {fn_feats}")
            features = [
                (x, x ^ fn_feats)
                for x in tqdm(features, desc=f"Computing similarity scores...")
            ]
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
        raise NotImplementedError
        # with self.session() as session:
        #     if threshold == 0:
        #         return (
        #             session.query(BinarySigModel)
        #             .filter(BinarySigModel.sig == signature.signature)
        #             .all()
        #         )
        #     signatures = session.query(BinarySigModel).all()
        #     signatures = list(
        #         filter(
        #             lambda s: (s ^ signature).count(b"\x00")
        #             > threshold * len(signature.signature),
        #             signatures,
        #         )
        #     )
        #     sorted_signatures = sorted(
        #         signatures,
        #         key=lambda sig: BinarySignature.fromModel(sig) // signature,
        #         reverse=True,
        #     )
        #     return sorted_signatures

    def match_signature(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        raise NotImplementedError
        # if threshold == 0:
        #     raise ValueError("Threshold must be > 0")
        # logger.warning("This is a very slow operation. (O(n))")
        # with self.session() as session:
        #     if threshold == 1:
        #         return (
        #             session.query(BinarySigModel)
        #             .filter(BinarySigModel.sig == signature.zlib_signature)
        #             .all()
        #         )
        #     signatures = session.query(BinarySigModel).all()
        #     for sig in signatures:
        #         try:
        #             BinarySignature.fromModel(sig).signature
        #         except Exception as e:
        #             breakpoint()
        #             pass
        #     sorted_signatures = sorted(
        #         signatures,
        #         key=lambda sig: BinarySignature.fromModel(sig) // signature,
        #         reverse=True,
        #     )
        #     return [
        #         sig
        #         for sig in sorted_signatures
        #         if BinarySignature.fromModel(sig) // signature > threshold
        #     ]

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

    def summary(self, path_filter: str = ""):
        """Get summary of database.
        Return the number of binaries and functions in the database."""
        with self.session() as session:
            return (
                bins := session.query(BinarySigModel).filter(
                    BinarySigModel.filename.like(f"%{path_filter}%")
                )
            ).count(), session.query(FunctionFeatModel).filter(
                FunctionFeatModel.bin_id.in_({x.id for x in bins})
            ).count()

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


class SQLAlchemyHashRepository(AbstractHashRepository):
    def __init__(
        self,
    ):
        self.binary_repo: BinarySignatureRepository = (
            SQLAlchemyBinarySignatureRepository()
        )
        self.function_repo: FunctionFeatureRepository = (
            SQLAlchemyFunctionFeatureRepository()
        )

    def insert(self, signature: BinarySignature):
        binary = self.binary_repo.store_signature(signature)
        for feat in signature.functionFeatureList:
            feat.binary_id = binary.id
            self.function_repo.store_feature(feat)

    def get(self, path: Path) -> Optional[BinarySignature]:
        return self.binary_repo.get(path)

    def getAll(self):
        return self.binary_repo.getAll()

    def match(
        self, signature: BinarySignature, threshold: float = 0.5
    ) -> List[BinarySignature]:
        return self.binary_repo.match_signature(signature, threshold)

    def drop(self, option: Optional[Union[str, Path]] = None):
        self.binary_repo.drop(option)

    def summary(self, path_filter: str = ""):
        if path_filter != "":
            raise NotImplementedError
        return self.binary_repo.summary()

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

    # def hashAll(
    #     self, target_path: Union[Path, List[Path]]
    # ) -> Iterable[BinarySignature]:
    #     # TODO: Write this better smh, hardcoding binja is bad :(
    #     if isinstance(target_path, list) and not all(p.exists() for p in target_path):
    #         raise ValueError(
    #             f"List of target binaries contains non-existent paths: {target_path}"
    #         )
    #     if isinstance(target_path, Path) and not target_path.exists():
    #         raise ValueError(f"Target binary does not exist: {target_path}")
    #     extractor = BinjaFeatureExtractor()
    #     targets = get_binaries(target_path, progress=True, recursive=True)
    #     pbar = tqdm(targets)  # type: ignore
    #     for t in pbar:
    #         pbar.set_description(f"Hashing {t}")  # type: ignore
    #         cached = self.get(t)
    #         if cached is not None:
    #             pbar.set_description(f"Retrieved {t} from db")  # type: ignore
    #             logger.debug(f"Binary {t} already hashed, skipping")
    #             yield cached
    #             continue
    #         else:
    #             logger.debug(f"{t} not found in cache")
    #         try:
    #             target_signature = extractor.extract_from_file(
    #                 t,
    #                 progress_kwargs={
    #                     "desc": lambda f: f"Extracting fn {f.name} @ {f.start:#x}",
    #                     "position": 1,
    #                     "leave": False,
    #                 },
    #             )
    #         except BinjaFeatureExtractor.NotABinaryError as e:
    #             logger.warning(f"Skipping {t}: {e}")
    #             continue
    #         self.save(target_signature)
    #         yield target_signature

    def __len__(self) -> int:
        return len(self.binary_repo)
