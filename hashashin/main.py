import argparse
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator
from typing import Iterable
from typing import Optional
from typing import Union
from typing import Tuple, List

import numpy as np
from tqdm import tqdm
import pickle

from hashashin.classes import BinarySignature
from hashashin.classes import BinjaFeatureExtractor
from hashashin.classes import FeatureExtractor
from hashashin.classes import FunctionFeatures
from hashashin.db import HashRepository
from hashashin.db import BinarySignatureRepository
from hashashin.db import SQLAlchemyBinarySignatureRepository
from hashashin.db import SQLAlchemyFunctionFeatureRepository
from hashashin.metrics import stacked_norms
from hashashin.utils import get_binaries
from hashashin.utils import list_rindex
from hashashin.utils import get_parser
import logging

from hashashin.db import NET_SNMP_BINARY_CACHE

logger = logging.getLogger(__name__)


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

        if not (
            # db operations
            (args.status or args.drop or args.summary)
            # app operations
            or (args.task and args.target)
            # demo operations
            or (args.fast_match or args.robust_match)
        ):
            parser.error("Invalid arguments. See --help for usage.")

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=level,
    )

    if args.status:
        print("Database status:")
        print(f"\t{len(SQLAlchemyBinarySignatureRepository())} binaries")
        print(f"\t{len(SQLAlchemyFunctionFeatureRepository())} functions")
        return

    if args.summary:
        print("Database summary:")
        print(f"\t{SQLAlchemyBinarySignatureRepository().summary()}")
        return

    if args.drop is not None:
        # Confirm drop
        if not input(f"Confirm drop {args.drop}? [y/N] ").lower().startswith("y"):
            print("Aborting")
            return
        print("Dropping database tables...")
        SQLAlchemyBinarySignatureRepository().drop(args.drop)
        print("Done!")
        args.drop, args.status = False, True
        main(args)
        return

    if args.fast_match:
        for target in args.fast_match:
            target = Path(__file__).parent / target
            if not target.is_file() or len(get_binaries(target, silent=True)) == 0:
                logger.debug(f"{target} is not a valid binary")
                continue
            hash_repo = HashRepository()
            target = hash_repo.get(target)
            if target is None:
                logger.error(f"Failed to hash {target}")
                continue
            logger.info(
                f"Fast matching target {target.path.absolute().relative_to(Path(__file__).parent / 'binary_data')}"
            )
            matches = hash_repo.binary_repo.fast_match(target, args.threshold)
            matches = list(filter(lambda x: x.path != str(target.path), matches))
            _log = f"Found {len(matches)} matches above {args.threshold} similarity"
            if len(matches) == 0:
                logger.info(_log + ".\n")
                continue
            logger.info(
                _log
                + ":\n"
                + "\n".join(
                    [f"\t{match.path}: {target ^ match.sig}" for match in matches]
                )
                + "\n"
            )
            # matches = "\n".join([f"\t{match.path}: {target ^ match.sig}" for match in matches]) + "\n"
            # logger.info(f"Found {len(matches)} matches above {args.threshold} similarity" + ("\n" if len(matches) == 0 else ""))
            # for match in matches:
            #     newline = "\n" if matches.index(match) == len(matches) - 1 else ""
            #     logger.info(f"\t{match.path}: {target ^ match.sig}{newline}")
        return

    if args.robust_match:
        hash_repo = HashRepository()
        targets = [
            hash_repo.get(path)
            for path in args.robust_match
            if len(get_binaries(Path(path), silent=True)) > 0
        ]
        if any(target is None for target in targets):
            logger.error(f"Failed to hash {args.robust_match}")
            return
        # target = hash_repo.get(args.robust_match)
        # logger.info("Stacking target functions into matrix")
        target_np_functions = [
            np.stack([f.asArray() for f in target.functionFeatureList])
            for target in tqdm(targets, desc="Collecting target functions")
        ]
        binary_scores: list[dict[str, float]] = [
            {} for _ in range(len(target_np_functions))
        ]
        logger.info("Collecting binaries from database...")
        for binary in tqdm(hash_repo.binary_repo.getAll(), desc="Scoring binaries"):
            features_matrix, features = hash_repo.function_repo.get_feature_matrix(
                binary.path
            )
            scores = stacked_norms(target_np_functions, features_matrix)
            for i, score in enumerate(scores):
                binary_scores[i][binary.path] = score

        for i, target in enumerate(targets):
            print(
                f"Top 5 scores for {target.path.absolute().relative_to(Path(__file__).parent / 'binary_data')}:"
            )
            top5 = sorted(binary_scores[i].items(), key=lambda x: x[1], reverse=True)[
                :5
            ]
            spaces = max([len(str(x[0])) for x in top5])
            format = f"{{path:{{spaces}}}}: {{score}}"
            print(
                "\t"
                + "\n\t".join(
                    [
                        format.format(path=str(x[0]), score=x[1], spaces=spaces)
                        for x in top5
                    ]
                )
                + "\n"
            )

        return

    if args.stdlib:
        logger.info("Computing stdlib signatures...")
        # Get stdlib binary
        stdlib_path = "binary_data/libc.so.6"
        (
            stdlib_matrix,
            stdlib_features,
        ) = app.context.hash_repo.function_repo.get_feature_matrix(stdlib_path)
        if isinstance(ret[0], FunctionFeatures):
            # Single function
            stdlib_score = np.linalg.norm(stdlib_matrix - ret[0].asArray(), axis=1)
            # get top 5 stdlib_features
            top5 = sorted(
                zip(stdlib_score, stdlib_features), key=lambda x: x[0], reverse=False
            )[:5]
            print(f"Top 5 stdlib functions for {ret[0].function.name}:")
            spaces = max([len(str(x[1])) for x in top5])
            format = f"{{path:{{spaces}}}}: {{score}}"
            print(
                "\t"
                + "\n\t".join(
                    [
                        format.format(path=x[1].name, score=x[0], spaces=spaces)
                        for x in top5
                    ]
                )
                + "\n"
            )
            breakpoint()
        elif isinstance(ret[0], BinarySignature):
            for bin in ret:
                # best_matches = []
                logger.debug(f"Matching {bin.path} to stdlib...")
                hits = 0
                for func in bin.functionFeatureList:
                    if func.cyclomatic_complexity <= 2:
                        continue
                    stdlib_score = np.linalg.norm(
                        stdlib_matrix - func.asArray(), axis=1
                    )
                    if min(stdlib_score) == 0:
                        hits += 1
                        logger.info(f"{bin.path}:\t{func.function.name}")
                        logger.info(
                            f"{Path(stdlib_path).name}:\t{stdlib_features[np.argmin(stdlib_score)].name}\n"
                        )
                logger.info(
                    f"Found {hits} perfect matches for {bin.path} out of {len(bin.functionFeatureList)} functions"
                )

                # z_scores = (np.array(best_matches) - np.mean(best_matches)) / np.std(best_matches)
                # target_z_score = -0.9
                # match_idx = np.where(z_scores < -0.9)[0]
                # match_idx = np.where(np.array(best_matches) == 0)[0]
                # logger.info(f"Found {len(match_idx)} matches for {bin.path}")
                # for i in match_idx:
                #     logger.debug(f"{bin.path}:\t{bin.functionFeatureList[i].function.name}")
                #     logger.debug(f"{Path(stdlib_path).name}:\t{stdlib_features[i].name}\n")
                #     # logger.debug(f"Dist:\t\t{best_matches[i]}\n")
                #     breakpoint()
    if args.snmp:
        if not isinstance(ret[0], BinarySignature):
            raise NotImplementedError(
                "SNMP matching not implemented for single function yet"
            )
        logger.info("Gathering SNMP signatures...")
        if Path(".snmp_signatures").exists():
            with open(".snmp_signatures", "rb") as f:
                snmp_signatures = pickle.load(f)
        else:
            snmp_signatures = app.context.hash_repo.get_snmp_signatures()
        try:
            for target_bin in ret:
                logger.info(f"Calculating distances for {target_bin.path}")
                try:
                    jaccard_estimate = [
                        target_bin // sig
                        for sig in tqdm(snmp_signatures, desc="Jaccard Estimate")
                    ]
                    minhash_similarity = [
                        target_bin ^ sig
                        for sig in tqdm(snmp_signatures, desc="Minhash Similarity")
                    ]
                except KeyboardInterrupt:
                    breakpoint()
                logger.info("Sorting results...")
                jaccard_estimate = sorted(
                    zip(jaccard_estimate, snmp_signatures),
                    key=lambda x: x[0],
                    reverse=True,
                )
                minhash_similarity = sorted(
                    zip(minhash_similarity, snmp_signatures),
                    key=lambda x: x[0],
                    reverse=True,
                )
                jaccard_closest_bins = jaccard_estimate[
                    : list_rindex(
                        [x[0] for x in jaccard_estimate],
                        max(jaccard_estimate, key=lambda x: x[0])[0],
                    )
                    + 1
                ]
                jaccard_closest_paths = [x[1].path for x in jaccard_closest_bins]
                minhash_closest_bins = minhash_similarity[
                    : list_rindex(
                        [x[0] for x in minhash_similarity],
                        max(minhash_similarity, key=lambda x: x[0])[0],
                    )
                    + 1
                ]
                minhash_closest_paths = [x[1].path for x in minhash_closest_bins]
                if any(
                    target_bin.path.name in str(path) for path in jaccard_closest_paths
                ):
                    likely_matches = [
                        x
                        for x in jaccard_closest_bins
                        if target_bin.path.name in str(x[1].path)
                    ]
                    print(f"Jaccard likely matches for {target_bin.path}:")
                    spaces = max(
                        [
                            len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)))
                            for x in likely_matches
                        ]
                    )
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print(
                        "\t"
                        + "\n\t".join(
                            [
                                format.format(
                                    path=str(
                                        x[1].path.relative_to(NET_SNMP_BINARY_CACHE)
                                    ),
                                    score=x[0],
                                    spaces=spaces,
                                )
                                for x in likely_matches
                            ]
                        )
                        + "\n"
                    )
                else:
                    # print top 5 closest matches
                    print(f"Top 5 Jaccard Estimate matches for {target_bin.path}:")
                    spaces = max(
                        [
                            len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)))
                            for x in jaccard_estimate[:5]
                        ]
                    )
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print(
                        "\t"
                        + "\n\t".join(
                            [
                                format.format(
                                    path=str(
                                        x[1].path.relative_to(NET_SNMP_BINARY_CACHE)
                                    ),
                                    score=x[0],
                                    spaces=spaces,
                                )
                                for x in jaccard_estimate[:5]
                            ]
                        )
                        + "\n"
                    )

                if any(
                    target_bin.path.name in str(path) for path in minhash_closest_paths
                ):
                    likely_matches = [
                        x
                        for x in minhash_closest_bins
                        if target_bin.path.name in str(x[1].path)
                    ]
                    print(f"Minhash likely matches for {target_bin.path}:")
                    spaces = max(
                        [
                            len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)))
                            for x in likely_matches
                        ]
                    )
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print(
                        "\t"
                        + "\n\t".join(
                            [
                                format.format(
                                    path=str(
                                        x[1].path.relative_to(NET_SNMP_BINARY_CACHE)
                                    ),
                                    score=x[0],
                                    spaces=spaces,
                                )
                                for x in likely_matches
                            ]
                        )
                        + "\n"
                    )
                else:
                    # print top 5 closest matches
                    print(f"Top 5 Minhash Similarity matches for {target_bin.path}:")
                    spaces = max(
                        [
                            len(str(x[1].path.relative_to(NET_SNMP_BINARY_CACHE)))
                            for x in minhash_similarity[:5]
                        ]
                    )
                    format = f"{{path:{{spaces}}}}: {{score}}"
                    print(
                        "\t"
                        + "\n\t".join(
                            [
                                format.format(
                                    path=str(
                                        x[1].path.relative_to(NET_SNMP_BINARY_CACHE)
                                    ),
                                    score=x[0],
                                    spaces=spaces,
                                )
                                for x in minhash_similarity[:5]
                            ]
                        )
                        + "\n"
                    )

        except Exception as e:
            print(e)
            breakpoint()

            # print(f"Top {args.matches} Jaccard Estimate matches for {target_bin.path}:")
            # spaces = max([len(str(x[1])) for x in jaccard_estimate[:args.matches]])
            # format = f"{{path:{{spaces}}}}: {{score}}"
            # print("\t" + "\n\t".join([format.format(path=x[1].path, score=x[0], spaces=spaces) for x in jaccard_estimate[:args.matches]]) + "\n")
            # print(f"Top {args.matches} Minhash Similarity matches for {target_bin.path}:")
            # spaces = max([len(str(x[1])) for x in minhash_similarity[:args.matches]])
            # format = f"{{path:{{spaces}}}}: {{score}}"
            # print("\t" + "\n\t".join([format.format(path=x[1].path, score=x[0], spaces=spaces) for x in minhash_similarity[:args.matches]]) + "\n")

    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborting")
    except Exception as e:
        print(f"Error: {e}")
        breakpoint()
