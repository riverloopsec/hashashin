from typing import Optional, List
import argparse
from hashashin.db import AbstractHashRepository
from hashashin.app import HashApp
from hashashin.utils import get_binaries
from hashashin.classes import BinarySignature
from hashashin.db import RepositoryType
from hashashin.sqlalchemy import SQLAlchemyHashRepository
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _options_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Match threshold"
    )


def _db_parser(parser: argparse.ArgumentParser):
    db_group = parser.add_argument_group("Database Operations")
    db_group.add_argument(
        "--summary",
        "--status",
        "-db",
        type=str,
        nargs="?",
        const="",
        metavar="GLOB",
        help="Print database summary. Optionally provide path to filter on.",
    )
    db_group.add_argument(
        "--drop",
        type=str,
        default=None,
        help='Perform database drop. Accepts "all" or binary path',
    )


def _demo_parser(parser: argparse.ArgumentParser):
    demo_group = parser.add_argument_group("Demo Operations")
    demo_group.add_argument(
        "--fast-match",
        type=str,
        nargs="+",
        metavar="BINARY_PATH",
        help="Fast match a binary against the database",
    )
    demo_group.add_argument(
        "--robust-match",
        type=str,
        nargs="+",
        metavar="BINARY_PATH",
        help="Robust match a binary against the database",
    )
    demo_group.add_argument(
        "--stdlib", action="store_true", help="Match against standard library"
    )
    demo_group.add_argument(
        "--snmp", action="store_true", help="Match against net-snmp"
    )


def _app_parser(parser: argparse.ArgumentParser):
    app_group = parser.add_argument_group("HashApp Operations")
    app_group.add_argument(
        "--hash",
        "--save",
        type=str,
        nargs="*",
        metavar="BINARY_PATH",
        help="Hash a binary or directory of binaries and save to db. Pass with --match to save and match.",
    )
    app_group.add_argument(
        "--match",
        type=str,
        nargs="+",
        metavar="BINARY_PATH",
        help="Match a binary or directory of binaries against the db. Pass paths to --save and use --match without args to save and match.",
    )


PARSER_OPTIONS = [_options_parser, _db_parser, _demo_parser, _app_parser]


def _get_parser_group_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
):
    for parser_group in parser._action_groups[2:]:
        yield list(
            filter(
                lambda option: option[0]
                in [g.dest for g in parser_group._group_actions],
                args._get_kwargs(),
            )
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    list(map(lambda x: x(parser), PARSER_OPTIONS))
    return parser


def validate_args(args, parser=None) -> None:
    if parser is None:
        parser = get_parser()
    parser_group_options = list(_get_parser_group_options(parser, args))
    for pg in parser_group_options:
        if any(x[1] is not None and x[1] is not False for x in pg):
            break
    else:
        parser.error("Invalid arguments. See --help for more information.")


def _db_handler(args: argparse.Namespace, db: AbstractHashRepository) -> bool:
    logger.debug(
        "DB args: "
        + str(
            list(_get_parser_group_options(get_parser(), args))[
                PARSER_OPTIONS.index(_db_parser) - 1
            ]
        )
    )

    if getattr(args, "summary", None) is not None:
        HashApp._log_summary(db, args.summary)
        return False  # Continue execution if summary is printed
    if getattr(args, "drop", None) is not None:
        if not input(f"Confirm drop {args.drop}? [y/N] ").lower().startswith("y"):
            logger.info("Aborting drop")
            return True
        db.drop(args.drop)
        return True  # Exit after drop
    return False


def _demo_handler(args: argparse.Namespace, app: HashApp) -> bool:
    logger.debug(
        "Demo args: "
        + str(
            list(_get_parser_group_options(get_parser(), args))[
                PARSER_OPTIONS.index(_demo_parser) - 1
            ]
        )
    )

    if getattr(args, "fast_match", None) is not None:
        if not isinstance(app.repo, SQLAlchemyHashRepository):
            raise NotImplementedError
        logger.debug("Fast matching binaries")
        for target in map(Path, args.fast_match):
            if not target.is_file() or len(get_binaries(target, silent=True)) == 0:
                logger.debug(f"Skipping {target} as it is not a binary")
                continue
            hashed_target = app.hash_path(target)
            if hashed_target is None:
                logger.error(f"Failed to hash {target}")
                continue
            if len(hashed_target) != 1:
                logger.error(f"Something went wrong...")
                breakpoint()
                raise Exception
            hashed_target = hashed_target[0]
            logger.info(f"Fast matching {hashed_target.path}")
            matches = app.repo.binary_repo.fast_match(target, args.threshold)
            matches = list(filter(lambda x: x.path != (target.path), matches))
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
        return True
    if getattr(args, "robust_match", None) is not None:
        if not isinstance(app.repo, SQLAlchemyHashRepository):
            raise NotImplementedError
        logger.debug("Robust matching binaries")
        top5 = app.match(args.robust_match, n=5)
        for target, matches in top5.items():
            logger.info(
                f"Robust matching {target}:\n"
                + "\n".join(
                    [f"\t{match.path}: {target ^ match.sig}" for match in matches]
                )
                + "\n"
            )
        return True
    if getattr(args, "stdlib", None):
        raise NotImplementedError
    if getattr(args, "snmp", None):
        raise NotImplementedError
    return False


def hash_binaries(binaries: List[str], app: HashApp) -> List[BinarySignature]:
    # TODO: multi-threading
    logger.debug("Hashing binaries")
    return app.hash_list(map(Path, binaries))


def _app_handler(args: argparse.Namespace, app: HashApp) -> bool:
    logger.debug(
        "App args: "
        + str(
            list(_get_parser_group_options(get_parser(), args))[
                PARSER_OPTIONS.index(_app_parser) - 1
            ]
        )
    )

    hash_bins = list()
    if getattr(args, "hash", None) is not None:
        logger.info(
            f"Saving {len(hash_bins := hash_binaries(args.hash, app))} binaries to db."
        )
        app.save(hash_bins)

    if getattr(args, "match", None) is not None:
        logger.info("Matching binaries")
        match_bins = list()
        for target in map(Path, args.match):
            match_bins.extend(app.hash_path(target))
        for target in hash_bins + match_bins:
            matches = app.match(target)
            logger.info(
                f"Matched {target.path}:\n"
                + "\n".join(
                    [f"\t{match.signature.path}: {match.score}" for match in matches]
                )
                + "\n"
            )

    return bool(getattr(args, "match", False) or getattr(args, "hash", False))


def cli_handler(
    args: argparse.Namespace,
    parser: Optional[argparse.ArgumentParser] = None,
    app: Optional[HashApp] = None,
):
    validate_args(args, parser)

    level = logging.DEBUG if getattr(args, "debug", False) else logging.INFO
    HashApp._initialize_logger(level)

    if app is None:
        app = HashApp.from_type(RepositoryType.SQLALCHEMY, extractor="binja")

    if _db_handler(args, app.repo):
        return
    if _demo_handler(args, app):
        return
    if _app_handler(args, app):
        return


def cli(args: Optional[argparse.Namespace] = None):
    parser = get_parser()
    if args is None:
        args = parser.parse_args()

    cli_handler(args)
