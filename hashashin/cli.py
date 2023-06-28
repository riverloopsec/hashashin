from typing import Optional
import argparse
from hashashin.app import HashApp
from hashashin.utils import get_binaries
from hashashin.classes import BinarySignature
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

APP: HashApp


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
        help="Match a binary or directory of binaries against the db.",
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


def _db_handler(args: argparse.Namespace) -> bool:
    logger.debug(
        "DB args: "
        + str(
            list(_get_parser_group_options(get_parser(), args))[
                PARSER_OPTIONS.index(_db_parser) - 1
            ]
        )
    )

    if getattr(args, 'summary', None) is not None:
        logger.debug("Printing database summary")
        num_binaries, num_functions = APP.repo.binary_repo.summary(args.summary)
        msg = f"*{args.summary}*" if args.summary else "all"
        logger.info(f"Summary for {msg} binary paths:")
        logger.info(f"\tBinaries: {num_binaries}")
        logger.info(f"\tFunctions: {num_functions}")
        return False  # Continue execution if summary is printed
    if getattr(args, 'drop', None) is not None:
        if not input(f"Confirm drop {args.drop}? [y/N] ").lower().startswith("y"):
            logger.info("Aborting drop")
            return True
        APP.repo.drop(args.drop)
        return True  # Exit after drop
    return False


def _demo_handler(args: argparse.Namespace) -> bool:
    logger.debug(
        "Demo args: "
        + str(
            list(_get_parser_group_options(get_parser(), args))[
                PARSER_OPTIONS.index(_demo_parser) - 1
            ]
        )
    )

    if getattr(args, 'fast_match', None) is not None:
        logger.debug("Fast matching binaries")
        for target in map(Path, args.fast_match):
            if not target.is_file() or len(get_binaries(target, silent=True)) == 0:
                logger.debug(f"Skipping {target} as it is not a binary")
                continue
            hashed_target = APP.hash(target)
            if hashed_target is None:
                logger.error(f"Failed to hash {target}")
                continue
            if len(hashed_target) != 1:
                logger.error(f"Something went wrong...")
                breakpoint()
                raise Exception
            hashed_target = hashed_target[0]
            logger.info(f"Fast matching {hashed_target.path}")
            matches = APP.repo.binary_repo.fast_match(target, args.threshold)
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
    if getattr(args, 'robust_match', None) is not None:
        logger.debug("Robust matching binaries")
        top5 = APP.match(args.robust_match, n=5)
        for target, matches in top5.items():
            logger.info(
                f"Robust matching {target}:\n"
                + "\n".join(
                    [f"\t{match.path}: {target ^ match.sig}" for match in matches]
                )
                + "\n"
            )
        return True
    if getattr(args, 'stdlib', None):
        raise NotImplementedError
    if getattr(args, 'snmp', None):
        raise NotImplementedError
    return False


def hash_binaries(binaries: list[str]) -> list[BinarySignature]:
    logger.debug("Hashing binaries")
    bins = list()
    for target in map(Path, binaries):
        if not target.is_file() or len(get_binaries(target, silent=True)) == 0:
            logger.debug(f"Skipping {target} as it is not a binary")
            continue
        logger.info(f"Hashing {target}")
        bins.extend(APP.hash(target))
    return bins


def _app_handler(args: argparse.Namespace) -> bool:
    logger.debug(
        "App args: "
        + str(
            list(_get_parser_group_options(get_parser(), args))[
                PARSER_OPTIONS.index(_app_parser) - 1
            ]
        )
    )

    if getattr(args, 'hash', None) is not None:
        logger.info(f"Saving {len(hash_bins := hash_binaries(args.hash))} binaries to db.")
        APP.repo.save(hash_bins)

    if getattr(args, 'match', None) is not None:
        logger.debug("Matching binaries")
        match_bins = APP.hash(args.match)
        if getattr(args, 'hash', None) is not None:
            logger.info(f"Saving {len(match_bins)} binaries to db.")
            APP.repo.save(match_bins)
        top10 = APP.match(match_bins, n=10)
        raise NotImplementedError

    return bool(args.match or args.hash)


def cli_handler(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None):
    validate_args(args, parser)

    level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-6s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=level,
    )

    global APP
    APP = HashApp()
    if _db_handler(args):
        return
    if _demo_handler(args):
        return
    if _app_handler(args):
        return


def cli(args: Optional[argparse.Namespace] = None):
    parser = get_parser()
    if args is None:
        args = parser.parse_args()

    cli_handler(args)
