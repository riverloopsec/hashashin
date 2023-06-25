from typing import Optional
import argparse
from hashashin.app import HashApp
from hashashin.utils import get_binaries
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

APP: HashApp


def _db_parser(parser: argparse.ArgumentParser):
    db_group = parser.add_argument_group("Database operations")
    # db_group.add_argument(
    #     "--status", "-db", action="store_true", help="Print database status"
    # )
    db_group.add_argument(
        "--summary", action="store_true", help="Print database summary"
    )
    db_group.add_argument(
        "--drop",
        type=str,
        default=None,
        help='Drop database tables. Accepts "all" or binary path',
    )


def _demo_parser(parser: argparse.ArgumentParser):
    demo_group = parser.add_argument_group("Demo operations")
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


def _optional_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Match threshold"
    )


PARSER_OPTIONS = [_db_parser, _demo_parser, _optional_parser]


def _get_parser_group_options(parser: argparse.ArgumentParser, args: argparse.Namespace):
    for parser_group in parser._action_groups[2:]:
        yield list(filter(
            lambda option: option[0] in [g.dest for g in parser_group._group_actions],
            args._get_kwargs(),
        ))


def get_parser():
    parser = argparse.ArgumentParser()
    list(map(lambda x: x(parser), PARSER_OPTIONS))
    return parser


def _db_handler(args: argparse.Namespace):
    if args.summary:
        logger.debug("Printing database summary")
        APP.repo.binary_repo.summary()
        return True
    if args.drop is not None:
        if not input(f"Confirm drop {args.drop}? [y/N] ").lower().startswith("y"):
            logger.info("Aborting drop")
            return True
        APP.repo.drop(args.drop)
        return True
    return False


def _demo_handler(args: argparse.Namespace):
    if args.fast_match is not None:
        logger.debug("Fast matching binaries")
        for target in map(Path, args.fast_match):
            if not target.is_file() or len(get_binaries(target, silent=True)) == 0:
                logger.debug(f"Skipping {target} as it is not a binary")
                continue
            hashed_target = APP.hash_binary(target)
            if hashed_target is None:
                logger.error(f"Failed to hash {target}")
                continue
            logger.info(
                f"Fast matching {hashed_target.path}"
            )
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
    if args.robust_match is not None:
        logger.debug("Robust matching binaries")
        APP.robust_match(args.robust_match, args.stdlib, args.snmp)
        return True
    return False


def cli(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = (parser := get_parser()).parse_args()

        # Must utilize some argument group
        parser_group_options = list(_get_parser_group_options(parser, args))
        for pg in parser_group_options:
            if any(x[1] for x in pg):
                break
        else:
            parser.error("Invalid arguments. See --help for more information.")
    
    level = logging.DEBUG if args.debug else logging.INFO
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
    
    
