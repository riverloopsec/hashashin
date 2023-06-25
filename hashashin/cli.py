from typing import Optional
import argparse
from hashashin.app import HashApp
import logging


def _db_parser(parser: argparse.ArgumentParser):
    db_group = parser.add_argument_group("Database operations")
    # db_group.add_argument(
    #     "--status", "-db", action="store_true", help="Print database status"
    # )
    db_group.add_argument(
        "--drop",
        type=str,
        default=None,
        help='Drop database tables. Accepts "all" or binary path',
    )
    db_group.add_argument(
        "--summary", action="store_true", help="Print database summary"
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


def _dev_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )


PARSER_OPTIONS = [_db_parser, _demo_parser, _dev_parser]


def _get_parser_group_options(parser: argparse.ArgumentParser, args: argparse.Namespace):
    for parser_group in parser._action_groups[2:]:
        yield filter(
            lambda g: g[0] in [g.dest for g in parser_group._group_actions],
            args._get_kwargs(),
        )


def get_parser():
    parser = argparse.ArgumentParser()
    list(map(lambda x: x(parser), PARSER_OPTIONS))
    return parser


def _db_handler(args: argparse.Namespace):
    if args.summary:
        HashApp().repo.summary()
    if args.drop is not None:
        HashApp().repo.drop(args.drop) # TODO: implement drop on repo


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
    
    
