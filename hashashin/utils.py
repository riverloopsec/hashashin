from __future__ import annotations
import glob
import json
import logging
import os
from pathlib import Path
from typing import Union, Iterable
import git
import re
import urllib.request
from io import BytesIO
import tarfile
import subprocess

import binaryninja  # type: ignore
import magic
import numpy as np
import numpy.typing as npt
from tqdm import tqdm  # type: ignore

logger = logging.getLogger(__name__)


def get_binaries(
    path: Union[Path, Iterable[Path]], bin_name=None, recursive=True, progress=False, silent=False
) -> list[Path]:
    """Get all binaries in a directory"""
    globber = "*" + "*" * recursive
    if not isinstance(path, (str, Path)) and isinstance(path, Iterable):
        files = []
        for p in path:
            files.extend(get_binaries(p, bin_name, recursive, progress))
        return files
    elif os.path.isfile(path):
        files = [path]
    elif bin_name is None:
        files = list(path.glob(globber + "/*"))
    else:
        files = list(path.glob(globber + f"/{bin_name}"))
    if (Path(path) / ".bin.idx").is_file():
        logger.debug(f"Loading cached binary paths from {Path(path) / '.bin.idx'}")
        with open(Path(path) / ".bin.idx", "r") as f:
            binaries = [Path(b) for b in json.load(f)]
        return binaries
    binaries = []
    if not progress:
        if "progress_warning" not in dir(logger):
            if not silent:
                logger.info(
                    f"Iterating over {len(files)} files. If you see this, consider using --progress.",
                )
                logger.progress_warning = True
    elif len(files) == 1:
        progress = False
    for f in tqdm(
        files,
        disable=not progress,
        desc=f"Gathering binaries in {os.path.relpath(path)}",
    ):
        if f.is_file():
            if "ELF" in magic.from_file(f):
                binaries.append(f)
    if Path(path).is_dir():
        # cache list of binaries in directory
        with open(path / ".bin.idx", "w") as f:
            json.dump([str(b) for b in binaries], f)
    return binaries


def split_int_to_uint32(x: int, pad=None, wrap=False) -> npt.NDArray[np.uint32]:
    """Split very large integers into array of uint32 values. Lowest bits are first."""
    _logger = logger.getChild("utils.split_int_to_uint32")
    if x < np.iinfo(np.uint32).max:
        if pad is None:
            return np.array([x], dtype=np.uint32)
        return np.pad([x], (0, pad - 1), "constant", constant_values=(0, 0))
    if pad is None:
        pad = int(np.ceil(len(bin(x)) / 32))
    elif pad < int(np.ceil(len(bin(x)) / 32)):
        if wrap:
            _logger.debug(f"Padding is too small for number {x}, wrapping")
            x = x % (2 ** (32 * pad))
        else:
            raise ValueError("Padding is too small for number")
    ret = np.array([(x >> (32 * i)) & 0xFFFFFFFF for i in range(pad)], dtype=np.uint32)
    assert merge_uint32_to_int(ret) == x, f"{merge_uint32_to_int(ret)} != {x}"
    if merge_uint32_to_int(ret) != x:
        _logger.warning(f"{merge_uint32_to_int(ret)} != {x}")
        raise ValueError("Splitting integer failed")
    return ret


def merge_uint32_to_int(x: npt.NDArray[np.uint32]) -> int:
    """Merge array of uint32 values into a single integer. Lowest bits first."""
    ret = 0
    for i, v in enumerate(x):
        ret |= int(v) << (32 * i)
    return ret


def bytes_distance(a: bytes, b: bytes) -> float:
    """Calculate the distance between two byte strings"""
    return np.mean(np.frombuffer(a, dtype=np.uint) != np.frombuffer(b, dtype=np.uint8))


def compute_net_snmp_db(output_dir: Union[str, Path] = Path(__file__).parent / "binary_data/net-snmp"):
    """Download the net-snmp database from GitHub and add signatures to db"""
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_url = "git@github.com:net-snmp/net-snmp.git"
    # clone repo if not already present
    if not (output_dir / ".git").is_dir():
        logger.info(f"Cloning {repo_url} to {output_dir}")
        repo = git.Repo.clone_from(repo_url, output_dir)
    else:
        logger.info(f"Found existing repo in {output_dir}")
        repo = git.Repo(output_dir)
    tags = [t for t in repo.tags if re.match(r"^v[0-9]+\.[0-9]+\.[0-9]+$", t.name)]
    for tag in tags:
        logger.info(f"Checking out {tag.name}")
        repo.git.checkout(tag)
        breakpoint()
        # logger.info(f"Running autoreconf")
        # subprocess.run(["autoreconf", "-fvi"], cwd=output_dir, check=True)
        # logger.info(f"Running configure")
        # subprocess.run(["./configure"], cwd=output_dir, check=True)
        # logger.info(f"Running make")
        # subprocess.run(["make"], cwd=output_dir, check=True)


    # repo = git.Repo.init(path=None)
    # if 'snmp-origin' not in [r.name for r in repo.remotes]:
    #     repo.create_remote('snmp-origin', repo_url)
    #     repo.remote('snmp-origin').fetch(tags=True)
    # tags = [t for t in repo.tags if re.match(r"^v[0-9]+\.[0-9]+\.[0-9]+$", t.name)]
    # tgz_path = "https://github.com/net-snmp/net-snmp/archive/refs/tags/%s.tar.gz"
    # for tag in reversed(tags):
    #     response = urllib.request.urlopen(tgz_path % tag.name)
    #     compressed_file = BytesIO(response.read())
    #     with tarfile.open(fileobj=compressed_file, mode="r:gz") as tar:
    #         tar.extractall(path=output_dir)
    #     breakpoint()
