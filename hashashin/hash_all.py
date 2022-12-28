import os

from hashashin.utils import cache_hash
from hashashin.lsh_test import get_binaries, BINARY_DIR
from tqdm import tqdm
from multiprocessing import Pool
import sys
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append(BINARY_DIR)
    binaries = set()
    logger.info(f"Getting binaries from {sys.argv[1:]}...")
    for path in sys.argv[1:]:
        binaries.update(set(get_binaries(path, progress=True)))
    # _ = [cache_hash(b) for b in binaries]
    logger.info(f"Hashing {len(binaries)} binaries...")
    with Pool(max(os.cpu_count() // 2, 1)) as p:
        list(tqdm(p.imap_unordered(cache_hash, binaries), total=len(binaries)))
