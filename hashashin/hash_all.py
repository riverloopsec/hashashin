import os

from hashashin.utils import cache_hash
from hashashin.lsh_tests import get_binaries, BINARY_DIR
from tqdm import tqdm
from multiprocessing import Pool
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append(BINARY_DIR)
    binaries = set()
    for path in sys.argv[1:]:
        binaries.update(set(get_binaries(path, progress=True)))
    # _ = [cache_hash(b) for b in binaries]
    with Pool(max(os.cpu_count() // 2, 1)) as p:
        list(tqdm(p.imap_unordered(cache_hash, binaries), total=len(binaries)))
