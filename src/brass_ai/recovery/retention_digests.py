"""Ephemeral cleanup-plan receipts. Final quarantine deletion rehashes bytes."""
from concurrent.futures import ThreadPoolExecutor
import stat
from threading import Lock
from brass_ai.bank import sha
from brass_ai.learning.contracts import require

# ctime/mtime are not content versions, including on POSIX filesystems. The
# g1250 lab reproduced a same-size rewrite with the complete stat stamp intact.
# Keep this metric/API for existing reports, but never authorize a cache hit
# from metadata alone. Final quarantine deletion still performs its own hash.
METADATA_REUSE=False
HASH_WORKERS=4


def stamp(path):
    s=path.stat(follow_symlinks=False)
    require(stat.S_ISREG(s.st_mode),'Cleanup hash requires a regular file')
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink)


def hash_batch(paths,digest):
    """Read independent files in parallel, retaining the journal's order."""
    if len(paths)<32:return list(map(digest,paths))
    with ThreadPoolExecutor(max_workers=HASH_WORKERS,thread_name_prefix='retention-hash') as pool:
        return list(pool.map(digest,paths))


class PlanDigests:
    def __init__(self):
        self.read_bytes=0;self.reused_bytes=0;self.hits=0;self.lock=Lock()

    def digest(self,path):
        return self.measure(path)[0]

    def measure(self,path):
        before=stamp(path)
        value=sha(path);after=stamp(path)
        require(after==before,'Artifact changed while hashing cleanup plan')
        with self.lock:self.read_bytes+=after[2]
        return value,after[2]

    def report(self, *, archived_bytes=0):
        return dict(plan_and_move_read_bytes=self.read_bytes+archived_bytes,reused_read_bytes=self.reused_bytes,
            reuse_hits=self.hits,metadata_reuse_supported=METADATA_REUSE,final_delete_rehash=True,
            scope='Apply hashes before moving and rehashes before deletion. Report archival hashes the same bytes it parses.')
