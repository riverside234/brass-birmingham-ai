"""Ephemeral cleanup-plan receipts. Final quarantine deletion rehashes bytes."""
import stat
import os
from brass_ai.bank import sha
from brass_ai.learning.contracts import require

METADATA_REUSE=os.name!='nt'


def stamp(path):
    s=path.stat(follow_symlinks=False)
    require(stat.S_ISREG(s.st_mode),'Cleanup hash requires a regular file')
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_nlink)


class PlanDigests:
    def __init__(self):
        self.cache={};self.read_bytes=0;self.reused_bytes=0;self.hits=0

    def digest(self,path):
        before=stamp(path);old=self.cache.get(path)
        # No reuse on filesystems without an identity. A changed stamp forces
        # a new read, including same-size edits with restored mtime (ctime).
        # Windows ctime is creation time; observed same-size rapid writes can
        # retain the complete stat stamp. Keep full reads on that platform.
        if METADATA_REUSE and before[1] and old is not None and old[0]==before:
            self.hits+=1;self.reused_bytes+=before[2];return old[1]
        value=sha(path);after=stamp(path)
        require(after==before,'Artifact changed while hashing cleanup plan')
        self.cache[path]=(after,value);self.read_bytes+=after[2]
        return value

    def report(self):
        return dict(plan_and_move_read_bytes=self.read_bytes,reused_read_bytes=self.reused_bytes,
            reuse_hits=self.hits,metadata_reuse_supported=METADATA_REUSE,final_delete_rehash=True,
            scope='Only unchanged same-path metadata within this prune call; no persisted hash trust or rename reuse')
