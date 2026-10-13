"""Atomic publication shared by the offline evidence writers."""
from pathlib import Path
import tempfile


def atomic_write(path, payload):
    """Publish bytes using a private temporary file on the target filesystem."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
