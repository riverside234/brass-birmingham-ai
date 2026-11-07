"""Restore the sealed, public Roslyn toolset into qualification evidence only."""
import hashlib
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'distribution/rust-lab-toolchain/microsoft.net.compilers.toolset.4.12.0.nupkg'
SHA256 = 'fe24ef31a6ffcb7c49383d2fd362763dee291ad9b9d98cc0c19ef80203b99ebc'
PREFIX = 'tasks/net472/'


def restore(out, *, root=ROOT):
    """Do not depend on a Windows developer cache or download during a gate."""
    package = Path(root) / PACKAGE
    if not package.is_file() or hashlib.sha256(package.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Missing/changed pinned Roslyn package; extract the current qualification bundle: ' + PACKAGE)
    target = Path(out).resolve() / 'toolchain/roslyn-4.12.0'
    if target.exists():
        raise ValueError('Use a fresh qualification directory; compiler target already exists')
    with zipfile.ZipFile(package) as archive:
        selected = [i for i in archive.infolist() if i.filename.startswith(PREFIX) and not i.is_dir()]
        names = [i.filename for i in selected]
        if (not selected or len(names) != len(set(names)) or PREFIX + 'csc.exe' not in names
                or sum(i.file_size for i in selected) > 100 * 1024**2):
            raise ValueError('Invalid pinned Roslyn layout')
        destinations = []
        for item in selected:
            relative = PurePosixPath(item.filename[len(PREFIX):])
            path = (target / relative).resolve()
            if (relative.is_absolute() or '..' in relative.parts or '\\' in item.orig_filename
                    or ':' in item.filename or not path.is_relative_to(target)
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe Roslyn package entry: ' + item.filename)
            destinations.append((item, path))
        target.mkdir(parents=True, exist_ok=False)
        for item, path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(archive.read(item))
    return target / 'csc.exe'
