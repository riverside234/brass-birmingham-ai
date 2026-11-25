"""Confined project paths, including byte-preserving Step 5 relocations.

Only this small, stdlib-only module is shared with the transfer CLI. Historical
references stay signed as written; a local sidecar maps whole root components
at file-access time. It is deliberately independent of cwd and environment vars.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat

RELOCATION_FILE = '.step5-relocation.json'
RELOCATION_FORMAT = 'step5-project-relocation/1.0.0'


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _parts(value):
    value = os.fspath(value)
    _require(isinstance(value, str) and value and '\0' not in value, 'Invalid project path')
    win = PureWindowsPath(value)
    if win.drive:
        _require(win.is_absolute() and len(win.drive) == 2 and win.drive[1] == ':',
                 'Unsupported drive-relative/UNC project path: ' + value)
        flavor, path = 'windows', win
    else:
        _require(not value.startswith('\\'), 'Ambiguous rooted project path: ' + value)
        flavor, path = 'posix', PurePosixPath(value.replace('\\', '/'))
    _require('..' not in path.parts, 'Path escapes project root (parent traversal): ' + value)
    _require(all(not p.endswith((' ', '.')) for p in (path.parts[1:] if path.is_absolute() else path.parts)),
             'Unsafe trailing space/dot in project path: ' + value)
    _require(all(':' not in p for p in path.parts[1:] if path.is_absolute()) and
             (path.is_absolute() or ':' not in value), 'Invalid project path component: ' + value)
    return flavor, path


def root_key(value):
    """Host-independent absolute-root identity (Windows components case-fold)."""
    flavor, path = _parts(value)
    _require(path.is_absolute() and len(path.parts) > 1, 'Expected a non-filesystem absolute project root: ' + str(value))
    parts = tuple(p.casefold() for p in path.parts) if flavor == 'windows' else path.parts
    return flavor, parts


def validate_roots(roots):
    _require(isinstance(roots, list) and len(roots) <= 64 and all(isinstance(r, str) for r in roots),
             'Invalid relocation source roots')
    keys = [root_key(r) for r in roots]
    for i, (flavor, parts) in enumerate(keys):
        for other_flavor, other in keys[:i]:
            _require(flavor != other_flavor or (parts[:len(other)] != other and other[:len(parts)] != parts),
                     'Ambiguous/duplicate overlapping relocation roots; choose disjoint project directories')
    return roots


def confined(root, suffix):
    """Join components below a canonical root, refusing links/junctions."""
    root = Path(root)
    _require(root.is_absolute(), 'Expected a canonical absolute destination root')
    _require(not root.exists() or root.is_dir(), 'Non-directory project root: ' + str(root))
    path = root.joinpath(*suffix)
    _require(path.is_relative_to(root), 'Path escapes project root: ' + str(path))
    for item in (path, *path.parents):
        if item == root:
            break
        if item.exists() or item.is_symlink():
            info = item.lstat()
            _require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                     'Linked project path (symlink/junction forbidden): ' + str(item))
            _require(item == path or stat.S_ISDIR(info.st_mode), 'Non-directory project parent: ' + str(item))
    _require(path.resolve().is_relative_to(root), 'Resolved path escapes project root: ' + str(path))
    return path


def _read_relocation(root):
    path = confined(root, (RELOCATION_FILE,))
    if not path.exists():
        return None
    _require(path.is_file() and path.stat().st_size < 65536, 'Invalid/oversized relocation metadata')
    def unique(pairs):
        result = dict(pairs)
        _require(len(result) == len(pairs), 'Duplicate relocation metadata field')
        return result
    try:
        data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique)
        _require(isinstance(data, dict) and set(data) == {'format', 'destination_root', 'source_roots', 'archive_id', 'id'},
                 'Invalid relocation metadata fields')
        _require(data['format'] == RELOCATION_FORMAT and data['id'] == _digest({k: v for k, v in data.items() if k != 'id'}),
                 'Invalid relocation metadata digest/format')
        _require(root_key(data['destination_root']) == root_key(str(root)),
                 'Relocation metadata belongs to another destination; use transfer export/restore --relocate to move again')
        validate_roots(data['source_roots'])
        validate_roots([str(root), *data['source_roots']])
        _require(isinstance(data['archive_id'], str) and len(data['archive_id']) == 64 and
                 all(c in '0123456789abcdef' for c in data['archive_id']), 'Invalid relocation archive receipt')
    except (TypeError, KeyError, json.JSONDecodeError) as error:
        raise ValueError('Invalid relocation metadata: ' + str(path)) from error
    return data


def read_relocation(root):
    return _read_relocation(Path(root).resolve())


def make_relocation(root, roots, archive_id):
    root = Path(root).resolve()
    aliases = [r for r in validate_roots(roots) if root_key(r) != root_key(str(root))]
    validate_roots([str(root), *aliases])
    data = dict(format=RELOCATION_FORMAT, destination_root=str(root), source_roots=aliases, archive_id=archive_id)
    data['id'] = _digest(data)
    return data


def source_roots(root):
    root = Path(root).resolve()
    data = _read_relocation(root)
    return [str(root), *(data['source_roots'] if data else [])]


def reference_suffix(roots, reference):
    """Return project components for an absolute reference, or None if relative."""
    flavor, path = _parts(reference)
    if not path.is_absolute():
        return None
    parts = tuple(p.casefold() for p in path.parts) if flavor == 'windows' else path.parts
    for origin in roots:
        origin_flavor, prefix = root_key(origin)
        if flavor == origin_flavor and parts[:len(prefix)] == prefix:
            return path.parts[len(prefix):]
    raise ValueError('Path outside project root and relocation mappings: ' + str(reference) +
                     '. Supply this dependency inside the project before export; arbitrary external roots are unsupported.')


def resolve_project_path(root, reference, *, base=None):
    """Resolve an absolute alias or a relative path, never consulting old roots.

    Relative paths remain root-relative unless a schema explicitly defines a
    local base (e.g. an IL checkpoint filename). Mapping is by components only.
    """
    root = Path(root).resolve()
    data = _read_relocation(root)  # Validate persistent metadata even for relative refs.
    roots = [str(root), *(data['source_roots'] if data else [])]
    suffix = reference_suffix(roots, reference)
    if suffix is not None:
        return confined(root, suffix)
    _, path = _parts(reference)
    anchor = root if base is None else resolve_project_path(root, base)
    return confined(root, (*anchor.relative_to(root).parts, *path.parts))


class PassResolver:
    """resolve_project_path for one read pass over many references (the retention pin scan, g5466).

    The same returned paths and the same refusals, but the root, its relocation metadata and each directory's
    link check are read once per pass instead of once per reference: a scan of tens of thousands of references
    spent most of its time re-reading the relocation file and lstat-ing the same parents. confined()'s final
    resolve() is redundant here: _parts() refuses '..' and every existing component is lstat-ed and refused if it
    is a symlink or junction, so the joined path is its own resolution. Build a new resolver for every pass; never
    keep one across passes (a directory checked in one pass may be replaced before the next)."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        _require(not self.root.exists() or self.root.is_dir(), 'Non-directory project root: ' + str(self.root))
        data = _read_relocation(self.root)  # validated once per pass, as every resolve_project_path call does
        self.roots = [str(self.root), *(data['source_roots'] if data else [])]
        self._keys = [root_key(r) for r in self.roots]
        self._root = str(self.root)
        self._checked = {}  # component tuple -> (exists, is_dir), for this pass only
        self._anchors = {}  # base -> its component tuple

    def _component(self, parts, leaf):
        hit = self._checked.get(parts)
        if hit is None:
            item = os.path.join(self._root, *parts)
            try:
                info = os.lstat(item)
            except (FileNotFoundError, NotADirectoryError):
                info = None
            if info is not None and (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400):
                raise ValueError('Linked project path (symlink/junction forbidden): ' + item)
            hit = (info is not None, info is not None and stat.S_ISDIR(info.st_mode))
            self._checked[parts] = hit
        if hit[0] and not leaf and not hit[1]:
            raise ValueError('Non-directory project parent: ' + os.path.join(self._root, *parts))

    def _suffix(self, reference, base):
        """The reference's components below the root (reference_suffix, or the base-relative join)."""
        flavor, path = _parts(reference)
        if path.is_absolute():
            parts = tuple(p.casefold() for p in path.parts) if flavor == 'windows' else path.parts
            for origin_flavor, prefix in self._keys:
                if flavor == origin_flavor and parts[:len(prefix)] == prefix:
                    return tuple(path.parts[len(prefix):])
            raise ValueError('Path outside project root and relocation mappings: ' + str(reference) +
                             '. Supply this dependency inside the project before export; arbitrary external roots are unsupported.')
        if base is None:
            return path.parts
        anchor = self._anchors.get(base)
        if anchor is None:
            anchor = self._anchors[base] = self._confine(self._suffix(base, None))
        return (*anchor, *path.parts)

    def _confine(self, suffix):
        for part in suffix:
            if not part or part == '..' or os.path.isabs(part):
                raise ValueError('Path escapes project root: ' + os.path.join(self._root, *suffix))
        for i in range(1, len(suffix) + 1):
            self._component(suffix[:i], i == len(suffix))
        return suffix

    def confined(self, suffix):
        return self.root.joinpath(*self._confine(tuple(suffix)))

    def resolve(self, reference, *, base=None):
        return self.root.joinpath(*self._confine(self._suffix(reference, base)))


def bound_reference(root, actual, recorded, fields):
    """Keep the historical signed representation after checking exact identity.

    Only named path fields may differ textually; digests, model IDs, provenance
    and all other values must still be exactly equal. Never rewrite evidence.
    """
    _require(isinstance(recorded, dict) and set(actual) == set(recorded), 'Reference fields differ')
    for key, value in actual.items():
        same = (resolve_project_path(root, value) == resolve_project_path(root, recorded[key])
                if key in fields else value == recorded[key])
        _require(same, 'Bound reference differs: ' + key)
    return recorded
