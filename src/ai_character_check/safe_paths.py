"""Decide whether a resource path referenced by a model file may be read.

A model can name external files: glTF ``buffers[].uri`` and ``images[].uri``,
FBX texture file names. The checker may run on untrusted downloads, so by
default only files inside the model's own folder are read. Everything else
(parent-folder escapes, absolute paths, ``file:`` and other URI schemes, UNC
and other network paths, links that lead out of the folder) is refused before
any file system call touches the target. On Windows, merely probing a UNC path
can send the user's NTLM credentials to a remote SMB server.

With ``allow_external=True`` local paths outside the folder are allowed.
Network paths and URI schemes other than ``data:`` stay blocked.

This module uses only the standard library: the Blender scripts load it by
file path inside Blender's own Python.
"""

from __future__ import annotations

import os
import re
import stat
import urllib.parse
from typing import Optional

MAX_LINK_HOPS = 40
_IS_WINDOWS = os.name == "nt"
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_DRIVE = re.compile(r"^[A-Za-z]:")

# Result of a check: (path, None) when the file may be read, (None, reason) when blocked.
Result = tuple[Optional[str], Optional[str]]


class _Blocked(Exception):
    """Always blocked: network paths, odd syntax."""


class _Outside(_Blocked):
    """A local path outside the model folder: blocked unless allow_external is set."""


def check_uri(uri: str, base_dir: str, allow_external: bool = False) -> Result:
    """Check a glTF URI (not a ``data:`` URI) relative to base_dir.

    Returns the file system path to read, or a reason why it is blocked.
    """
    if not isinstance(uri, str) or not uri:
        return None, "empty uri"
    m = _SCHEME.match(uri)
    if m and len(m.group(1)) > 1:  # a single letter is a Windows drive, handled as a path below
        return None, f"uri scheme '{m.group(1).lower()}:' is not allowed (only data: and relative paths)"
    try:
        decoded = urllib.parse.unquote(uri, errors="strict")
    except UnicodeDecodeError:
        return None, "uri contains invalid percent-encoding"
    return check_path(decoded, base_dir, allow_external)


def check_path(path: str, base_dir: str, allow_external: bool = False) -> Result:
    """Check a decoded file system path, relative to base_dir or absolute."""
    try:
        return _check_path(path, base_dir, allow_external), None
    except _Blocked as exc:
        return None, str(exc)


# --------------------------------------------------------------------------


def _classify(text: str) -> tuple[str, str]:
    """Return (kind, rest) with kind 'relative', 'absolute' (rest keeps the anchor) or raise _Blocked."""
    if "\x00" in text:
        raise _Blocked("path contains a NUL character")
    s = text.replace("\\", "/")
    if s.startswith("//"):
        # \\server\share, //server/share, \\?\UNC\..., \\.\pipe\..., \\?\C:\...: never touched.
        raise _Blocked("network (UNC) or device path")
    if _DRIVE.match(s):
        if not _IS_WINDOWS:
            raise _Blocked("Windows drive path")
        if len(s) == 2 or s[2] != "/":
            raise _Blocked("drive-relative Windows path")
        if ":" in s[2:]:
            raise _Blocked("path contains a colon")
        return "absolute", s
    if ":" in s:
        # Windows alternate data streams, device names like "CON:", schemes after decoding.
        raise _Blocked("path contains a colon")
    if s.startswith("/"):
        return "absolute", s
    return "relative", s


def _check_component(part: str) -> None:
    if part.rstrip(". ") == "" and part not in (".", ".."):
        # "...", ". " and similar: Windows trims trailing dots and spaces.
        raise _Blocked(f"suspicious path component {part!r}")


def _lexical_parts(s: str) -> list[str]:
    """Split a '/'-separated relative path and apply '.' and '..' lexically.

    Raises _Blocked when '..' would climb above the start.
    """
    out: list[str] = []
    for part in s.split("/"):
        if part in ("", "."):
            continue
        _check_component(part)
        if part == "..":
            if not out:
                raise _Outside("path leaves the model folder")
            out.pop()
        else:
            out.append(part)
    return out


def _relative_to(path: str, root: str) -> Optional[list[str]]:
    """Components of path below root (both absolute), or None if path is not inside root."""
    np_, nr = os.path.normcase(os.path.normpath(path)), os.path.normcase(os.path.normpath(root))
    if np_ == nr:
        return []
    prefix = nr if nr.endswith(os.sep) else nr + os.sep
    if not np_.startswith(prefix):
        return None
    rest = os.path.normpath(path)[len(prefix):]
    return [p for p in rest.split(os.sep) if p]


def _link_target(path: str) -> Optional[str]:
    """The target of a symbolic link or junction at path, or None. Uses lstat only, never follows."""
    try:
        st = os.lstat(path)
    except (OSError, ValueError):
        return None
    is_link = stat.S_ISLNK(st.st_mode)
    tag = getattr(st, "st_reparse_tag", 0)
    if tag and tag in (getattr(stat, "IO_REPARSE_TAG_SYMLINK", -1), getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)):
        is_link = True
    if not is_link:
        return None
    try:
        target = os.readlink(path)
    except (OSError, ValueError) as exc:
        raise _Blocked("unreadable symbolic link") from exc
    target = os.fsdecode(target)
    # Windows may report the NT form of the target.
    for prefix in ("\\\\?\\UNC\\", "\\??\\UNC\\"):
        if target.startswith(prefix):
            raise _Blocked("symbolic link to a network path")
    for prefix in ("\\\\?\\", "\\??\\"):
        if target.startswith(prefix):
            target = target[len(prefix):]
    return target


def _anchor_split(abs_path: str) -> tuple[str, list[str]]:
    drive, rest = os.path.splitdrive(abs_path)
    anchor = drive + os.sep
    return anchor, [p for p in rest.replace("/", os.sep).split(os.sep) if p]


def _walk(root: str, parts: list[str], confine: bool, aliases: tuple[str, ...] = ()) -> str:
    """Resolve parts below root, following links one at a time with lstat/readlink.

    parts must be lexically clean (no '.', '..'). With confine=True every link
    target must stay inside root (or one of its aliases, other spellings of the
    same folder); otherwise only network targets are refused.
    """
    current = root
    pending = list(parts)
    hops = 0
    while pending:
        part = pending.pop(0)
        _check_component(part)
        nxt = os.path.join(current, part)
        target = _link_target(nxt)
        if target is None:
            current = nxt
            continue
        hops += 1
        if hops > MAX_LINK_HOPS:
            raise _Blocked("too many symbolic links")
        kind, s = _classify(target)
        # Resolve the target the way the OS does: lexically against the link's folder.
        full = os.path.normpath(os.path.join(current, s.replace("/", os.sep)) if kind == "relative"
                                else s.replace("/", os.sep))
        if full.replace("\\", "/").startswith("//"):
            raise _Blocked("symbolic link to a network path")
        if confine:
            rel = None
            for r in (root, *aliases):
                rel = _relative_to(full, r)
                if rel is not None:
                    break
            if rel is None:
                raise _Outside("symbolic link leads out of the model folder")
            current, pending = root, rel + pending
        else:
            anchor, rest = _anchor_split(full)
            current, pending = anchor, rest + pending
    return current


def _check_path(path: str, base_dir: str, allow_external: bool) -> str:
    kind, s = _classify(path)
    base_abs = os.path.abspath(base_dir)
    base_real = os.path.realpath(base_abs)  # the model folder itself was chosen by the user
    if kind == "absolute":
        native = s.replace("/", os.sep)
        rel = None
        for root in (base_abs, base_real):
            # Compare without resolving '..' first, so "base/../x" does not count as inside.
            prefix = os.path.normcase(root.rstrip(os.sep) + os.sep)
            if os.path.normcase(native).startswith(prefix):
                rel = native[len(prefix):].replace(os.sep, "/")
                break
        if rel is not None:
            try:
                return _walk(base_real, _lexical_parts(rel), confine=True, aliases=(base_abs,))
            except _Outside:
                if not allow_external:
                    raise
        if not allow_external:
            raise _Outside("absolute path outside the model folder")
        anchor, rest = _anchor_split(os.path.normpath(native))
        return _walk(anchor, rest, confine=False)
    try:
        parts = _lexical_parts(s)
    except _Outside:
        if not allow_external:
            raise
        full = os.path.normpath(os.path.join(base_abs, s.replace("/", os.sep)))
        anchor, rest = _anchor_split(full)
        return _walk(anchor, rest, confine=False)
    try:
        return _walk(base_real, parts, confine=True, aliases=(base_abs,))
    except _Outside:
        if not allow_external:
            raise
        return _walk(base_real, parts, confine=False)
