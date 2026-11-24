"""Bounded seekable HTTP ZIP/ZIP64 access for a pinned public archive.

Metadata preparation must reserve entries first. This module never selects data,
extracts paths, decodes audio or downloads an archive when range support is absent.
"""
import hashlib
import io
from pathlib import PurePosixPath
import re
import stat
import zipfile


class RangeArchive(io.RawIOBase):
    def __init__(self, url, session=None, budget=128 * 1024 * 1024,
                 max_request=16 * 1024 * 1024):
        super().__init__()
        if not url.startswith('https://') or budget <= 0 or max_request <= 0:
            raise ValueError('Expected HTTPS and positive byte limits')
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.url = url
        self.budget = budget
        self.max_request = max_request
        self.transferred = 0
        self.position = 0
        self.size = None
        self.etag = None
        self.requests = []
        self.cache = []
        # Longest permitted ZIP comment plus footer/ZIP64 locator. zipfile
        # itself interprets ZIP64 64-bit offsets, without a custom ZIP parser.
        self._part(None, 65557)

    def _part(self, start, end):
        suffix = start is None
        limit = end if suffix else end - start + 1
        if limit <= 0 or limit > self.max_request or self.transferred + limit > self.budget:
            raise ValueError('Archive transfer budget or request limit exceeded')
        value = f'bytes=-{end}' if suffix else f'bytes={start}-{end}'
        with self.session.get(self.url, headers={'Range': value, 'Accept-Encoding': 'identity'},
                              stream=True, timeout=(10, 30)) as response:
            if response.status_code != 206:
                raise ValueError('Server did not honor byte ranges; full-download fallback prohibited')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if match is None or response.headers.get('Content-Encoding', 'identity') != 'identity':
                raise ValueError('Missing exact content range or encoded range body')
            actual_start, actual_end, total = map(int, match.groups())
            if total <= 0 or actual_end >= total or actual_start > actual_end:
                raise ValueError('Invalid content range')
            expected_start = max(0, total - end) if suffix else start
            expected_end = total - 1 if suffix else end
            if (actual_start, actual_end) != (expected_start, expected_end):
                raise ValueError('Returned range differs from request')
            if self.size is not None and self.size != total:
                raise ValueError('Archive length changed during transfer')
            etag = response.headers.get('ETag')
            if self.requests and etag != self.etag:
                raise ValueError('Archive identity changed during transfer')
            expected_length = actual_end - actual_start + 1
            if 'Content-Length' in response.headers and int(response.headers['Content-Length']) != expected_length:
                raise ValueError('Content length differs from range')
            # Read at most one byte beyond the declared response; never stream
            # a 14 GB body merely because a server ignored a limit.
            payload = response.raw.read(expected_length + 1)
            self.transferred += len(payload)
            if len(payload) != expected_length:
                raise ValueError('Truncated or oversized range body')
            self.size = total
            self.etag = etag
            self.requests.append(dict(start=actual_start, end=actual_end, bytes=len(payload),
                                      etag=etag, sha256=hashlib.sha256(payload).hexdigest()))
            if suffix:
                self.cache.append((actual_start, payload))
            return payload

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}.get(whence)
        if base is None or base + offset < 0:
            raise ValueError('Invalid archive seek')
        self.position = base + offset
        return self.position

    def read(self, size=-1):
        if self.closed:
            raise ValueError('Archive is closed')
        if size < 0:
            size = self.size - self.position
        size = min(size, max(0, self.size - self.position))
        if size == 0:
            return b''
        if size > self.max_request:
            raise ValueError('Unbounded archive read refused')
        start = self.position
        for cached_start, payload in self.cache:
            if cached_start <= start and start + size <= cached_start + len(payload):
                result = payload[start - cached_start:start - cached_start + size]
                break
        else:
            result = self._part(start, start + size - 1)
        self.position += len(result)
        return result


def checked_entries(archive, max_entries=30000):
    rows = archive.infolist()
    if len(rows) > max_entries:
        raise ValueError('Too many archive entries')
    names = set()
    for row in rows:
        path = PurePosixPath(row.filename)
        if (row.orig_filename != row.filename or '\\' in row.filename or ':' in row.filename or path.is_absolute()
                or '..' in path.parts or row.filename in names):
            raise ValueError('Unsafe or duplicate archive entry')
        if stat.S_ISLNK(row.external_attr >> 16) or row.flag_bits & 1:
            raise ValueError('Symlink/encrypted archive entry refused')
        names.add(row.filename)
    return rows


def read_reserved_member(archive, name, max_decoded=16 * 1024 * 1024,
                         max_compressed=8 * 1024 * 1024):
    # No extract()/extractall(): paths from remote archives are never executed
    # or directly written. zipfile verifies decoded length and CRC at EOF.
    checked_entries(archive)
    entry = archive.getinfo(name)
    if entry.is_dir() or entry.file_size > max_decoded or entry.compress_size > max_compressed:
        raise ValueError('Reserved entry exceeds individual limit')
    data = bytearray()
    with archive.open(entry) as stream:
        while True:
            chunk = stream.read(min(65536, max_decoded + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > max_decoded:
                raise ValueError('Decoded entry exceeds limit')
    if len(data) != entry.file_size:
        raise ValueError('Decoded entry length mismatch')
    return bytes(data)


def open_archive(source):
    archive = zipfile.ZipFile(source, 'r', allowZip64=True)
    checked_entries(archive)
    return archive
