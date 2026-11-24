"""Finite local HTTP/ZIP fixtures, including sparse >4 GB ZIP64 offsets."""
import importlib.util
import io
from pathlib import Path
import re
import struct
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('bounded_remote_zip', Path(__file__).resolve().parents[1] / 'tools/bounded_remote_zip.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def zip_bytes(name='all_wav/F01_0001_0001_1.wav'):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(name, b'example waveform bytes')
    # Windows zipfile normalizes a backslash when writing. Restore the raw
    # remote-name bytes so this fixture tests a foreign archive, not the writer.
    return stream.getvalue().replace(name.replace('\\', '/').encode(), name.encode())


class Response:
    def __init__(self, payload, start, end, size, status=206, etag='"locked"'):
        self.status_code = status
        self.headers = {'Content-Range': f'bytes {start}-{end}/{size}',
                        'Content-Length': str(end - start + 1), 'ETag': etag}
        self.raw = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session:
    def __init__(self, blob=None, segments=None, size=None):
        self.segments = segments if segments is not None else [(0, blob)]
        self.size = size if size is not None else len(blob)
        self.status = 206
        self.etag = '"locked"'
        self.responses = []
        self.wrong_range = False

    def get(self, url, headers, **kwargs):
        suffix = re.fullmatch(r'bytes=-(\d+)', headers['Range'])
        if suffix:
            start, end = max(0, self.size - int(suffix[1])), self.size - 1
        else:
            start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', headers['Range']).groups())
        body = bytearray(end - start + 1)
        for offset, data in self.segments:
            low, high = max(start, offset), min(end + 1, offset + len(data))
            if low < high:
                body[low - start:high - start] = data[low - offset:high - offset]
        response = Response(bytes(body), start, end, self.size, self.status, self.etag)
        if self.wrong_range:
            response.headers['Content-Range'] = f'bytes {start + 1}-{end}/{self.size}'
        self.responses.append(response)
        return response


class BoundedRemoteZipTests(unittest.TestCase):
    def test_only_reserved_member_crc_checked(self):
        source = module.RangeArchive('https://example.test/pinned.zip', session=Session(zip_bytes()))
        with module.open_archive(source) as archive:
            self.assertEqual(module.read_reserved_member(archive, archive.namelist()[0]), b'example waveform bytes')
        self.assertLess(source.transferred, 65557)
        with self.assertRaisesRegex(ValueError, 'individual'):
            with module.open_archive(source) as archive:
                module.read_reserved_member(archive, archive.namelist()[0], max_decoded=1)

    def test_sparse_zip64_central_directory_beyond_four_gib(self):
        original = zip_bytes()
        end = original.rfind(b'PK\x05\x06')
        fields = struct.unpack_from('<4s4H2LH', original, end)
        central_size, old_start = fields[5:7]
        central = original[old_start:old_start + central_size]
        new_start = (1 << 32) + 12345
        zip64 = struct.pack('<4sQ2H2L4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, 1, 1, len(central), new_start)
        locator = struct.pack('<4sLQL', b'PK\x06\x07', 0, new_start + len(central), 1)
        footer = struct.pack('<4s4H2LH', b'PK\x05\x06', 0, 0, 65535, 65535, 0xffffffff, 0xffffffff, 0)
        tail = central + zip64 + locator + footer
        session = Session(segments=[(0, original[:old_start]), (new_start, tail)], size=new_start + len(tail))
        source = module.RangeArchive('https://example.test/pinned.zip', session=session)
        with module.open_archive(source) as archive:
            self.assertEqual(module.read_reserved_member(archive, archive.namelist()[0]), b'example waveform bytes')
        self.assertGreater(source.size, 1 << 32)
        self.assertLess(source.transferred, 70000)

    def test_range_ignored_refused_without_reading_body(self):
        session = Session(zip_bytes())
        session.status = 200
        with self.assertRaisesRegex(ValueError, 'full-download'):
            module.RangeArchive('https://example.test/pinned.zip', session=session)
        self.assertEqual(session.responses[0].raw.tell(), 0)

    def test_wrong_range_and_changed_identity_refused(self):
        session = Session(zip_bytes())
        session.wrong_range = True
        with self.assertRaisesRegex(ValueError, 'differs'):
            module.RangeArchive('https://example.test/pinned.zip', session=session)
        session = Session(zip_bytes())
        source = module.RangeArchive('https://example.test/pinned.zip', session=session)
        session.etag = '"replacement"'
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            source._part(0, 4)

    def test_budget_and_unsafe_paths_refused(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            module.RangeArchive('https://example.test/pinned.zip', session=Session(zip_bytes()), budget=100)
        for name in ('../escape.wav', '/absolute.wav', 'C:/escape.wav', 'dir\\escape.wav'):
            source = module.RangeArchive('https://example.test/pinned.zip', session=Session(zip_bytes(name)))
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                module.open_archive(source)

    def test_corrupt_payload_refused_by_crc(self):
        data = zip_bytes().replace(b'example waveform bytes', b'example waveform byteX')
        source = module.RangeArchive('https://example.test/pinned.zip', session=Session(data))
        with module.open_archive(source) as archive:
            with self.assertRaises(zipfile.BadZipFile):
                module.read_reserved_member(archive, archive.namelist()[0])


if __name__ == '__main__':
    unittest.main()
