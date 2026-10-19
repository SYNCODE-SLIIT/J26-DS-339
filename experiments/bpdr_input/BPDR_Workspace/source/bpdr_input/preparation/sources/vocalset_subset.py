"""Download only a preselected VocalSet long-tone pilot through strict ZIP ranges.

No full archive download, audio scoring, fitting or intervention creation occurs.
Originals are CC BY 4.0, from the creators' pinned Zenodo 1.0 deposit.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zlib

import requests
import soundfile as sf


RECORD = "https://zenodo.org/api/records/1193957"
PAGE = "https://zenodo.org/records/1193957"
ARCHIVE = "https://zenodo.org/records/1193957/files/VocalSet.zip?download=1"
EXPECTED_ARCHIVE_SIZE = 2077087366
EXPECTED_ARCHIVE_MD5 = "c44f60d34b8724b9a6f6d15e0a3158a9"
MAX_BYTES = 80 * 1024 * 1024
RULE = (
    "Before acoustic analysis: female1..female6 and male1..male6; long_tones/straight; "
    "vowels a and e; lexicographically first exact singer/vowel match if duplicates. "
    "IDs 1..4 train, 5 calibration, 6 pilot_evaluation. All originals retained regardless of later eligibility."
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class BoundedSource:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "BPDR-VocalSet-Bounded-Research-Pilot/1"
        self.session.headers["Accept-Encoding"] = "identity"
        self.bytes = 0
        self.requests = []

    def _read(self, response, limit):
        if self.bytes + limit > MAX_BYTES:
            raise RuntimeError("Transfer budget would exceed 80 MiB")
        data = response.raw.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("Server exceeded the bounded response size")
        self.bytes += len(data)
        return data

    def metadata(self):
        with self.session.get(RECORD, stream=True, timeout=30) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            data = self._read(response, 512 * 1024)
        return json.loads(data), data

    def part(self, start, end):
        if not 0 <= start <= end < EXPECTED_ARCHIVE_SIZE:
            raise ValueError("Invalid archive range")
        expected = f"bytes {start}-{end}/{EXPECTED_ARCHIVE_SIZE}"
        with self.session.get(ARCHIVE, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=30) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError("STOP: archive server did not honor the exact bounded HTTP range")
            data = self._read(response, end - start + 1)
            if len(data) != end - start + 1:
                raise RuntimeError("Truncated archive range")
            self.requests.append(dict(start=start, end=end, bytes=len(data), status=206,
                                      etag=response.headers.get("ETag"),
                                      last_modified=response.headers.get("Last-Modified")))
        return data


def entries(source):
    tail = source.part(EXPECTED_ARCHIVE_SIZE - 65536, EXPECTED_ARCHIVE_SIZE - 1)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise RuntimeError("ZIP directory footer unavailable")
    footer = struct.unpack_from("<4s4H2LH", tail, offset)
    _, disk, central_disk, disk_count, count, size, start, _ = footer
    if disk != 0 or central_disk != 0 or disk_count != count or count > 10000 or size > 2 * 1024 * 1024:
        raise RuntimeError("Unexpected or unbounded ZIP directory")
    directory = source.part(start, start + size - 1)
    result = []
    at = 0
    while at + 46 <= len(directory) and directory[at:at + 4] == b"PK\x01\x02":
        f = struct.unpack_from("<4s6H3L5H2L", directory, at)
        name = directory[at + 46:at + 46 + f[10]].decode("utf-8" if f[3] & 2048 else "cp437")
        result.append(dict(name=name, flags=f[3], compression=f[4], crc32=f[7],
                           compressed_bytes=f[8], decoded_bytes=f[9], local_header=f[16]))
        at += 46 + f[10] + f[11] + f[12]
    if at != len(directory) or len(result) != count:
        raise RuntimeError("ZIP directory count or extent mismatch")
    return result, dict(entry_count=count, directory_bytes=size, directory_sha256=sha256(directory))


def select(rows):
    selected = []
    for gender in ("female", "male"):
        for number in range(1, 7):
            singer = f"{gender}{number}"
            split = "train" if number <= 4 else "calibration" if number == 5 else "pilot_evaluation"
            for vowel in ("a", "e"):
                matches = sorted((row for row in rows
                                  if row["name"].startswith(f"FULL/{singer}/long_tones/straight/")
                                  and row["name"].endswith(f"_{vowel}.wav")), key=lambda row: row["name"])
                if not matches:
                    raise RuntimeError(f"Predeclared singer/vowel is missing: {singer}/{vowel}")
                row = dict(matches[0], singer_id=singer, split=split, vowel=vowel)
                if row["compressed_bytes"] > 4 * 1024 * 1024 or row["decoded_bytes"] > 8 * 1024 * 1024:
                    raise RuntimeError("Preselected file exceeds bounded individual file size")
                selected.append(row)
    if len(selected) != 24:
        raise RuntimeError("Expected exactly 24 preselected files")
    return selected


def member(source, row):
    start = row["local_header"]
    header = source.part(start, start + 29)
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != row["compression"]:
        raise RuntimeError("Invalid, encrypted or mismatched ZIP member header")
    name_length, extra_length = fields[-2:]
    payload_start = start + 30 + name_length + extra_length
    compressed = source.part(payload_start, payload_start + row["compressed_bytes"] - 1)
    if row["compression"] == 8:
        payload = zlib.decompress(compressed, -15)
    elif row["compression"] == 0:
        payload = compressed
    else:
        raise RuntimeError("Unsupported member compression")
    if len(payload) != row["decoded_bytes"] or zlib.crc32(payload) & 0xFFFFFFFF != row["crc32"]:
        raise RuntimeError("ZIP member CRC32 or decoded length mismatch")
    return payload


def prepare(destination):
    destination = Path(destination).resolve()
    component = next(p for p in Path(__file__).resolve().parents
                     if p.name == "bpdr_input" and (p / "pyproject.toml").is_file())
    workspace = (component / "BPDR_Workspace").resolve()
    if not destination.is_relative_to(workspace):
        raise ValueError("Dataset output must remain inside BPDR_Workspace")
    if destination.exists():
        raise FileExistsError("Pilot destination already exists; preserve it and choose a new explicit directory")
    source = BoundedSource()
    metadata, metadata_bytes = source.metadata()
    file = next(item for item in metadata["files"] if item["key"] == "VocalSet.zip")
    if (metadata["metadata"].get("license", {}).get("id") != "cc-by-4.0"
            or file["size"] != EXPECTED_ARCHIVE_SIZE
            or file["checksum"] != "md5:" + EXPECTED_ARCHIVE_MD5):
        raise RuntimeError("Pinned official license/archive identity changed")
    rows, directory_metadata = entries(source)
    selection = select(rows)
    destination.mkdir(parents=True)
    (destination / "official_record.json").write_bytes(metadata_bytes)
    declaration = dict(schema_version="vocalset-bounded-pilot-declaration-1", source_page=PAGE,
                       doi="10.5281/zenodo.1193957", source_archive=ARCHIVE, license="CC-BY-4.0",
                       creators=["Julia Wilkins", "Prem Seetharaman", "Alison Wahl", "Bryan Pardo"],
                       archive_bytes=EXPECTED_ARCHIVE_SIZE, archive_declared_md5=EXPECTED_ARCHIVE_MD5,
                       full_archive_downloaded=False, full_archive_md5_verified=False,
                       selection_rule=RULE, selected=selection, maximum_transfer_bytes=MAX_BYTES,
                       source_metadata_sha256=sha256(metadata_bytes), zip_directory=directory_metadata,
                       scope="Straight long-tone controlled intervention pilot; no natural-error/intention labels",
                       restrictions=["Do not call source straight tones perfectly in tune",
                                     "Known local shifts target recovery toward each original performance",
                                     "Keep singer partitions disjoint before any intervention or model selection",
                                     "Do not use pilot_evaluation singers to fit or tune parameters",
                                     "Do not infer natural singing repair or population accuracy from this pilot"],
                       model_training=False, target_creation=False, status="DECLARED_BEFORE_AUDIO_DOWNLOAD")
    write_json(destination / "declaration.json", declaration)
    receipt = dict(schema_version="vocalset-bounded-pilot-receipt-1", status="IN_PROGRESS", records=[])
    write_json(destination / "manifest.json", receipt)
    try:
        for index, row in enumerate(selection, 1):
            payload = member(source, row)
            info = sf.info(io.BytesIO(payload))
            relative = Path("originals") / row["split"] / row["singer_id"] / Path(row["name"]).name
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            receipt["records"].append(dict(source_entry=row["name"], singer_id=row["singer_id"], split=row["split"],
                                           vowel=row["vowel"], path=str(path), relative_path=relative.as_posix(),
                                           sha256=sha256(payload), archive_crc32=f"{row['crc32']:08x}",
                                           archive_crc32_verified=True, bytes=len(payload),
                                           sample_rate_hz=info.samplerate, sample_count=info.frames,
                                           channels=info.channels, duration_seconds=info.duration,
                                           subtype=info.subtype, intervention_applied=False,
                                           note_reference="None: original performance is a future intervention recovery reference"))
            receipt.update(transferred_bytes=source.bytes, requests=source.requests)
            write_json(destination / "manifest.json", receipt)
            print(f"{index}/24 {row['singer_id']} {row['vowel']}: {info.duration:.3f}s", flush=True)
        receipt.update(status="COMPLETE", recording_count=24, singer_count=12,
                       split_counts={split: sum(r["split"] == split for r in receipt["records"])
                                     for split in ("train", "calibration", "pilot_evaluation")},
                       total_audio_seconds=sum(r["duration_seconds"] for r in receipt["records"]),
                       declaration_sha256=sha256((destination / "declaration.json").read_bytes()))
    except Exception as exc:
        receipt.update(status="STOPPED", error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        receipt.update(transferred_bytes=source.bytes, requests=source.requests)
        write_json(destination / "manifest.json", receipt)
    (destination / "ATTRIBUTION.txt").write_text(
        "VocalSet: A Singing Voice Dataset, version 1.0. Julia Wilkins, Prem Seetharaman, Alison Wahl and Bryan Pardo (2018).\n"
        "https://doi.org/10.5281/zenodo.1193957\nLicense: Creative Commons Attribution 4.0 International.\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "This pilot contains 24 unchanged original WAV files obtained from the official ZIP through HTTP ranges.\n"
        "Future pitch-shifted derivatives must be identified as modified and retain attribution.\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = prepare(args.destination)
    print(json.dumps({k: result[k] for k in ("status", "recording_count", "singer_count", "split_counts",
                                            "transferred_bytes", "total_audio_seconds")}, indent=2))


if __name__ == "__main__":
    main()
