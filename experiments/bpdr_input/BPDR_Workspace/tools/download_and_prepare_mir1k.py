"""Download and prepare the MIR-1K Karaoke Singing Evaluation Dataset.

Source: Figshare DOI: 10.6084/m9.figshare.5802891 (797 MB RAR archive).
Structure:
- Stereo 16 kHz WAV clips (Left channel = Accompaniment / Original, Right channel = Amateur Karaoke Singing)
- PitchLabel (*.pv): 20 ms pitch annotations in semitones (0 = unvoiced)

This script:
1. Downloads the archive to datasets/mir_1k/MIR-1K.rar with resume support and progress logging.
2. Extracts it using bsdtar (built-in on Windows).
3. Indexes and verifies clips, generating datasets/mir_1k/manifest.json.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

WS = Path(__file__).resolve().parents[1]
MIR_DIR = WS / 'datasets' / 'mir_1k'
ARCHIVE_PATH = MIR_DIR / 'MIR-1K.rar'
FIGSHARE_URL = 'https://ndownloader.figshare.com/files/10256751'


def download_file(url, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix('.tmp')

    existing_size = temp_path.stat().st_size if temp_path.exists() else 0
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    if existing_size > 0:
        req.add_header('Range', f'bytes={existing_size}-')

    print(f"Downloading MIR-1K from {url}...")
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get('Content-Length', 0)) + existing_size
            mode = 'ab' if existing_size > 0 else 'wb'
            downloaded = existing_size

            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / total_size * 100.0) if total_size else 0.0
                    mb_done = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    elapsed = time.time() - start_time
                    speed = (downloaded - existing_size) / (1024 * 1024 * max(elapsed, 0.1))
                    if int(mb_done) % 25 == 0:
                        print(f"Downloaded {mb_done:.1f}/{mb_tot:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s", flush=True)

        if temp_path.exists():
            temp_path.rename(dest_path)
            print(f"Download complete: {dest_path} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
            return True
    except Exception as exc:
        print(f"Download error: {exc}", flush=True)
        return False


def extract_archive(archive_path, extract_dir):
    print(f"Extracting {archive_path} to {extract_dir} via tar...")
    cmd = ['tar', '-xf', str(archive_path), '-C', str(extract_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print("Archive extracted successfully!", flush=True)
        return True
    else:
        print(f"tar extraction stderr: {res.stderr}", flush=True)
        return False


def index_and_split_mir1k(extract_dir):
    # Search for Wavfile and PitchLabel
    wav_dirs = list(extract_dir.glob('**/Wavfile'))
    pitch_dirs = list(extract_dir.glob('**/PitchLabel'))
    
    if not wav_dirs:
        print("Wavfile directory not found after extraction.", flush=True)
        return None

    wav_dir = wav_dirs[0]
    pitch_dir = pitch_dirs[0] if pitch_dirs else None
    
    wav_files = sorted(list(wav_dir.glob('*.wav')))
    print(f"Found {len(wav_files)} WAV clips in {wav_dir}.", flush=True)

    manifest = []
    for wf in wav_files:
        stem = wf.stem
        # Extract singer code and clip id (e.g., abjones_1_01 -> singer: abjones)
        parts = stem.split('_')
        singer = parts[0] if parts else 'unknown'
        
        pv_file = (pitch_dir / f"{stem}.pv") if pitch_dir else None
        has_pv = pv_file.exists() if pv_file else False

        manifest.append({
            'clip_id': stem,
            'singer': singer,
            'wav_path': str(wf),
            'has_pitch_label': has_pv,
            'pitch_path': str(pv_file) if has_pv else None,
        })

    manifest_file = MIR_DIR / 'manifest.json'
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Indexed {len(manifest)} clips to {manifest_file}.", flush=True)
    return manifest


def main():
    MIR_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Check if already extracted
    wav_files = list(MIR_DIR.glob('**/Wavfile/*.wav'))
    if len(wav_files) >= 500:
        print(f"MIR-1K already extracted ({len(wav_files)} clips found).")
        index_and_split_mir1k(MIR_DIR)
        return

    # 2. Check if archive exists
    if not ARCHIVE_PATH.exists() or ARCHIVE_PATH.stat().st_size < 500 * 1024 * 1024:
        ok = download_file(FIGSHARE_URL, ARCHIVE_PATH)
        if not ok:
            print("Download failed.")
            return

    # 3. Extract
    ok = extract_archive(ARCHIVE_PATH, MIR_DIR)
    if ok:
        index_and_split_mir1k(MIR_DIR)


if __name__ == '__main__':
    main()
