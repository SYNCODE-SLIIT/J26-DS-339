"""Hash-checked PyPI wheels in a sidecar folder; never modify active packages."""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import importlib.metadata
import json
import sys
import urllib.request
import zipfile
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / 'results/transformer_usability/dependencies/fcpe_0_0_4'


def fetch(url, maximum):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Download exceeds declared size limit')
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    site = ROOT / 'site_packages'; site.mkdir(exist_ok=True)
    if (ROOT / 'receipt.json').exists():
        receipt = json.loads((ROOT / 'receipt.json').read_text())
        for path, digest in receipt['file_hashes'].items():
            assert sha(Path(path).read_bytes()) == digest, path
        print('ISOLATED_PACKAGES_ALREADY_VERIFIED', receipt['packages'], flush=True)
        return
    queue = deque([Requirement('torchfcpe==0.0.4')]); packages = {}; inherited = {}
    supported = set(sys_tags()); total_downloaded = 0
    while queue:
        requirement = queue.popleft()
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        name = canonicalize_name(requirement.name)
        if name in packages:
            assert packages[name]['version'] in requirement.specifier
            continue
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is not None:
            if installed not in requirement.specifier:
                raise ValueError(f'Existing {name} incompatible; no active-package upgrade authorized by this helper')
            inherited[name] = installed
            continue
        if len(packages) >= 12:
            raise ValueError('Dependency count exceeds declared limit')
        metadata_path = ROOT / (name + '_metadata.json')
        if metadata_path.exists():
            payload = metadata_path.read_bytes()
        else:
            version = '0.0.4/' if name == 'torchfcpe' else ''
            payload = fetch('https://pypi.org/pypi/' + name + '/' + version + 'json', 5_000_000)
            metadata_path.write_bytes(payload)
        metadata = json.loads(payload); version = metadata['info']['version']
        assert version in requirement.specifier
        wheels = []
        for item in metadata['urls']:
            if not item['filename'].endswith('.whl'):
                continue
            _, _, _, tags = parse_wheel_filename(item['filename'])
            if supported & tags:
                wheels.append(item)
        if not wheels:
            raise ValueError(f'No compatible wheel for {name}; do not build or replace core dependencies')
        item = min(wheels, key=lambda x: (not x['filename'].endswith('none-any.whl'), x['size']))
        assert item['url'].startswith('https://files.pythonhosted.org/')
        if item['size'] > 100_000_000 or total_downloaded + item['size'] > 150_000_000:
            raise ValueError('Wheel size exceeds download budget')
        wheel = ROOT / item['filename']
        data = wheel.read_bytes() if wheel.exists() else fetch(item['url'], item['size'])
        assert len(data) == item['size'] and sha(data) == item['digests']['sha256']
        if not wheel.exists():
            wheel.write_bytes(data)
        total_downloaded += len(data)
        with zipfile.ZipFile(wheel) as archive:
            if sum(i.file_size for i in archive.infolist()) > 300_000_000:
                raise ValueError('Unpacked wheel exceeds size limit')
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                target = (site / member.filename).resolve()
                if path.is_absolute() or '..' in path.parts or not target.is_relative_to(site.resolve()):
                    raise ValueError('Unsafe wheel member path')
            archive.extractall(site)
        packages[name] = dict(version=version, wheel=str(wheel), wheel_sha256=sha(data),
                              url=item['url'], metadata_sha256=sha(payload),
                              license=metadata['info'].get('license'), homepage=metadata['info'].get('home_page'))
        queue.extend(Requirement(r) for r in metadata['info'].get('requires_dist') or [])
        print('ISOLATED_WHEEL_READY', name, version, len(data), flush=True)
    for name, version in inherited.items():
        assert importlib.metadata.version(name) == version
    files = {str(path): sha(path.read_bytes()) for path in site.rglob('*') if path.is_file()}
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__).read_bytes()),
                   packages=packages, inherited_active_packages=inherited, file_hashes=files,
                   python=sys.version, active_environment_modified=False, audio_accessed=False,
                   model_activated=False, research_validation_passed=False)
    (ROOT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('ISOLATED_DEPENDENCY_RECEIPT', list(packages), 'inherited', inherited, flush=True)


if __name__ == '__main__':
    main()
