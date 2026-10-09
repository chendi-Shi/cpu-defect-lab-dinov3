"""Download only the MVTec AD bottle archive; record provenance and hash."""
import argparse
import hashlib
import json
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOTTLE_URL = ('https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/'
              'download/420937370-1629951468/bottle.tar.xz')


def download_mirror(target, category='bottle'):
    repo = 'foersben/mvtec-ad'
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}', timeout=45) as r:
        revision = json.load(r)['sha']
    url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{category}?recursive=true&limit=1000'
    with urllib.request.urlopen(url, timeout=45) as r:
        files = [x for x in json.load(r) if x['type'] == 'file']
        if r.headers.get('Link'):
            raise ValueError('Unexpected pagination; refusing an incomplete dataset')
    print(f'Mirror revision {revision}; {len(files)} files; {sum(x["size"] for x in files)/1024**2:.1f} MiB', flush=True)

    def fetch(item):
        relative = Path(item['path'])
        dst = (target / relative).resolve()
        if target.resolve() not in dst.parents:
            raise ValueError('Invalid mirror path')
        dst.parent.mkdir(parents=True, exist_ok=True)
        source = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{item["path"]}'
        expected = item.get('lfs', {}).get('oid')
        if dst.exists():
            payload = dst.read_bytes()
        else:
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(source, timeout=45) as r:
                        payload = r.read()
                    break
                except Exception:
                    print(f'Retrying {relative}: attempt {attempt+1}/6', flush=True)
                    if attempt == 5:
                        raise
                    time.sleep(min(2**attempt, 8))
            if len(payload) != item['size']:
                raise ValueError(f'Size mismatch: {relative}')
            dst.with_suffix('.partial').write_bytes(payload)
            dst.with_suffix('.partial').replace(dst)
        checksum = hashlib.sha256(payload).hexdigest()
        if len(payload) != item['size'] or (expected and checksum != expected):
            raise ValueError(f'Checksum or size mismatch: {relative}')
        return {'path': item['path'], 'sha256': checksum, 'bytes': len(payload)}

    records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for record in executor.map(fetch, files):
            records.append(record)
            if len(records) % 25 == 0:
                print(f'Downloaded {len(records)}/{len(files)} files', flush=True)
    record = {'dataset': 'MVTec AD', 'category': category, 'mirror': repo,
              'revision': revision, 'files': records, 'license': 'CC BY-NC-SA 4.0',
              'official_page': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
              'hash_note': 'Hashes recorded from mirror; LFS hashes checked when provided. Not independently checked against publisher archive.'}
    record_path = target / ('provenance.json' if category == 'bottle' else f'provenance-{category}.json')
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'{category} dataset ready.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=['bottle', 'screw', 'hazelnut', 'metal_nut'], default='bottle')
    parser.add_argument('--url', help='Optional bottle archive URL; default uses a version-pinned public mirror')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded bottle.tar.xz')
    args = parser.parse_args()
    target = ROOT / 'data' / 'mvtec'
    target.mkdir(parents=True, exist_ok=True)
    if not args.archive and not args.url:
        download_mirror(target, args.category)
        return
    if args.category != 'bottle':
        parser.error('Archive import currently supports bottle only; use mirror for screw')
    archive = args.archive or ROOT / 'data' / 'bottle.tar.xz'
    if not archive.exists():
        partial = archive.with_suffix('.partial')
        req = urllib.request.Request(args.url, headers={'User-Agent': 'cpu-defect-lab/0.1'})
        with urllib.request.urlopen(req, timeout=60) as source, partial.open('wb') as dst:
            total = 0
            while chunk := source.read(1024 * 1024):
                dst.write(chunk)
                total += len(chunk)
                if total % (10 * 1024 * 1024) == 0:
                    print(f'Downloaded {total // (1024 * 1024)} MiB', flush=True)
        partial.rename(archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    # Reject paths outside the dataset root, symlinks, hardlinks and devices.
    with tarfile.open(archive, 'r:xz') as tar:
        members = tar.getmembers()
        for member in members:
            path = (target / member.name).resolve()
            if target.resolve() not in path.parents or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {member.name}')
        tar.extractall(target, members=members)
    if not (target / 'bottle' / 'train' / 'good').is_dir():
        raise ValueError('Archive must contain bottle/train/good')
    record = {'dataset': 'MVTec AD', 'category': 'bottle', 'url': args.url,
              'archive_sha256': checksum, 'archive_bytes': archive.stat().st_size,
              'license': 'CC BY-NC-SA 4.0',
              'official_page': 'https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
              'hash_note': 'Recorded content hash for reproducibility; not independently verified against publisher checksum'}
    (target / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
