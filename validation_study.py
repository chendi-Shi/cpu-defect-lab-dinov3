"""Select CPU bank parameters using ONLY held-out training-side synthetic validation."""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lab import ROOT, Extractor, model_features, nearest, validation_partition
from metrics import auroc


def synthesize(path, dst, index):
    with Image.open(path) as im:
        image = im.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    rng = np.random.default_rng(8000 + index)
    angle = float(rng.uniform(0, 2 * math.pi))
    radius = float(rng.uniform(60, 78))
    x, y = 112 + radius * math.cos(angle), 112 + radius * math.sin(angle)
    draw = ImageDraw.Draw(image)
    if index % 2 == 0:
        length = int(rng.integers(12, 25))
        dx, dy = -math.sin(angle) * length / 2, math.cos(angle) * length / 2
        draw.line([(x-dx, y-dy), (x+dx, y+dy)], fill=(235, 235, 235), width=3)
        kind = 'scratch'
    else:
        r = int(rng.integers(3, 6))
        draw.ellipse((x-r, y-r, x+r, y+r), fill=(12, 12, 12))
        kind = 'dark_spot'
    image.save(dst)
    return {'source': str(path), 'synthetic': str(dst), 'kind': kind,
            'centre_at_224': [x, y], 'generator_seed': 8000 + index}


def main():
    torch.set_num_threads(4)
    candidates = []
    for path in (ROOT / 'outputs').glob('*/config.json'):
        candidate = json.loads(path.read_text(encoding='utf-8'))
        if (candidate['category'] == 'bottle' and candidate['method'] == 'local'
                and not candidate['smoke_run'] and candidate['dims'] == 64
                and candidate['bank_size'] == 2000 and candidate['seed'] == 42
                and candidate.get('threshold_split', 'all') == 'all'
                and (path.parent / 'summary.json').exists()):
            candidates.append(path.parent)
    if not candidates:
        raise ValueError('Run run_baselines.ps1 before training-side validation')
    baseline = max(candidates, key=lambda p: p.name)
    cfg = json.loads((baseline / 'config.json').read_text(encoding='utf-8'))
    cache = torch.load(ROOT / 'cache' / f"features-{cfg['feature_cache_key']}.pt", weights_only=True)
    nfit = len(cfg['fit_files'])
    tune, reserved = validation_partition(len(cfg['calibration_files']))
    assert not set(tune) & set(reserved)
    folder = Path(cfg['data']) / cfg['category']
    generated = ROOT / 'data' / 'synthetic_validation'
    generated.mkdir(parents=True, exist_ok=True)
    manifest, paths = [], []
    for index, source_index in enumerate(tune):
        src = folder / cfg['calibration_files'][source_index]
        dst = generated / f'{index:03d}.png'
        manifest.append(synthesize(src, dst, index))
        paths.append(dst)
    (generated / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    sg, sl, _, _ = Extractor(224).extract(paths, 224, 4)
    normal_g = cache['global'][[nfit+i for i in tune]]
    normal_l = cache['local'][[nfit+i for i in tune]]
    query_g, query_l = torch.cat([normal_g, sg]), torch.cat([normal_l, sl])
    labels = [0] * len(tune) + [1] * len(paths)
    rows = []
    for dims in [32, 64, 128]:
        fit = model_features(cache['global'][:nfit], cache['local'][:nfit], 'local', dims, 42)
        flat = fit.reshape(-1, dims)
        order = torch.randperm(len(flat), generator=torch.Generator().manual_seed(42))
        queries = model_features(query_g, query_l, 'local', dims, 42)
        for bank_size in [500, 2000, 5000]:
            bank = flat[order[:bank_size]]
            nearest(queries[0], bank)
            scores, timings = [], []
            for q in queries:
                start = time.perf_counter()
                scores.append(float(nearest(q, bank).max()))
                timings.append((time.perf_counter() - start) * 1000)
            row = {'dims': dims, 'bank_size': bank_size, 'validation_auroc': auroc(labels, scores),
                   'bank_mib': bank.numel() * 4 / 1024**2,
                   'scoring_median_ms': float(np.median(timings)),
                   'normal_images': len(tune), 'synthetic_images': len(paths)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    selected = min(rows, key=lambda r: (-r['validation_auroc'], r['bank_mib'], r['bank_size'], r['dims']))
    report = ROOT / 'reports'
    with (report / 'VALIDATION_SWEEP.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    selection = {'selected': selected, 'selection_rule': 'max synthetic validation AUROC, then min bank bytes, vectors, dimensions',
                 'fit_files': cfg['fit_files'],
                 'tuning_normal_files': [cfg['calibration_files'][i] for i in tune],
                 'reserved_threshold_files': [cfg['calibration_files'][i] for i in reserved],
                 'synthetic_manifest': manifest,
                 'official_test_used_for_selection': False,
                 'note': 'Synthetic validation may not predict real-defect performance. Bottle test previously observed.'}
    (report / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    print('Selected:', json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
