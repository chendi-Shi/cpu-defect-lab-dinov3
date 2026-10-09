"""CPU-only MVTec experiment. Frozen ResNet18 + nearest-neighbour scoring."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from metrics import auroc, threshold_stats

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache' / 'torch'))


def images(folder):
    return sorted(p for p in folder.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})


def digest(paths, size):
    h = hashlib.sha256(f'resnet18-imagenet-v1-layer2-layer3-layer4-v1-{size}'.encode())
    for p in paths:
        h.update(str(p.resolve()).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:20]


class Extractor:
    def __init__(self, size):
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().cpu()
        self.transform = Compose([Resize((size, size)), ToTensor(),
                                  Normalize([.485, .456, .406], [.229, .224, .225])])

    @torch.inference_mode()
    def forward_batch(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer1(x)
        a = n.layer2(x)
        b = n.layer3(a)
        c = n.layer4(b)
        g = n.avgpool(c).flatten(1)
        local = torch.cat([F.adaptive_avg_pool2d(a, (14, 14)),
                           F.adaptive_avg_pool2d(b, (14, 14))], dim=1)
        return g, local.flatten(2).transpose(1, 2).contiguous()

    @torch.inference_mode()
    def extract(self, paths, size, batch):
        key = digest(paths, size)
        cache = ROOT / 'cache' / f'features-{key}.pt'
        if cache.exists():
            value = torch.load(cache, map_location='cpu', weights_only=True)
            return value['global'], value['local'], True, value['seconds_per_image']
        gs, ls = [], []
        elapsed = 0.0
        for start in range(0, len(paths), batch):
            t = time.perf_counter()
            xs = []
            for p in paths[start:start + batch]:
                with Image.open(p) as im:
                    xs.append(self.transform(im.convert('RGB')))
            x = torch.stack(xs)
            # A fixed 14x14 grid keeps bank/query work bounded on a laptop.
            global_batch, local_batch = self.forward_batch(x)
            gs.append(global_batch)
            ls.append(local_batch)
            elapsed += time.perf_counter() - t
            print(f'features {min(start + batch, len(paths))}/{len(paths)}', flush=True)
        g, l = torch.cat(gs), torch.cat(ls)
        cache.parent.mkdir(parents=True, exist_ok=True)
        per_image = elapsed / len(paths)
        torch.save({'global': g, 'local': l, 'seconds_per_image': per_image}, cache)
        return g, l, False, per_image


def nearest(query, bank, chunk=128):
    """Bound the temporary distance matrix to chunk * bank_size floats."""
    return torch.cat([torch.cdist(x, bank).amin(1) for x in query.split(chunk)])


def validation_partition(count):
    """Disjoint tuning/threshold subsets of held-out normal training images."""
    order = np.random.default_rng(2026).permutation(count)
    return sorted(order[:count // 2].tolist()), sorted(order[count // 2:].tolist())


def model_features(global_features, local_features, method, dims, seed):
    if method == 'global':
        return F.normalize(global_features, dim=-1)
    generator = torch.Generator().manual_seed(seed)
    channels = torch.randperm(local_features.shape[-1], generator=generator)[:dims]
    selected = local_features[..., channels]
    if method == 'global_shared':
        selected = selected.mean(dim=1)
    return F.normalize(selected, dim=-1)


def score_one(features, bank, method, size):
    distances = nearest(features.reshape(-1, features.shape[-1]), bank)
    if method in {'global', 'global_shared'}:
        return float(distances[0]), None
    anomaly = F.interpolate(distances.reshape(1, 1, 14, 14), size=(size, size),
                            mode='bilinear', align_corners=False)[0, 0].numpy()
    return float(distances.max()), anomaly


def ground_truth(path, category_dir, size):
    if path.parent.name == 'good':
        return np.zeros((size, size), dtype=np.uint8)
    mask = category_dir / 'ground_truth' / path.parent.name / (path.stem + '_mask.png')
    if not mask.is_file():
        raise FileNotFoundError(f'Missing ground truth: {mask}')
    with Image.open(mask) as im:
        return (np.asarray(im.convert('L').resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def picture(path, mask, anomaly, dst, size, title):
    with Image.open(path) as im:
        original = im.convert('RGB').resize((size, size))
    canvas = Image.new('RGB', (size * 3, size + 35), 'white')
    canvas.paste(original, (0, 35))
    canvas.paste(Image.fromarray(mask * 255).convert('RGB'), (size, 35))
    # Per-image scaling is for visualization only, never for metrics.
    v = (anomaly - anomaly.min()) / max(float(np.ptp(anomaly)), 1e-8)
    colors = np.stack([v, np.zeros_like(v), 1 - v], -1)
    heat = Image.fromarray((colors * 255).astype(np.uint8))
    canvas.paste(Image.blend(original, heat, .45), (size * 2, 35))
    ImageDraw.Draw(canvas).text((6, 8), title, fill='black')
    canvas.save(dst)


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    folder = args.data.resolve() / args.category
    train = images(folder / 'train' / 'good')
    test = images(folder / 'test')
    if len(train) < 10 or not test:
        raise ValueError(f'Expected MVTec folders under {folder}; need >=10 normal training images and test images')
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(train))
    count = max(2, int(round(len(train) * .2)))
    calibration = sorted(indices[:count].tolist())
    fit = sorted(indices[count:].tolist())
    if args.limit_train:
        fit = fit[:args.limit_train]
    if args.limit_test:
        # Smoke mode keeps a sample of every defect type, including good.
        groups = sorted({p.parent.name for p in test})
        test = [p for group in groups
                for p in [q for q in test if q.parent.name == group][:args.limit_test]]
    selected = [train[i] for i in fit] + [train[i] for i in calibration] + test
    stamp = time.strftime('%Y%m%d-%H%M%S')
    output = ROOT / 'outputs' / f'{args.category}-{stamp}-{args.method}-{args.dims}-{args.bank_size}'
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config['data'] = str(args.data.resolve())
    config.update({'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                   'platform': platform.platform(), 'processor': platform.processor(),
                   'fit_files': [str(train[i].relative_to(folder)) for i in fit],
                   'calibration_files': [str(train[i].relative_to(folder)) for i in calibration],
                   'test_files': [str(p.relative_to(folder)) for p in test],
                   'feature_cache_key': digest(selected, args.size),
                   'smoke_run': bool(args.limit_train or args.limit_test)})
    tuning_ids, threshold_ids = validation_partition(len(calibration))
    if args.threshold_split == 'all':
        threshold_ids = list(range(len(calibration)))
    config['threshold_calibration_files'] = [config['calibration_files'][i] for i in threshold_ids]
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    extractor = Extractor(args.size)
    g, l, hit, extract_seconds = extractor.extract(selected, args.size, args.batch)
    features = model_features(g, l, args.method, args.dims, args.seed)
    nfit, ncal = len(fit), len(calibration)
    bank = features[:nfit].reshape(-1, features.shape[-1])
    if args.method == 'local':
        order = torch.randperm(len(bank), generator=torch.Generator().manual_seed(args.seed))
        bank = bank[order[:min(args.bank_size, len(bank))]]
    # Threshold uses ONLY held-out normal TRAIN images. No test labels are used.
    cal = [score_one(features[nfit+i], bank, args.method, args.size)[0] for i in threshold_ids]
    threshold = float(np.quantile(cal, .95))
    labels, scores, times, rows, maps, masks = [], [], [], [], [], []
    visual_counts = {}
    nearest(features[nfit+ncal].reshape(-1, features.shape[-1]), bank)  # warm-up
    for i, path in enumerate(test):
        t = time.perf_counter()
        score, anomaly = score_one(features[nfit+ncal+i], bank, args.method, args.size)
        duration = time.perf_counter() - t
        mask = ground_truth(path, folder, args.size)
        label = int(path.parent.name != 'good')
        labels.append(label); scores.append(score); times.append(duration)
        rows.append({'path': str(path.relative_to(folder)), 'defect': path.parent.name,
                     'label': label, 'score': score, 'prediction': int(score > threshold),
                     'scoring_ms': duration * 1000})
        if anomaly is not None:
            maps.append(anomaly); masks.append(mask)
            defect = path.parent.name
            visual_counts[defect] = visual_counts.get(defect, 0) + 1
            if visual_counts[defect] <= 3:
                picture(path, mask, anomaly, output / f'{path.parent.name}-{path.stem}.png', args.size,
                        f'Image | Ground truth | Heatmap   score={score:.3f}')
    summary = {'method': args.method, 'category': args.category,
               'image_auroc': auroc(labels, scores),
               'pixel_auroc': auroc(np.stack(masks), np.stack(maps)) if maps else None,
               'threshold': threshold, 'threshold_rule': '95th percentile of held-out normal training scores',
               'threshold_metrics': threshold_stats(labels, scores, threshold),
               'bank_vectors': len(bank), 'feature_dimensions': bank.shape[-1],
               'bank_mib': bank.numel() * bank.element_size() / 1024**2,
               'scoring_median_ms': float(np.median(times) * 1000),
               'scoring_p95_ms': float(np.quantile(times, .95) * 1000),
               'feature_extraction_mean_ms': extract_seconds * 1000,
               'feature_cache_hit': hit, 'fit_images': nfit, 'calibration_images': len(cal),
               'test_images': len(test), 'smoke_run': config['smoke_run'],
               'timing_note': 'Extraction includes decode+resize+forward; scoring excludes decode, visualization and metrics. Cache records original extraction timing.'}
    summary['defect_breakdown'] = {}
    for defect in sorted({r['defect'] for r in rows}):
        subset = [r for r in rows if r['defect'] == defect]
        summary['defect_breakdown'][defect] = {
            'images': len(subset),
            'predicted_anomalous': sum(r['prediction'] for r in subset),
            'mean_score': float(np.mean([r['score'] for r in subset]))}
    if maps:
        failures = [i for i, r in enumerate(rows) if r['label'] != r['prediction']]
        for i in failures[:8]:
            picture(test[i], masks[i], maps[i], output / f'error-{test[i].parent.name}-{test[i].stem}.png',
                    args.size, f'Error: label={labels[i]} score={scores[i]:.3f} threshold={threshold:.3f}')
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    torch.save({'bank': bank, 'threshold': threshold, 'config': config}, output / 'model.pt')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Output: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data' / 'mvtec')
    p.add_argument('--category', default='bottle')
    p.add_argument('--method', choices=['global', 'global_shared', 'local'], default='local')
    p.add_argument('--size', type=int, choices=[128, 224], default=224)
    p.add_argument('--dims', type=int, choices=[32, 64, 128], default=64)
    p.add_argument('--bank-size', type=int, default=2000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold-split', choices=['all', 'reserved'], default='all',
                   help='reserved uses a disjoint held-out normal half for post-selection calibration')
    p.add_argument('--limit-train', type=int, default=0)
    p.add_argument('--limit-test', type=int, default=0, help='Images per defect type; smoke test only')
    args = p.parse_args()
    if min(args.bank_size, args.batch, args.threads) < 1 or min(args.limit_train, args.limit_test) < 0:
        p.error('bank size, batch, threads must be positive; limits must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
