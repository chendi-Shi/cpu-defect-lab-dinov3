"""Trace the frozen first-run local false positive back to bank source patches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from lab import ROOT, model_features, nearest, score_one


def tile(path, size=224):
    with Image.open(path) as im:
        return im.convert('RGB').resize((size, size))


def box_for(index, size=224):
    row, col = divmod(index, 14)
    return (col * size // 14, row * size // 14,
            (col + 1) * size // 14, (row + 1) * size // 14)


def contact_sheet(query_path, query_patch, reference_path, reference_patch, heat, dst):
    query = tile(query_path)
    ref = tile(reference_path)
    query_box = box_for(query_patch)
    ref_box = box_for(reference_patch)
    qcrop = query.crop(query_box).resize((224, 224))
    rcrop = ref.crop(ref_box).resize((224, 224))
    ImageDraw.Draw(query).rectangle(query_box, outline='red', width=3)
    ImageDraw.Draw(ref).rectangle(ref_box, outline='red', width=3)
    v = (heat - heat.min()) / max(float(np.ptp(heat)), 1e-8)
    colors = np.stack([v, np.zeros_like(v), 1 - v], -1)
    overlay = Image.blend(tile(query_path), Image.fromarray((colors * 255).astype(np.uint8)), .45)
    canvas = Image.new('RGB', (224 * 5, 260), 'white')
    for i, (im, title) in enumerate(zip([query, overlay, qcrop, ref, rcrop],
                                      ['Normal test: max patch', 'Anomaly map', 'Query cell enlarged',
                                       'Nearest normal training', 'Reference cell enlarged'])):
        canvas.paste(im, (224 * i, 36))
        ImageDraw.Draw(canvas).text((224 * i + 5, 10), title, fill='black')
    canvas.save(dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.run is None:
        candidates = []
        for path in (ROOT / 'outputs').glob('*/config.json'):
            cfg = json.loads(path.read_text(encoding='utf-8'))
            if (cfg['category'] == 'bottle' and cfg['method'] == 'local' and not cfg['smoke_run']
                    and cfg['dims'] == 64 and cfg['bank_size'] == 2000
                    and cfg.get('threshold_split', 'all') == 'all'
                    and (path.parent / 'summary.json').exists()):
                candidates.append(path.parent)
        if not candidates:
            parser.error('Run the formal bottle local baseline first')
        args.run = max(candidates, key=lambda p: p.name)
    cfg = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    if cfg['method'] != 'local' or cfg['smoke_run']:
        raise ValueError('Use a completed formal local run')
    torch.set_num_threads(cfg['threads'])
    saved = torch.load(args.run / 'model.pt', map_location='cpu', weights_only=True)
    cache = torch.load(ROOT / 'cache' / f"features-{cfg['feature_cache_key']}.pt", weights_only=True)
    features = model_features(cache['global'], cache['local'], 'local', cfg['dims'], cfg['seed'])
    nfit, ncal = len(cfg['fit_files']), len(cfg['calibration_files'])
    flat = features[:nfit].reshape(-1, cfg['dims'])
    selected = torch.randperm(len(flat), generator=torch.Generator().manual_seed(cfg['seed']))[:len(saved['bank'])]
    torch.testing.assert_close(flat[selected], saved['bank'])
    folder = Path(cfg['data']) / cfg['category']
    query_rel = str(Path('test') / 'good' / '006.png')
    index = cfg['test_files'].index(query_rel)
    query_features = features[nfit+ncal+index]
    distances = nearest(query_features, saved['bank'])
    maximum = int(distances.argmax())
    matches = torch.cdist(query_features[maximum:maximum+1], saved['bank'])[0]
    top = torch.topk(matches, 3, largest=False).indices.tolist()
    references = []
    for match in top:
        image_index, patch_index = divmod(int(selected[match]), 196)
        references.append({'training_file': cfg['fit_files'][image_index],
                           'patch_index': patch_index, 'grid_row_col': list(divmod(patch_index, 14)),
                           'distance': float(matches[match])})
    score, heat = score_one(query_features, saved['bank'], 'local', cfg['size'])
    calibration = []
    for i, rel in enumerate(cfg['calibration_files']):
        d = nearest(features[nfit+i], saved['bank'])
        peak = int(d.argmax())
        calibration.append({'file': rel, 'score': float(d.max()), 'grid_row_col': list(divmod(peak, 14))})
    calibration.sort(key=lambda x: x['score'], reverse=True)
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    contact_sheet(folder / query_rel, maximum, folder / references[0]['training_file'],
                  references[0]['patch_index'], heat, output / 'FALSE_POSITIVE_TRACE.png')
    record = {'frozen_run': args.run.name, 'feature_cache_key': cfg['feature_cache_key'],
              'query_file': query_rel, 'score': score, 'threshold': saved['threshold'],
              'score_above_threshold': score - saved['threshold'],
              'max_patch_index': maximum, 'max_patch_grid_row_col': list(divmod(maximum, 14)),
              'max_patch_box_at_224': list(box_for(maximum)), 'nearest_training_patches': references,
              'highest_normal_calibration_samples': calibration[:5],
              'interpretation_note': 'Grid boxes are not full receptive fields. Observation is not proof of cause.'}
    (output / 'error_diagnosis.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
