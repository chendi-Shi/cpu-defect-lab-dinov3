"""Predict a new image with a saved model bank, on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lab import ROOT, Extractor, model_features, score_one, picture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'outputs' / 'prediction.png')
    args = p.parse_args()
    saved = torch.load(args.model, map_location='cpu', weights_only=True)
    cfg = saved['config']
    torch.set_num_threads(cfg['threads'])
    net = Extractor(cfg['size'])
    g, l, _, _ = net.extract([args.image], cfg['size'], 1)
    feature = model_features(g, l, cfg['method'], cfg['dims'], cfg['seed'])[0]
    score, heat = score_one(feature, saved['bank'], cfg['method'], cfg['size'])
    print(json.dumps({'image': str(args.image), 'score': score,
                      'threshold': saved['threshold'], 'anomalous': score > saved['threshold']}, indent=2))
    if heat is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        picture(args.image, np.zeros_like(heat, dtype=np.uint8), heat, args.out, cfg['size'],
                'Image | No ground truth provided | Heatmap')
        print(f'Heatmap: {args.out}')


if __name__ == '__main__':
    main()
