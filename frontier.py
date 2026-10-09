"""DINOSaur (ECCV 2026) core reproduction, adapted to timm and bounded CPU memory.

See FRONTIER_PROTOCOL.md for fixed settings and deviations from the paper benchmark.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
import torchvision
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import v2

from frontier_setup import SHA256 as WEIGHTS_SHA
from frontier_core import build_spatial_bank, spatial_distances, unrestricted_distances, choose_task
from lab import ground_truth, picture
from metrics import auroc, threshold_stats

ROOT = Path(__file__).resolve().parent
CATEGORIES = ['bottle', 'screw', 'hazelnut', 'metal_nut']
DATA_COUNTS = {'bottle':(209,83), 'screw':(320,160), 'hazelnut':(391,110), 'metal_nut':(220,115)}
VARIANTS = ['spatial_kcenter_r3', 'spatial_random_r3', 'unrestricted_kcenter', 'spatial_kcenter_r0']


def transform_image(image):
    # Match official tensor resize order, rather than timm's 256/bicubic default.
    transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True),
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize([.485, .456, .406], [.229, .224, .225])])
    return transform(image.convert('RGB'))


class DinoExtractor:
    def __init__(self):
        weights = ROOT / 'cache/frontier/model.safetensors'
        if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHTS_SHA:
            raise ValueError('Run frontier_setup.py to obtain verified DINOv3 weights.')
        self.net = timm.create_model('vit_small_patch16_dinov3', pretrained=False, img_size=224).eval().cpu()
        self.net.load_state_dict(load_file(str(weights)), strict=True)
        self.net.rope.periods = self.net.rope.periods.to(torch.bfloat16).to(torch.float32)
        self.net.requires_grad_(False)
        if isinstance(self.net.norm, torch.nn.Identity):
            if not isinstance(self.net.fc_norm, torch.nn.LayerNorm):
                raise ValueError('Expected final LayerNorm in timm fc_norm.')
            self.final_token_norm = self.net.fc_norm
        else:
            self.final_token_norm = torch.nn.Identity()
        if self.net.num_prefix_tokens != 5:
            raise ValueError('Expected one CLS and four register tokens.')

    @torch.inference_mode()
    def forward_batch(self, tensor):
        tokens = self.final_token_norm(self.net.forward_features(tensor.float().cpu()))
        if tokens.shape[1:] != (201, 384):
            raise ValueError(f'Unexpected DINOv3 feature shape: {tokens.shape}')
        # Final LayerNorm outputs match official DINOSaur source: no extra L2.
        return tokens[:, 0].contiguous(), tokens[:, 5:].contiguous()

    def extract(self, paths, batch=2):
        identity = {'weights': WEIGHTS_SHA, 'timm': str(timm.__version__),
                    'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                    'features': '224-tensor-bilinear-final-LN-no-L2-rope-bf16-v2',
                    'batch': batch, 'threads': torch.get_num_threads()}
        h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for path in paths:
            h.update(str(path.relative_to(ROOT)).encode())
            h.update(path.read_bytes())
        key = h.hexdigest()[:24]
        cache = ROOT / 'cache/frontier' / f'features-{key}.pt'
        if cache.exists():
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            return saved['cls'], saved['patches'], saved['extraction_mean_ms'], key, True
        cls, patches, durations = [], [], []
        for start in range(0, len(paths), batch):
            before = time.perf_counter()
            values = []
            for path in paths[start:start+batch]:
                with Image.open(path) as image:
                    values.append(transform_image(image))
            c, p = self.forward_batch(torch.stack(values))
            durations.append(time.perf_counter()-before)
            cls.append(c); patches.append(p)
            if start == 0 or (start // batch) % 20 == 0 or start+batch >= len(paths):
                print(f'DINOv3 features: {min(start+batch,len(paths))}/{len(paths)}', flush=True)
        cls, patches = torch.cat(cls), torch.cat(patches)
        mean = sum(durations)*1000/len(paths)
        tmp = cache.with_suffix('.partial')
        torch.save({'cls': cls, 'patches': patches, 'extraction_mean_ms': mean}, tmp)
        tmp.replace(cache)
        return cls, patches, mean, key, False


def prepare_dataset(category, extractor, batch=2, smoke=False):
    folder = ROOT / 'data/mvtec' / category
    train = sorted((folder / 'train/good').glob('*.png'))
    test = sorted((folder / 'test').rglob('*.png'))
    if (len(train),len(test)) != DATA_COUNTS[category]:
        raise ValueError(f'Download complete data for {category} first: expected train/test '
                         f'{DATA_COUNTS[category]}, found {(len(train),len(test))}.')
    order = np.random.default_rng(42).permutation(len(train))
    ncal = max(2, round(len(train)*.2))
    fit = sorted(order[ncal:].tolist())
    calibration = sorted(order[:ncal].tolist())
    if smoke:
        fit = fit[:20]
        calibration = calibration[:5]
        test = [next(p for p in test if p.parent.name == name) for name in sorted({p.parent.name for p in test})]
    paths = [train[i] for i in fit] + [train[i] for i in calibration] + test
    cls, patches, mean, key, hit = extractor.extract(paths, batch)
    masks = np.stack([ground_truth(path, folder, 224) for path in test])
    return {'category':category, 'folder':folder, 'fit_paths':[train[i] for i in fit],
            'calibration_paths':[train[i] for i in calibration], 'test_paths':test,
            'cls':cls, 'patches':patches, 'nfit':len(fit), 'ncal':len(calibration),
            'masks':masks, 'labels':[int(p.parent.name!='good') for p in test],
            'extraction_mean_ms':mean, 'cache_key':key, 'cache_hit':hit, 'smoke_run':smoke}


def patch_scores(features, bank, variant):
    if variant == 'unrestricted_kcenter':
        return unrestricted_distances(features, bank)
    radius = 0 if variant == 'spatial_kcenter_r0' else 3
    return spatial_distances(features, bank, radius=radius, chunk=8)


def score_features(features, bank, variant):
    distances = patch_scores(features, bank, variant)
    heat = F.interpolate(distances.reshape(1,1,14,14), size=(224,224), mode='bilinear',
                         align_corners=False)[0,0].numpy()
    return float(distances.max()), heat


def run_variant(data, seed, variant, bank, indices, build_seconds):
    suffix = 'smoke' if data['smoke_run'] else 'seed'
    output = ROOT / 'outputs/frontier' / data['category'] / f'{suffix}-{seed}' / variant
    output.mkdir(parents=True, exist_ok=True)
    cfg = {'category':data['category'], 'variant':variant, 'seed':seed, 'split_seed':42,
           'coreset_rate':.1, 'radius':0 if variant=='spatial_kcenter_r0' else 3,
           'sampler':'random' if variant=='spatial_random_r3' else 'kcenter',
           'size':224, 'dims':384, 'threads':torch.get_num_threads(), 'threshold_quantile':.95,
           'backbone':'DINOv3 ViT-S/16 (timm CPU adaptation)', 'weights_sha256':WEIGHTS_SHA,
           'timm':str(timm.__version__), 'torch':str(torch.__version__),
           'torchvision':str(torchvision.__version__), 'feature_normalization':'final LayerNorm; no extra L2',
           'rope_periods':'bf16 truncated, float32 runtime',
           'feature_cache_key':data['cache_key'], 'smoke_run':data['smoke_run'],
           'fit_files':[p.relative_to(data['folder']).as_posix() for p in data['fit_paths']],
           'calibration_files':[p.relative_to(data['folder']).as_posix() for p in data['calibration_paths']],
           'test_files':[p.relative_to(data['folder']).as_posix() for p in data['test_paths']],
           'feature_note':'Final LayerNorm features, no extra L2; RoPE periods truncated to bf16 then fp32.'}
    config_path = output / 'config.json'
    artifacts=['summary.json','config.json','predictions.csv','model.pt','test_features.pt']
    if all((output/name).is_file() for name in artifacts) and json.loads(config_path.read_text(encoding='utf-8')) == cfg:
        print(f'Reuse completed: {output.relative_to(ROOT)}', flush=True)
        return output
    # A stale completion marker must not survive a changed-config interrupted run.
    (output/'summary.json').unlink(missing_ok=True)
    config_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    nfit, ncal = data['nfit'], data['ncal']
    features = data['patches']
    calibration = [score_features(features[nfit+i],bank,variant)[0] for i in range(ncal)]
    threshold = float(np.quantile(calibration,.95))
    rows, maps, timings = [], [], []
    counts = {}
    patch_scores(features[nfit+ncal],bank,variant)
    for i,path in enumerate(data['test_paths']):
        start = time.perf_counter()
        score,heat = score_features(features[nfit+ncal+i],bank,variant)
        milliseconds = (time.perf_counter()-start)*1000
        timings.append(milliseconds); maps.append(heat)
        label = data['labels'][i]
        rows.append({'path':path.relative_to(data['folder']).as_posix(),'defect':path.parent.name,
                     'label':label,'score':score,'prediction':int(score>threshold),'scoring_ms':milliseconds})
        if seed==42 and variant=='spatial_kcenter_r3':
            kind=path.parent.name;counts[kind]=counts.get(kind,0)+1
            if counts[kind]<=2 or (label!=int(score>threshold) and sum(x.name.startswith('error-') for x in output.glob('*.png'))<8):
                prefix='error-' if label!=int(score>threshold) else ''
                picture(path,data['masks'][i],heat,output/f'{prefix}{kind}-{path.stem}.png',224,
                        f'DINOv3 | Ground truth | Heatmap  score={score:.3f}, threshold={threshold:.3f}')
        if i==0 or (i+1)%40==0:
            print(f"{data['category']} seed{seed} {variant}: scored {i+1}/{len(data['test_paths'])}",flush=True)
    scores=[row['score'] for row in rows]
    summary={'category':data['category'],'variant':variant,'seed':seed,'image_auroc':auroc(data['labels'],scores),
             'pixel_auroc':auroc(data['masks'],np.stack(maps)),'threshold':threshold,
             'threshold_metrics':threshold_stats(data['labels'],scores,threshold),
             'bank_vectors':bank.numel()//bank.shape[-1], 'bank_mib':bank.numel()*bank.element_size()/1024**2,
             'scoring_median_ms':float(np.median(timings)),'scoring_p95_ms':float(np.quantile(timings,.95)),
             'feature_extraction_mean_ms':data['extraction_mean_ms'],'coreset_seconds':build_seconds,
             'fit_images':nfit,'calibration_images':ncal,'test_images':len(rows),'smoke_run':data['smoke_run'],
             'feature_cache_hit':data['cache_hit'],'timing_note':'Scoring includes map interpolation; excludes feature extraction, image IO, metrics and drawing.'}
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    prototype=data['cls'][:nfit].mean(0)
    torch.save({'bank':bank,'indices':indices,'prototype':prototype,'threshold':threshold,'config':cfg},output/'model.pt')
    # Save per-image grids for audit and cached continual routing, avoiding bulky 224 maps.
    torch.save({'cls':data['cls'][nfit+ncal:],'scores':torch.tensor(scores)},output/'test_features.pt')
    summary_pending=output/'summary.partial'
    summary_pending.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    summary_pending.replace(output/'summary.json')
    print(json.dumps({'category':data['category'],'variant':variant,'seed':seed,'AUROC':summary['image_auroc'],
                      'F1':summary['threshold_metrics']['f1'],'bank_MiB':summary['bank_mib']}),flush=True)
    return output


def run_continual(datasets, models):
    stages=[];prototypes={};bank_hashes={}
    # Identical known-category scoring has already been measured in seed42.
    # Reuse those exact scores; recompute every cross-category routed image.
    known_scores={}
    for category in datasets:
        path=ROOT/'outputs/frontier'/category/'seed-42/spatial_kcenter_r3/predictions.csv'
        with path.open(encoding='utf-8',newline='') as stream:
            known_scores[category]={row['path']:float(row['score']) for row in csv.DictReader(stream)}
    for stage,category in enumerate(CATEGORIES,1):
        if category not in datasets:
            continue
        prototypes[category]=models[category]['prototype']
        bank_hashes[category]=hashlib.sha256(models[category]['bank'].numpy().tobytes()).hexdigest()
        rows=[]
        for known in prototypes:
            data=datasets[known];offset=data['nfit']+data['ncal']
            labels,scores,predictions=[],[],[]
            routing=0
            for i,path in enumerate(data['test_paths']):
                chosen=choose_task(data['cls'][offset+i],prototypes)
                routing+=int(chosen==known)
                if chosen == known:
                    score=known_scores[known][path.relative_to(data['folder']).as_posix()]
                else:
                    score,_=score_features(data['patches'][offset+i],models[chosen]['bank'],'spatial_kcenter_r3')
                labels.append(data['labels'][i]);scores.append(score)
                predictions.append(int(score>models[chosen]['threshold']))
            y=np.asarray(labels,dtype=bool);p=np.asarray(predictions,dtype=bool)
            tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
            rows.append({'category':known,'routing_accuracy':routing/len(labels),'image_auroc':auroc(labels,scores),
                         'tp':tp,'fp':fp,'fn':fn,'tn':tn,'test_images':len(labels)})
        actual_hashes={c:hashlib.sha256(models[c]['bank'].numpy().tobytes()).hexdigest() for c in prototypes}
        stages.append({'stage':stage,'added_category':category,'categories':list(prototypes),'results':rows,
                       'bank_hashes':actual_hashes})
        print(f'Continual stage {stage}: {json.dumps(rows)}',flush=True)
    output={'seed':42,'protocol':'Category addition with frozen backbone and retained category banks; unknown task routed by CLS prototype.',
            'stages':stages,'bank_contents_unchanged':all(
                s['bank_hashes'][c]==bank_hashes[c] for s in stages for c in s['bank_hashes']),
            'note':'Unchanged banks do not guarantee unchanged automatically routed predictions. Routing errors and metric changes are reported.',
            'score_reuse':'Exact seed42 scores reused only when routed to the original category; cross-category routes recomputed.'}
    (ROOT/'outputs/frontier/continual.json').write_text(json.dumps(output,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories',nargs='+',choices=CATEGORIES,default=CATEGORIES)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--continual',action='store_true')
    args=parser.parse_args()
    if args.batch < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.categories)) != len(args.categories):
        parser.error('Batch must be positive, and categories/seeds must not repeat.')
    if args.continual and 42 not in args.seeds:
        parser.error('Continual evaluation requires the fixed seed42 primary models.')
    torch.set_num_threads(1)
    print('Loading verified DINOv3 ViT-S/16 on CPU...',flush=True)
    extractor=DinoExtractor();datasets={};models={}
    for category in args.categories:
        data=prepare_dataset(category,extractor,args.batch,args.smoke)
        if args.continual:
            datasets[category]=data
        for seed in args.seeds:
            for sampler in ['kcenter','random']:
                if seed != 42 and sampler == 'random':
                    continue
                before=time.perf_counter()
                bank,indices=build_spatial_bank(data['patches'][:data['nfit']],rho=.1,seed=seed,sampler=sampler)
                seconds=time.perf_counter()-before
                variants=['spatial_random_r3'] if sampler=='random' else [v for v in VARIANTS if v!='spatial_random_r3']
                if seed != 42:
                    variants=['spatial_kcenter_r3']
                for variant in variants:
                    path=run_variant(data,seed,variant,bank,indices,seconds)
                    if seed==42 and variant=='spatial_kcenter_r3':
                        models[category]=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    if args.continual and not args.smoke:
        run_continual(datasets,models)


if __name__=='__main__':
    main()
