"""Training-side development of CPU screw inspection; never select on real test.

Run --stage develop first. The immutable selection.json is required before
--stage evaluate may read the previously exposed MVTec test images.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from frontier import WEIGHTS_SHA, transform_image
from frontier_core import build_spatial_bank
from metrics import auroc, threshold_stats

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'data/mvtec/screw'
OUT = ROOT / 'outputs/screw_refinement'
REPORT = ROOT / 'reports/screw_refinement'
SYNTH = ROOT / 'data/screw_refinement'
KINDS = ['scratch', 'pit', 'texture_shift']
CANDIDATES = ['raw224', 'aligned224', 'raw336', 'multi224_336']
SPLIT_SEED = 20261009
VERSION = 'screw-refinement-v1'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_splits():
    old = json.loads((ROOT / 'outputs/frontier/screw/seed-42/spatial_kcenter_r3/config.json').read_text())
    fit_old = [FOLDER / value for value in old['fit_files']]
    cal = [FOLDER / value for value in old['calibration_files']]
    if len(fit_old) != 256 or len(cal) != 64:
        raise ValueError('Expected original screw split: 256 fit / 64 calibration.')
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(fit_old))
    dev = [fit_old[int(i)] for i in permutation[:64]]
    groups = {'fit': sorted(fit_old[int(i)] for i in permutation[64:]),
              'select': sorted(dev[:48]), 'audit': sorted(dev[48:]), 'calibration': sorted(cal)}
    sets = [set(values) for values in groups.values()]
    if sum(map(len, sets)) != len(set.union(*sets)) or len(set.union(*sets)) != 320:
        raise ValueError('Training partitions must be disjoint and cover all 320 normal images.')
    value = {'version': VERSION, 'seed': SPLIT_SEED,
             'groups': {key: [{'path': relative(path), 'sha256': sha(path)} for path in paths]
                        for key, paths in groups.items()},
             'real_test_status': 'Previously exposed test; exploratory evaluation only.'}
    target = OUT / 'splits.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != value:
        raise ValueError('Frozen split changed. Use a new experiment version, not an overwrite.')
    atomic_json(target, value)
    return groups, value


def synthesize(path, kind, partition):
    """Procedural proxies at original resolution; no test labels or donor data."""
    from screw_geometry import foreground_mask
    seed = int(hashlib.sha256((VERSION + relative(path) + kind).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    array = np.asarray(image).astype(np.float32)
    mask, diagnostics = foreground_mask(image)
    # Prefer interior foreground: prevent a fake defect simply changing background.
    interior = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(
        __import__('PIL.ImageFilter', fromlist=['MinFilter']).MinFilter(5))) > 0
    coordinates = np.argwhere(interior)
    if len(coordinates) < 20:
        raise ValueError(f'Cannot generate foreground proxy for {path.name}')
    y, x = map(int, coordinates[int(rng.integers(len(coordinates)))])
    height, width = mask.shape
    scale = max(height, width) / 1024.0
    defect = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(defect)
    parameters = {'x': x, 'y': y}
    if kind == 'scratch':
        length = int(rng.integers(40, 81) * scale)
        line_width = max(1, int(rng.integers(3, 6) * scale))
        angle = float(rng.uniform(0, math.pi))
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        draw.line((x-dx, y-dy, x+dx, y+dy), fill=255, width=line_width)
        strength = int(rng.integers(25, 61))
        changed = np.clip(array + strength, 0, 255)
        parameters.update(length=length, width=line_width, angle=angle, brightness=strength)
    elif kind == 'pit':
        radius = max(1, int(rng.integers(6, 13) * scale))
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        changed = np.clip(array - 40, 0, 255)
        parameters.update(radius=radius, brightness=-40)
    elif kind == 'texture_shift':
        half = max(2, int(24 * scale))
        shift = max(1, int(rng.integers(8, 17) * scale))
        draw.rectangle((x-half, y-half, x+half, y+half), fill=255)
        # Edge padding avoids wrapping the opposite image edge into the proxy.
        yy = np.clip(np.arange(height) - shift, 0, height-1)
        changed = array[yy]
        parameters.update(window=2*half+1, shift_y=shift)
    else:
        raise ValueError(kind)
    actual_mask = (np.asarray(defect) > 0) & mask
    if actual_mask.sum() < 3:
        raise ValueError('Empty synthetic defect after foreground clipping.')
    output = np.where(actual_mask[..., None], changed, array).astype(np.uint8)
    destination = SYNTH / partition / kind / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(destination)
    mask_path = destination.with_name(destination.stem + '-mask.png')
    Image.fromarray((actual_mask * 255).astype(np.uint8)).save(mask_path)
    return {'source': relative(path), 'path': relative(destination), 'mask': relative(mask_path),
            'kind': kind, 'partition': partition, 'seed': seed, 'parameters': parameters,
            'sha256': sha(destination), 'mask_sha256': sha(mask_path),
            'defect_mask_pixels': int(actual_mask.sum()),
            'changed_pixels': int(np.any(output != array.astype(np.uint8),axis=2).sum()),
            'foreground_diagnostics': diagnostics,
            'label': 1, 'status': 'Synthetic proxy, not a real industrial defect.'}


def prepare(groups):
    target=OUT/'synthetic_manifest.json'
    expected={(relative(path),kind,partition) for partition in ('select','audit')
              for path in groups[partition] for kind in KINDS}
    if target.exists():
        records=json.loads(target.read_text(encoding='utf-8'))
        if {(r['source'],r['kind'],r['partition']) for r in records}==expected and all(
                sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['mask'])==r['mask_sha256'] for r in records):
            # Upgrade an earlier preparation-only manifest's descriptive field.
            for record in records:
                if 'defect_mask_pixels' not in record:
                    record['defect_mask_pixels']=record['changed_pixels']
                    with Image.open(ROOT/record['source']) as image:
                        source=np.asarray(image.convert('RGB'))
                    with Image.open(ROOT/record['path']) as image:
                        changed=np.asarray(image.convert('RGB'))
                    record['changed_pixels']=int(np.any(source!=changed,axis=2).sum())
            atomic_json(target,records)
            print('Reuse verified original-resolution synthetic proxies.',flush=True)
            return records
    records=[]
    for partition in ('select','audit'):
        for path in groups[partition]:
            for kind in KINDS:
                records.append(synthesize(path,kind,partition))
            if len(records)%24==0:
                print(f'Synthetic preparation: {len(records)}/192',flush=True)
    atomic_json(OUT / 'synthetic_manifest.json', records)
    return records


def make_tensor(image, size):
    from torchvision.transforms import v2
    return v2.Compose([v2.ToImage(), v2.Resize((size, size), antialias=True),
                       v2.ToDtype(torch.float32, scale=True),
                       v2.Normalize([.485, .456, .406], [.229, .224, .225])])(image.convert('RGB'))


def geometry_for(paths):
    from screw_geometry import alignment_transform, GEOMETRY_VERSION
    result = {}
    cache_dir = ROOT/'cache/screw_geometry'
    cache_dir.mkdir(parents=True,exist_ok=True)
    for path in paths:
        key = hashlib.sha256((GEOMETRY_VERSION+sha(path)+sha(ROOT/'screw_geometry.py')).encode()).hexdigest()
        cached = cache_dir/f'{key}.json'
        if cached.exists():
            metadata = json.loads(cached.read_text(encoding='utf-8'))
        else:
            with Image.open(path) as image:
                _, metadata = alignment_transform(image.convert('RGB'))
            atomic_json(cached,metadata)
        result[relative(path)] = metadata
    return result


def load_features(paths, spec, geometry=None):
    from refine_features import DinoExtractor
    from screw_geometry import alignment_transform, GEOMETRY_VERSION
    size = 336 if spec == 'raw336' else 224
    extractor = DinoExtractor(size=size, threads=1)
    def preprocess(image):
        return make_tensor(image,size)
    identity = {'pipeline': VERSION, 'view': spec, 'geometry': GEOMETRY_VERSION,
                'geometry_source_sha256':sha(ROOT/'screw_geometry.py'),
                'runtime_source_sha256':sha(ROOT/'refine_features.py'),
                'preprocessing_source_sha256':sha(Path(__file__))}
    if spec == 'aligned224':
        # Extract calls the preprocessor once per path, in input order, and
        # never calls it on a cache hit. Reuse already checked affine metadata.
        path_iterator = iter(paths)
        def preprocess(image):
            path = next(path_iterator)
            if geometry is None:
                aligned, _ = alignment_transform(image)
            else:
                metadata = geometry[relative(path)]
                if list(image.size) != metadata['source_size']:
                    raise ValueError('Cached geometry dimensions differ from source image.')
                aligned = image.convert('RGB') if metadata['fallback'] else image.convert('RGB').transform(
                    image.size,Image.Transform.AFFINE,metadata['inverse_affine'],
                    resample=Image.Resampling.BILINEAR,fillcolor=tuple(metadata['fill_color']))
            return transform_image(aligned)
    cls, patches, milliseconds, key, hit = extractor.extract(
        paths, batch=2, preprocess=preprocess, identity=identity)
    del extractor
    return {'cls': cls, 'patches': patches, 'mean_ms': milliseconds,
            'cache_key': key, 'cache_hit': hit,
            'index': {relative(path): i for i, path in enumerate(paths)}}


def nearest(patches, bank, radius):
    from refine_features import spatial_distances_gemm
    return spatial_distances_gemm(patches, bank, radius=radius, memory_mib=32)


def single_score(patches, bank, radius):
    values = nearest(patches, bank, radius)
    side = bank.shape[0]
    heat = F.interpolate(values.reshape(1, 1, side, side), size=(224,224),
                         mode='bilinear', align_corners=False)[0,0].numpy()
    return float(values.max()), heat


def get_single(path, spec, features, banks):
    data = features[spec]
    # Reuse frozen-view distances across normalization, singles and fusion.
    cache = data.setdefault('score_cache', {})
    key = (relative(path), id(banks[spec]))
    if key not in cache:
        cache[key] = single_score(data['patches'][data['index'][relative(path)]],
                                  banks[spec], 4 if spec == 'raw336' else 3)
    return cache[key]


def candidate_score(path, candidate, features, banks, medians, geometry):
    from screw_geometry import inverse_heatmap
    if candidate == 'multi224_336':
        _, low = get_single(path, 'raw224', features, banks)
        _, high = get_single(path, 'raw336', features, banks)
        heat = .5 * (low / medians['raw224'] + high / medians['raw336'])
        return float(heat.max()), heat, 'multi224_336'
    spec = candidate
    metadata = geometry.get(relative(path), {})
    if spec == 'aligned224' and metadata.get('fallback', False):
        spec = 'raw224'
    score, heat = get_single(path, spec, features, banks)
    score, heat = score / medians[spec], heat / medians[spec]
    if spec == 'aligned224':
        heat = inverse_heatmap(heat, metadata)
    return score, heat, spec


def summarize(rows, threshold=None):
    value = {'n': len(rows), 'image_auroc': auroc([r['label'] for r in rows], [r['score'] for r in rows])}
    if threshold is not None:
        value.update(threshold=threshold,
                     threshold_metrics=threshold_stats([r['label'] for r in rows], [r['score'] for r in rows], threshold))
    return value


def proxy_metrics(rows):
    result = {kind: auroc([r['label'] for r in rows if r['kind'] in ('good', kind)],
                         [r['score'] for r in rows if r['kind'] in ('good', kind)]) for kind in KINDS}
    return {'by_kind_auroc': result, 'macro_auroc': float(np.mean(list(result.values()))),
            'pooled_auroc': summarize(rows)['image_auroc']}


def choose_candidate(metrics, min_gain=.005, tie_tol=1e-9):
    baseline = metrics['raw224']['macro_auroc']
    available=[c for c in CANDIDATES if c in metrics and metrics[c].get('macro_auroc') is not None]
    highest = max(metrics[c]['macro_auroc'] for c in available)
    best = next(c for c in available if highest-metrics[c]['macro_auroc'] <= tie_tol)
    selected = best if metrics[best]['macro_auroc'] >= baseline+min_gain else 'raw224'
    return selected, best


def validate_development():
    complete=json.loads((OUT/'development_complete.json').read_text(encoding='utf-8'))
    for name, expected in complete['artifacts'].items():
        path=ROOT/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError(f'Frozen development artifact changed: {name}')
    protocol=json.loads((OUT/'protocol.json').read_text())
    for key,name in [('proxy_generator_source_sha256','screw_refine.py'),
                     ('geometry_source_sha256','screw_geometry.py'),
                     ('runtime_source_sha256','refine_features.py')]:
        if sha(ROOT/name)!=protocol[key]:
            raise ValueError(f'Frozen implementation changed: {name}')
    return complete


def resource_probe(groups):
    from refine_features import DinoExtractor
    records = []
    for size in (224, 336):
        model = DinoExtractor(size=size, threads=1)
        times = []
        for path in groups['fit'][:8]:
            with Image.open(path) as image:
                tensor = make_tensor(image, size).unsqueeze(0)
            start = time.perf_counter()
            _, patches = model.forward_batch(tensor)
            times.append((time.perf_counter()-start)*1000)
        records.append({'size': size, 'normal_images': [relative(p) for p in groups['fit'][:8]],
                        'forward_ms': times, 'median_ms': float(np.median(times)),
                        'patch_shape': list(patches.shape), 'threads': 1})
        del model
    atomic_json(OUT / 'resource_probe.json', records)
    if records[1]['median_ms'] > 20000:
        raise RuntimeError('336 exceeds fixed CPU feasibility cap (20s median forward); probe saved.')
    return records


def develop():
    torch.set_num_threads(1)
    if (OUT/'development_complete.json').exists():
        validate_development()
        print('Reuse verified completed development; selection unchanged.',flush=True)
        return
    groups, split = freeze_splits()
    protocol = {'version': VERSION, 'split_seed': SPLIT_SEED, 'coreset_seed': 42,
                'candidates': CANDIDATES, 'fit': 192, 'select_sources': 48, 'audit_sources': 16,
                'calibration': 64, 'synthetic_kinds': KINDS, 'resolution': [224,336],
                'radius': {'raw224': 3, 'aligned224': 3, 'raw336': 4},
                'retrieval':'Float64 masked GEMM candidate selection; direct float32 final distance.',
                'resource_cap_ms':20000,
                'selection': 'Macro AUROC of 3 proxy kinds; minimum gain .005; fixed candidate tie order.',
                'normalization': 'Per-view image-score median from 48 selection normal images only.',
                'fusion': 'Mean normalized 224-coordinate maps; image score is fused map maximum.',
                'threshold': '95th percentile of 64 original normal calibration images; after selection.',
                'test': 'Previously exposed MVTec test; only baseline and preselected winner evaluated.',
                'weights_sha256': WEIGHTS_SHA, 'proxy_generator_source_sha256': sha(Path(__file__)),
                'geometry_source_sha256': sha(ROOT/'screw_geometry.py'),
                'runtime_source_sha256': sha(ROOT/'refine_features.py')}
    target = OUT / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Frozen implementation/protocol differs. Do not reuse old selections.')
    probe = resource_probe(groups)
    atomic_json(target, protocol)
    records = prepare(groups)
    normal_paths = groups['fit'] + groups['select'] + groups['audit'] + groups['calibration']
    select_proxy = [ROOT/r['path'] for r in records if r['partition']=='select']
    paths = normal_paths + select_proxy
    geometry = geometry_for(paths)
    atomic_json(OUT/'geometry_development.json', geometry)
    features, banks, bank_indices,fit_sources,disabled = {}, {}, {},{},{}
    for spec in ('raw224','raw336','aligned224'):
        features[spec] = load_features(paths, spec,geometry)
        fit_sources[spec]=[relative(p) for p in groups['fit'] if spec!='aligned224' or not geometry[relative(p)]['fallback']]
        indexes = [features[spec]['index'][relative(p)] for p in groups['fit']
                   if spec != 'aligned224' or not geometry[relative(p)]['fallback']]
        if len(indexes) < 20:
            disabled[spec]='Fewer than 20 reliable aligned fit images.'
            continue
        before = time.perf_counter()
        banks[spec], bank_indices[spec] = build_spatial_bank(features[spec]['patches'][indexes], seed=42)
        banks[spec]=banks[spec].clone()
        print(f'{spec}: features cached; bank {tuple(banks[spec].shape)}, fit={len(indexes)}, build={time.perf_counter()-before:.2f}s', flush=True)
    medians = {}
    normal_scores = {}
    for spec in ('raw224','raw336','aligned224'):
        if spec in disabled:
            continue
        rows = []
        for path in groups['select']:
            if spec == 'aligned224' and geometry[relative(path)]['fallback']:
                continue
            rows.append(get_single(path,spec,features,banks)[0])
        if not rows:
            disabled[spec]='No reliable aligned selection normal images.'
            banks.pop(spec,None)
            continue
        medians[spec] = float(np.median(rows))
        if not medians[spec] > 0:
            raise ValueError('Non-positive development normal scale.')
        normal_scores[spec] = rows
    atomic_json(OUT/'normalization.json', {'medians':medians,'normal_scores':normal_scores})
    selection_metrics = {}
    # Compute each single view once and reuse the exact values for fusion.
    cached_scores = {}
    all_records = [{'path': relative(p),'kind':'good','label':0,'source':relative(p)} for p in groups['select']] + [r for r in records if r['partition']=='select']
    for candidate in CANDIDATES:
        if candidate in disabled:
            selection_metrics[candidate]={'macro_auroc':None,'disabled':disabled[candidate]}
            atomic_json(OUT/candidate/'disabled.json',selection_metrics[candidate])
            continue
        csv_path=OUT/candidate/'development_predictions.csv'
        marker=OUT/candidate/'development_complete.json'
        if marker.exists():
            old=json.loads(marker.read_text())
            fingerprint={'protocol_sha256':sha(target),'split_sha256':sha(OUT/'splits.json'),
                         'normalization_sha256':sha(OUT/'normalization.json'),
                         'feature_keys':{k:v['cache_key'] for k,v in features.items()}}
            if all(old.get(k)==v for k,v in fingerprint.items()) and old.get('csv_sha256')==sha(csv_path):
                with csv_path.open(encoding='utf-8',newline='') as stream:
                    rows=list(csv.DictReader(stream))
                for row in rows:
                    row['label']=int(row['label']);row['score']=float(row['score'])
                selection_metrics[candidate]=proxy_metrics(rows)
                print(f'Reuse verified development scores: {candidate}',flush=True)
                continue
        rows = []
        for i, record in enumerate(all_records):
            path = ROOT / record['path']
            started = time.perf_counter()
            score, heat, branch = candidate_score(path,candidate,features,banks,medians,geometry)
            rows.append({'path':record['path'],'source':record['source'],'kind':record['kind'],
                         'label':record['label'],'score':score,'branch':branch,
                         'scoring_ms':(time.perf_counter()-started)*1000})
            if i % 40 == 0:
                print(f'development {candidate}: {i+1}/{len(all_records)}',flush=True)
        write_rows(csv_path,rows)
        selection_metrics[candidate] = proxy_metrics(rows)
        atomic_json(marker,{'protocol_sha256':sha(target),'csv_sha256':sha(csv_path),
                    'split_sha256':sha(OUT/'splits.json'),
                    'normalization_sha256':sha(OUT/'normalization.json'),
                    'feature_keys':{k:v['cache_key'] for k,v in features.items()}})
    baseline = selection_metrics['raw224']['macro_auroc']
    selected, best = choose_candidate(selection_metrics)
    selection = {'selected':selected, 'best_proxy_candidate':best, 'metrics':selection_metrics,
                 'minimum_gain':.005, 'baseline_macro_auroc':baseline,
                 'test_used_for_selection':False, 'audit_used_for_selection':False,
                 'protocol_sha256':sha(target),'split_sha256':sha(OUT/'splits.json'),
                 'normalization_sha256':sha(OUT/'normalization.json'),
                 'development_csv_sha256':{c:sha(OUT/c/'development_predictions.csv') for c in CANDIDATES}}
    selected_path=OUT/'selection.json'
    if selected_path.exists() and json.loads(selected_path.read_text())!=selection:
        raise ValueError('Frozen candidate selection cannot be overwritten.')
    atomic_json(selected_path,selection)
    # The selection is now immutable; audit/calibration happen afterwards.
    audit_records = [{'path':relative(p),'kind':'good','label':0,'source':relative(p)} for p in groups['audit']] + [r for r in records if r['partition']=='audit']
    audit_paths = [ROOT/r['path'] for r in audit_records]
    audit_geometry = geometry_for(audit_paths)
    needed={'raw224'}
    if selected in ('raw336','multi224_336'):
        needed.add('raw336')
    if selected=='aligned224':
        needed.add('aligned224')
    audit_features = {spec:load_features(audit_paths,spec,audit_geometry) for spec in sorted(needed)}
    audit_results, calibration = {}, {}
    for candidate in sorted(set(['raw224',selected])):
        rows = []
        for record in audit_records:
            score, _, branch = candidate_score(ROOT/record['path'],candidate,audit_features,banks,medians,audit_geometry)
            rows.append({'path':record['path'],'source':record['source'],'kind':record['kind'],
                         'label':record['label'],'score':score,'branch':branch})
        write_rows(OUT/candidate/'audit_predictions.csv',rows)
        audit_results[candidate] = proxy_metrics(rows)
        cal_rows = []
        for path in groups['calibration']:
            score, _, branch = candidate_score(path,candidate,features,banks,medians,geometry)
            cal_rows.append({'path':relative(path),'score':score,'branch':branch})
        threshold = float(np.quantile([r['score'] for r in cal_rows],.95))
        calibration[candidate] = {'threshold':threshold,'n':64}
        write_rows(OUT/candidate/'calibration_predictions.csv',cal_rows)
        config = dict(protocol,candidate=candidate,selection_sha256=sha(OUT/'selection.json'),
                      medians=medians,threshold=threshold)
        required={'raw224'} if candidate=='raw224' else ({'raw336'} if candidate=='raw336' else
                  ({'raw224','aligned224'} if candidate=='aligned224' else {'raw224','raw336'}))
        model = {'config':config,'banks':{k:banks[k] for k in sorted(required)},'medians':medians,'threshold':threshold,
                 'candidate':candidate,'selection':selection,'fit_bank_indices':bank_indices,
                 'fit_sources':fit_sources}
        (OUT/candidate).mkdir(parents=True,exist_ok=True)
        torch.save(model,OUT/candidate/'model.pt')
    artifact_paths=[OUT/'protocol.json',OUT/'splits.json',OUT/'normalization.json',
                    OUT/'selection.json',OUT/'synthetic_manifest.json',OUT/'geometry_development.json']
    for candidate in CANDIDATES:
        artifact_paths += ([OUT/candidate/'disabled.json'] if candidate in disabled else
                           [OUT/candidate/'development_predictions.csv',OUT/candidate/'development_complete.json'])
    for candidate in sorted(set(['raw224',selected])):
        artifact_paths += [OUT/candidate/name for name in ('model.pt','audit_predictions.csv','calibration_predictions.csv')]
    atomic_json(OUT/'development_complete.json',{'selection':selection,'audit':audit_results,
                'calibration':calibration,'resource_probe':probe,
                'cache_keys':{k:v['cache_key'] for k,v in features.items()},
                'artifacts':{relative(path):sha(path) for path in artifact_paths},
                'fit_sources':fit_sources,
                'alignment_fallback_counts':{g:sum(geometry[relative(p)]['fallback'] for p in ps) for g,ps in groups.items()}})
    print(f'DEVELOPMENT COMPLETE: selected={selected}; real test not used.',flush=True)


def evaluate():
    torch.set_num_threads(1)
    selection = json.loads((OUT/'selection.json').read_text())
    complete = validate_development()
    if selection != complete['selection']:
        raise ValueError('Development must complete before real-test evaluation.')
    if sha(OUT/'protocol.json') != selection['protocol_sha256']:
        raise ValueError('Frozen selection protocol changed.')
    paths = sorted((FOLDER/'test').rglob('*.png'))
    if len(paths) != 160:
        raise ValueError('Expected 160 screw test images.')
    candidates = list(dict.fromkeys(['raw224',selection['selected']]))
    specs = {'raw224'}
    if selection['selected'] in ('raw336','multi224_336'):
        specs.add('raw336')
    elif selection['selected']=='aligned224':
        specs.add('aligned224')
    geometry = geometry_for(paths) if 'aligned224' in specs else {}
    features = {spec:load_features(paths,spec,geometry) for spec in sorted(specs)}
    from lab import ground_truth,picture
    from screw_geometry import inverse_heatmap
    masks=np.stack([ground_truth(path,FOLDER,224) for path in paths])
    # An aligned heatmap does not cover every original background pixel after
    # rotation. Exclude that same region from BOTH methods' pixel comparison.
    valid=np.stack([inverse_heatmap(np.ones((224,224),np.float32),geometry[relative(path)])>.999
                    if 'aligned224' in specs and not geometry[relative(path)]['fallback']
                    else np.ones((224,224),bool) for path in paths])
    results = {}
    for candidate in candidates:
        saved = torch.load(OUT/candidate/'model.pt',map_location='cpu',weights_only=True)
        if saved['config']['selection_sha256']!=sha(OUT/'selection.json'):
            raise ValueError('Model differs from frozen selection.')
        rows, examples, maps = [], {},[]
        failure_count=0
        for i,path in enumerate(paths):
            started = time.perf_counter()
            score, heat, branch = candidate_score(path,candidate,features,saved['banks'],saved['medians'],geometry)
            label = int(path.parent.name!='good')
            prediction = int(score>saved['threshold'])
            maps.append(heat)
            rows.append({'path':path.relative_to(FOLDER).as_posix(),'kind':path.parent.name,
                         'label':label,'score':score,'prediction':prediction,'branch':branch,
                         'scoring_ms':(time.perf_counter()-started)*1000})
            if path.parent.name not in examples:
                examples[path.parent.name]=(path,heat,score)
            if label!=prediction and failure_count<8:
                picture(path,ground_truth(path,FOLDER,224),heat,
                        OUT/candidate/f'error-{path.parent.name}-{path.stem}.png',224,
                        f'{candidate} | GT | Heatmap  score={score:.3f}, threshold={saved["threshold"]:.3f}')
                failure_count+=1
            if i%40==0:
                print(f'exploratory real test {candidate}: {i+1}/160',flush=True)
        write_rows(OUT/candidate/'test_predictions.csv',rows)
        result=summarize(rows,saved['threshold'])
        result['pixel_auroc_common_valid_224']=auroc(masks[valid],np.stack(maps)[valid])
        result['pixel_valid_fraction']=float(valid.mean())
        result['pixel_metric_note']='Original 224 frame; same evaluable pixels for baseline and selected candidate; not official original-resolution metric.'
        result['by_kind']={kind:threshold_stats([r['label'] for r in rows if r['kind']==kind],
                                                [r['score'] for r in rows if r['kind']==kind],saved['threshold'])
                           for kind in sorted(set(r['kind'] for r in rows))}
        result['bank_mib']={key:bank.numel()*bank.element_size()/1024**2 for key,bank in saved['banks'].items()}
        result['alignment_fallback_count']=sum(r['branch']=='raw224' for r in rows) if candidate=='aligned224' else 0
        result['scoring_median_ms']=float(np.median([r['scoring_ms'] for r in rows]))
        results[candidate]=result
        for kind,(path,heat,score) in examples.items():
            picture(path,ground_truth(path,FOLDER,224),heat,OUT/candidate/f'{kind}-{path.stem}.png',224,
                    f'{candidate} | GT | Heatmap  score={score:.3f}, threshold={saved["threshold"]:.3f}')
        atomic_json(OUT/candidate/'test_summary.json',result)
    old=json.loads((ROOT/'outputs/frontier/screw/seed-42/spatial_kcenter_r3/summary.json').read_text())
    atomic_json(OUT/'evaluation.json',{'selected':selection['selected'],'results':results,
                'historical_256_fit':{'image_auroc':old['image_auroc'],'threshold_metrics':old['threshold_metrics']},
                'test_used_for_selection':False,'test_status':'Previously exposed; exploratory retest.'})
    print(json.dumps({'EVALUATION COMPLETE':results},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['develop','evaluate','all'],default='all')
    args=parser.parse_args()
    if args.stage in ('develop','all'):
        develop()
    if args.stage in ('evaluate','all'):
        evaluate()


if __name__=='__main__':
    main()
