"""Verify the complete DINOSaur release and real HTTP inference against offline scores."""
import csv
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import torch
from PIL import Image

from demo_server import Engine, handler_for
from frontier import ROOT, CATEGORIES, VARIANTS
from frontier_core import choose_task
from frontier_setup import SHA256


def main():
    unit = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(unit.stdout + unit.stderr, flush=True)
    if unit.returncode:
        raise RuntimeError('Unit checks failed')
    test_count = int(re.search(r'Ran (\d+) tests', unit.stdout + unit.stderr).group(1))
    manifest = json.loads((ROOT/'release/frontier/manifest.json').read_text(encoding='utf-8'))
    assert set(manifest['models']) == set(CATEGORIES)
    assert hashlib.sha256((ROOT/'cache/frontier/model.safetensors').read_bytes()).hexdigest() == SHA256
    # Independent pairwise oracle, rather than the experiment's rank-based metric.
    auc_audit=[]
    for category in CATEGORIES:
        for seed in [42,43,44]:
            variants=VARIANTS if seed==42 else ['spatial_kcenter_r3']
            for variant in variants:
                run=ROOT/'outputs/frontier'/category/f'seed-{seed}'/variant
                summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
                with (run/'predictions.csv').open(encoding='utf-8',newline='') as stream:
                    rows=list(csv.DictReader(stream))
                positive=[float(row['score']) for row in rows if int(row['label'])==1]
                negative=[float(row['score']) for row in rows if int(row['label'])==0]
                pairwise=sum((p>n)+.5*(p==n) for p in positive for n in negative)/(len(positive)*len(negative))
                difference=abs(pairwise-summary['image_auroc'])
                assert difference < 1e-12, (category,seed,variant,difference)
                auc_audit.append({'category':category,'seed':seed,'variant':variant,
                                  'pairwise_image_auroc':pairwise,'difference':difference})
    assert len(auc_audit)==24
    released = {}
    references = {}
    test_features = {}
    for category, entry in manifest['models'].items():
        model_path = ROOT/entry['model']
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == entry['model_sha256']
        model = torch.load(model_path, map_location='cpu', weights_only=True)
        assert model['config']['seed'] == 42 and model['config']['variant'] == 'spatial_kcenter_r3'
        assert model['config']['threads'] == 1 and model['bank'].shape[:2] == (14, 14)
        assert torch.isfinite(model['bank']).all() and torch.isfinite(model['prototype']).all()
        released[category] = model
        run = ROOT/entry['run']
        with (run/'predictions.csv').open(encoding='utf-8', newline='') as stream:
            references[category] = list(csv.DictReader(stream))
        test_features[category] = torch.load(run/'test_features.pt', map_location='cpu', weights_only=True)['cls']
    engine = Engine()
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    checks, results = ['24 image AUROC records independently recomputed from positive/negative score pairs'], []

    def get(path, expected=200):
        try:
            with urllib.request.urlopen(base+path, timeout=90) as response:
                status, data, content_type = response.status, response.read(), response.headers.get('Content-Type')
        except urllib.error.HTTPError as exc:
            status, data, content_type = exc.code, exc.read(), exc.headers.get('Content-Type')
        assert status == expected, (path, status, expected)
        return data, content_type

    def post(payload, expected=200, headers=None):
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(base+'/api/predict', body,
                                        headers=headers or {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status, data = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            status, data = exc.code, json.load(exc)
        assert status == expected, (status, data, expected)
        return data

    try:
        html, content_type = get('/')
        assert b'DEFECT LAB' in html and 'text/html' in content_type
        checks.append('HTML page served')
        model_bytes, _ = get('/api/models')
        catalog = json.loads(model_bytes)
        expected_ids = {'dinov3-'+category for category in CATEGORIES} | {'dinov3-auto', 'bottle', 'screw'}
        actual_ids = {entry['id'] for entry in catalog}
        assert expected_ids <= actual_ids and actual_ids <= expected_ids | {'screw-refined'}
        checks.append('Four DINOSaur categories, automatic routing and two legacy baselines listed')
        prototypes = {category: model['prototype'] for category, model in released.items()}
        for category in CATEGORIES:
            for kind in ['normal', 'defect']:
                payload, _ = get(f'/api/example?model=dinov3-{category}&type={kind}')
                example = json.loads(payload)
                encoded = example['image'].split(',', 1)[1]
                result = post({'model':'dinov3-'+category, 'image':encoded})
                reference_index = next(i for i, row in enumerate(references[category])
                                       if (int(row['label']) == 0) == (kind == 'normal'))
                reference = references[category][reference_index]
                source = ROOT/'data/mvtec'/category/reference['path']
                with Image.open(source) as disk, Image.open(io.BytesIO(base64.b64decode(encoded))) as http:
                    assert disk.size == http.size
                    assert disk.convert('RGB').tobytes() == http.convert('RGB').tobytes()
                difference = abs(result['score']-float(reference['score']))
                assert difference < .0005, (category, kind, difference)
                assert result['anomalous'] == bool(int(reference['prediction']))
                assert abs(result['threshold']-released[category]['threshold']) < 1e-8
                assert result['selected_category'] == category
                assert 'DINOSaur' in result['method']
                assert result['heatmap'].startswith('data:image/png;base64,')
                checks.append(f'{category}/{kind}: uploaded HTTP image agrees with fixed seed42 offline score')
                results.append({'category':category, 'kind':kind, 'path':reference['path'],
                                'score':result['score'], 'threshold':result['threshold'],
                                'anomalous':result['anomalous'], 'inference_ms':result['inference_ms'],
                                'offline_score_difference':difference})
                automatic = post({'model':'dinov3-auto','image':encoded})
                expected_category = choose_task(test_features[category][reference_index], prototypes)
                assert automatic['selected_category'] == expected_category
                # Compare the exact same image with the manually selected routed bank.
                selected = result if expected_category == category else post(
                    {'model':'dinov3-'+expected_category,'image':encoded})
                assert abs(automatic['score']-selected['score']) < 1e-6
                assert automatic['threshold'] == selected['threshold']
                assert automatic.get('category_routing_note')
                checks.append(f'{category}/{kind}: automatic CLS route and selected-bank score agree with offline routing')
                results.append({'category':category,'kind':'auto-'+kind,'selected_category':expected_category,
                                'score':automatic['score'],'anomalous':automatic['anomalous'],
                                'inference_ms':automatic['inference_ms']})
        post({'model':'dinov3-bottle','image':'!!!!'}, 400)
        post({'model':'unknown','image':encoded}, 400)
        post({}, 415, {'Content-Type':'text/plain'})
        post({}, 403, {'Content-Type':'application/json','Origin':'http://example.org'})
        get('/api/example?model=dinov3-unknown&type=normal', 400)
        get('/missing', 404)
        checks.append('Invalid image/model/example, content type, origin and missing path rejected')
        record = {'status':'passed','unit_tests':test_count,'integration_checks':checks,
                  'http_predictions':results,'published_seed':42,'weights_sha256':SHA256,
                  'independent_image_auroc_audit':auc_audit,
                  'source':'Actual temporary localhost HTTP server; actual released tensors and dataset images.',
                  'note':'Inference timings include preprocessing, backbone and scoring; backend startup and rendering excluded.'}
        (ROOT/'reports/FRONTIER_QA.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2),flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


if __name__ == '__main__':
    main()
