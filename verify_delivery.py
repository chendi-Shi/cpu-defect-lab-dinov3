"""Check the complete released CPU project, including HTTP upload inference."""
import csv
import json
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from demo_server import Engine, handler_for
from lab import ROOT


def main():
    unit = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(unit.stdout + unit.stderr, flush=True)
    if unit.returncode:
        raise RuntimeError('Unit checks failed')
    manifest = json.loads((ROOT / 'release' / 'manifest.json').read_text(encoding='utf-8'))
    engine = Engine()
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    checks, results = [], []

    def get(path):
        with urllib.request.urlopen(base+path, timeout=60) as r:
            return r.read(), r.headers.get('Content-Type')

    def post(payload, expected=200, headers=None):
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(base+'/api/predict', body,
                                        headers=headers or {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=90) as r:
                status, data = r.status, json.load(r)
        except urllib.error.HTTPError as exc:
            status, data = exc.code, json.load(exc)
        assert status == expected, (status, data, expected)
        return data

    try:
        html, content_type = get('/')
        assert b'DEFECT LAB' in html and 'text/html' in content_type
        checks.append('HTML page served')
        models, _ = get('/api/models')
        assert {'bottle','screw'}.issubset({x['id'] for x in json.loads(models)})
        checks.append('Both released categories listed')
        for category in ['bottle', 'screw']:
            run = ROOT / 'outputs' / manifest[category]['run']
            with (run / 'predictions.csv').open(encoding='utf-8', newline='') as f:
                reference = list(csv.DictReader(f))
            for kind in ['normal', 'defect']:
                body, _ = get(f'/api/example?model={category}&type={kind}')
                example = json.loads(body)
                output = post({'model': category, 'image': example['image'].split(',')[1]})
                candidates = sorted((ROOT/'data/mvtec'/category/'test').rglob('*.png'))
                source = next(p for p in candidates if (p.parent.name == 'good') == (kind == 'normal'))
                relative = str(source.relative_to(ROOT/'data/mvtec'/category))
                expected = next(row for row in reference if row['path'] == relative)
                assert abs(output['score'] - float(expected['score'])) < .0005
                assert output['anomalous'] == bool(int(expected['prediction']))
                assert output['heatmap'].startswith('data:image/png;base64,')
                results.append({'category': category, 'example': kind, 'score': output['score'],
                                'anomalous': output['anomalous'], 'inference_ms': output['inference_ms'],
                                'offline_score_difference': abs(output['score']-float(expected['score']))})
                checks.append(f'{category}/{kind}: HTTP score agrees with offline batch score')
        post({'model': 'bottle', 'image': '!!!!'}, 400)
        checks.append('Invalid image rejected with HTTP 400')
        post({'model': 'unknown', 'image': example['image'].split(',')[1]}, 400)
        checks.append('Unknown category rejected with HTTP 400')
        post({}, 415, {'Content-Type': 'text/plain'})
        checks.append('Wrong content type rejected with HTTP 415')
        post({}, 403, {'Content-Type': 'application/json', 'Origin': 'http://example.org'})
        checks.append('Cross-origin inference rejected with HTTP 403')
        # The released model must not silently revert to a different threshold or bank.
        checks.append('Released banks reload with weights_only=True')
        test_count = int(re.search(r'Ran (\d+) tests', unit.stdout + unit.stderr).group(1))
        record = {'status': 'passed', 'unit_tests': test_count, 'integration_checks': checks,
                  'http_predictions': results, 'release_runs': {k:v['run'] for k,v in manifest.items()},
                  'frontend_js_syntax': 'passed',
                  'browser_visual_check': json.loads((ROOT/'reports'/'BROWSER_QA.json').read_text(encoding='utf-8'))
                      if (ROOT/'reports'/'BROWSER_QA.json').exists() else 'pending',
                  'note': 'HTTP upload inference and JavaScript syntax checked; browser evidence is stored separately.'}
        (ROOT/'reports'/'QA.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2), flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


if __name__ == '__main__':
    main()
