"""Local-only image upload demo; no web framework or cloud service required."""
import argparse
import base64
import hashlib
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from lab import ROOT, Extractor, model_features, score_one

MAX_BODY = 8 * 1024 * 1024
FRONTIER_CATEGORIES = ('bottle', 'screw', 'hazelnut', 'metal_nut')
FRONTIER_VARIANT = 'spatial_kcenter_r3'
CATEGORY_NAMES = {'bottle': '瓶口', 'screw': '螺丝', 'hazelnut': '榛子', 'metal_nut': '金属螺母'}


def decode_image(encoded):
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as source:
            if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                raise ValueError('请使用 PNG、JPEG 或 WebP 图片')
            if source.width * source.height > 10_000_000:
                raise ValueError('图片过大，请缩小到 1000 万像素以内')
            return source.convert('RGB')
    except (UnidentifiedImageError, OSError, base64.binascii.Error, TypeError) as exc:
        raise ValueError('无法读取图片，请检查文件格式') from exc


def image_url(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


class Engine:
    def __init__(self, release=ROOT / 'release'):
        torch.set_num_threads(1)
        release = Path(release).resolve()
        self.models = {}
        for category in ['bottle', 'screw']:
            path = release / category / 'model.pt'
            if path.exists():
                self.models[category] = torch.load(path, map_location='cpu', weights_only=True)
        self.frontier_models = {}
        self.prototypes = {}
        self.frontier_manifest = None
        self.dino_extractor = None
        manifest_path = release / 'frontier' / 'manifest.json'
        if manifest_path.is_file():
            self._load_frontier(release, manifest_path)
        if not self.models and not self.frontier_models:
            raise ValueError('No released models. Run finalize.py or frontier_report.py after experiments.')
        self.extractor = Extractor(224) if self.models else None
        self.lock = threading.Lock()

    def _load_frontier(self, release, manifest_path):
        # Import the optional backbone runtime only when a formal release exists.
        from frontier import DinoExtractor, WEIGHTS_SHA, score_features, transform_image
        from frontier_core import choose_task

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entries = manifest.get('models', {})
        if manifest.get('schema_version') != 1 or set(entries) != set(FRONTIER_CATEGORIES):
            raise ValueError('Frontier manifest must publish all four known categories.')
        for category in FRONTIER_CATEGORIES:
            entry = entries[category]
            expected = (release / 'frontier' / category / 'model.pt').resolve()
            declared = (release.parent / entry['model']).resolve()
            if declared != expected or entry.get('seed') != 42 or entry.get('variant') != FRONTIER_VARIANT:
                raise ValueError(f'Invalid fixed frontier release for {category}.')
            if not expected.is_file() or hashlib.sha256(expected.read_bytes()).hexdigest() != entry.get('model_sha256'):
                raise ValueError(f'Frontier release hash mismatch for {category}.')
            saved = torch.load(expected, map_location='cpu', weights_only=True)
            cfg = saved['config']
            if (cfg.get('category') != category or cfg.get('seed') != 42
                    or cfg.get('variant') != FRONTIER_VARIANT or cfg.get('size') != 224
                    or cfg.get('dims') != 384 or cfg.get('weights_sha256') != WEIGHTS_SHA):
                raise ValueError(f'Frontier model configuration mismatch for {category}.')
            bank, prototype = saved['bank'], saved['prototype']
            if (not isinstance(bank, torch.Tensor) or bank.ndim != 4
                    or bank.shape[:2] != (14, 14) or bank.shape[2] < 1
                    or bank.shape[3] != 384 or bank.dtype != torch.float32
                    or not torch.isfinite(bank).all().item()):
                raise ValueError(f'Invalid frontier spatial bank for {category}.')
            if (not isinstance(prototype, torch.Tensor) or prototype.shape != (384,)
                    or prototype.dtype != torch.float32 or not torch.isfinite(prototype).all().item()):
                raise ValueError(f'Invalid frontier CLS prototype for {category}.')
            saved['threshold'] = float(saved['threshold'])
            if not math.isfinite(saved['threshold']):
                raise ValueError(f'Invalid frontier threshold for {category}.')
            self.frontier_models[category] = saved
            self.prototypes[category] = prototype
        self.frontier_manifest = manifest
        self.dino_extractor = DinoExtractor()
        self.dino_transform = transform_image
        self.dino_score = score_features
        self.choose_task = choose_task

    def available_models(self):
        models = []
        for category in self.frontier_models:
            models.append({'id': 'dinov3-' + category, 'family': 'dinov3',
                           'name': CATEGORY_NAMES[category] + ' · DINOSaur 2026',
                           'description': 'DINOSaur 核心方法 CPU 适配：DINOv3 ViT-S/16 特征、位置记忆库与半径 3 邻域检索。'
                                          '使用' + CATEGORY_NAMES[category] + '正常样本建模，阈值由保留的正常训练样本校准。'})
        if self.frontier_models:
            models.append({'id': 'dinov3-auto', 'family': 'dinov3',
                           'name': '自动识别类别 · DINOSaur 2026',
                           'description': '先用 CLS 特征在瓶口、螺丝、榛子、金属螺母四个已知类别中选择，再执行对应类别的异常检测。'
                                          '此路由不识别类别库之外的新类别。'})
        for category in self.models:
            description = ('原版 ResNet18 正常样本匹配基线，用于瓶口俯视图实验和新版方法对照。'
                           if category == 'bottle' else
                           '原版 ResNet18 螺丝研究对照；此前跨类别实验检出效果不足，保留用于比较。')
            models.append({'id': category, 'family': 'resnet18',
                           'name': CATEGORY_NAMES[category] + ' · ResNet18 基线', 'description': description})
        return models

    def example_category(self, model_id):
        if not isinstance(model_id, str):
            raise ValueError('请选择已提供的检测类别')
        if model_id in self.models:
            return model_id
        if model_id == 'dinov3-auto' and self.frontier_models:
            return 'bottle'
        if model_id.startswith('dinov3-') and model_id[7:] in self.frontier_models:
            return model_id[7:]
        raise ValueError('请选择已提供的检测类别')

    def predict(self, model_id, image):
        category = self.example_category(model_id)
        frontier = model_id.startswith('dinov3-')
        with self.lock:
            start = time.perf_counter()
            if frontier:
                tensor = self.dino_transform(image).unsqueeze(0)
                cls, patches = self.dino_extractor.forward_batch(tensor)
                if model_id == 'dinov3-auto':
                    category = self.choose_task(cls[0], self.prototypes)
                saved = self.frontier_models[category]
                score, heat = self.dino_score(patches[0], saved['bank'], FRONTIER_VARIANT)
            else:
                saved = self.models[category]
                cfg = saved['config']
                tensor = self.extractor.transform(image).unsqueeze(0)
                g, l = self.extractor.forward_batch(tensor)
                feature = model_features(g, l, cfg['method'], cfg['dims'], cfg['seed'])[0]
                score, heat = score_one(feature, saved['bank'], cfg['method'], cfg['size'])
            elapsed = (time.perf_counter() - start) * 1000
        original = image.resize((224, 224), Image.Resampling.BILINEAR)
        result = {'category': category, 'score': score, 'threshold': saved['threshold'],
                  'anomalous': score > saved['threshold'], 'inference_ms': elapsed,
                  'method': 'DINOSaur 2026 / DINOv3 ViT-S/16' if frontier else 'ResNet18 基线',
                  'selected_category': category,
                  'original': image_url(original),
                  'heatmap_note': '热力图按当前图片单独缩放；颜色用于显示位置，不代表异常概率。'}
        if model_id == 'dinov3-auto':
            result['category_routing_note'] = ('CLS 最近原型在瓶口、螺丝、榛子、金属螺母四个已知类别中选择；'
                                               '类别库之外的图片也会被分到其中一类，不能据此证明类别识别正确。')
        if heat is not None:
            value = (heat - heat.min()) / max(float(np.ptp(heat)), 1e-8)
            colors = np.stack([value, np.zeros_like(value), 1-value], -1)
            overlay = Image.blend(original, Image.fromarray((colors * 255).astype(np.uint8)), .45)
            result['heatmap'] = image_url(overlay)
        return result


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def respond(self, status, payload, kind='application/json; charset=utf-8'):
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8') if isinstance(payload, (dict, list)) else payload
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                return self.respond(200, (ROOT / 'static' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if parsed.path == '/api/models':
                return self.respond(200, engine.available_models())
            if parsed.path == '/api/example':
                params = parse_qs(parsed.query)
                model_id = params.get('model', ['bottle'])[0]
                kind = params.get('type', ['normal'])[0]
                try:
                    category = engine.example_category(model_id)
                except ValueError:
                    return self.respond(400, {'error': '示例类别无效'})
                if kind not in {'normal', 'defect'}:
                    return self.respond(400, {'error': '示例类别无效'})
                folder = ROOT / 'data' / 'mvtec' / category / 'test'
                paths = sorted(folder.rglob('*.png'))
                candidates = [p for p in paths if (p.parent.name == 'good') == (kind == 'normal')]
                if not candidates:
                    return self.respond(404, {'error': '示例图片尚未下载'})
                with Image.open(candidates[0]) as im:
                    data_url = image_url(im.convert('RGB'))
                return self.respond(200, {'image': data_url, 'name': candidates[0].name})
            self.respond(404, {'error': '页面不存在'})

        def do_POST(self):
            if self.path != '/api/predict':
                return self.respond(404, {'error': '接口不存在'})
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                return self.respond(403, {'error': '请在本机演示页面使用此接口'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    return self.respond(413, {'error': '文件过大，请上传约 5 MB 以内的图片'})
                if not self.headers.get('Content-Type', '').startswith('application/json'):
                    return self.respond(415, {'error': '请使用 JSON 请求'})
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('请求格式无效')
                model_id = request.get('model', 'bottle')
                engine.example_category(model_id)
                image = decode_image(request.get('image', ''))
                result = engine.predict(model_id, image)
                self.respond(200, result)
            except (ValueError, json.JSONDecodeError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                self.respond(500, {'error': '检测失败，请查看模型与环境配置'})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18765)
    args = parser.parse_args()
    engine = Engine()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(engine))
    print(f'Demo ready: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
