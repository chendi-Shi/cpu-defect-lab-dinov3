"""Run a released DINOSaur model on one image using the demo's inference engine.

Example: python frontier_predict.py --image data/mvtec/bottle/test/good/000.png
         --model dinov3-bottle --output-dir reports/frontier/cli-example
The optional output directory receives original.png, heatmap.png and prediction.json.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError


MODEL_IDS = (
    "dinov3-bottle", "dinov3-screw", "dinov3-hazelnut",
    "dinov3-metal_nut", "dinov3-auto",
)


def load_input(path: Path) -> Image.Image:
    """Validate before loading model weights; match the demo's image policy."""
    if not path.is_file():
        raise ValueError(f"找不到图片文件：{path}")
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("图片文件过大，请使用 8 MiB 以内的图片")
    try:
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("请使用 PNG、JPEG 或 WebP 图片")
            if image.width * image.height > 10_000_000:
                raise ValueError("图片过大，请缩小到 1000 万像素以内")
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片，请检查文件格式或文件是否损坏") from exc


def save_png(data_url: str, path: Path) -> None:
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("检测引擎返回了无效图片")
    path.write_bytes(base64.b64decode(data_url[len(prefix):], validate=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, default="dinov3-auto")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        image_path = args.image.expanduser().resolve()
        image = load_input(image_path)
        # Optional runtime import occurs only after image validation.
        from demo_server import Engine

        engine = Engine()
        if args.model not in {item["id"] for item in engine.available_models()}:
            raise ValueError("新版 DINOSaur 模型尚未发布，请先完成实验并运行 frontier_report.py")
        prediction = engine.predict(args.model, image)
        result = {
            "model": args.model,
            "image": str(image_path),
            **{key: prediction[key] for key in (
                "score", "threshold", "anomalous", "method",
                "selected_category", "inference_ms",
            )},
        }
        for note in ("heatmap_note", "category_routing_note"):
            if note in prediction:
                result[note] = prediction[note]
        if args.output_dir is not None:
            output = args.output_dir.expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            files = {"original": str(output / "original.png")}
            save_png(prediction["original"], Path(files["original"]))
            if "heatmap" in prediction:
                files["heatmap"] = str(output / "heatmap.png")
                save_png(prediction["heatmap"], Path(files["heatmap"]))
            result["files"] = files
            (output / "prediction.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
