"""Re-run the fixed ten prior photos and four deterministic extra test samples."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "FRONTIER_IMAGE_TRIALS.json"
SELECTION = ROOT / "reports" / "IMAGE_TRIAL_SELECTION.json"
OLD_RESULTS = ROOT / "reports" / "IMAGE_TRIAL_RESULTS.json"
CATEGORIES = ("bottle", "screw", "hazelnut", "metal_nut")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_images():
    if REPORT.is_file():
        saved = read_json(REPORT)
        items = saved["selection"]["items"]
        if len(items) != 14:
            raise ValueError("Saved image selection must have exactly fourteen images")
        for item in items:
            if sha256(ROOT / item["image"]) != item["sha256"]:
                raise ValueError(f"Image content changed after selection: {item['image']}")
        return saved["selection"]
    prior = read_json(SELECTION)["items"]
    prior_results = {item["image"]: item for item in read_json(OLD_RESULTS)}
    if len(prior) != 10:
        raise ValueError("The fixed prior trial must contain exactly ten photos")
    items = []
    for original in prior:
        item = dict(original)
        path = ROOT / item["image"]
        checksum = sha256(path)
        old = prior_results.get(item["image"])
        if old is None or old["sha256"] != checksum:
            raise ValueError(f"Prior image record/hash is missing or changed: {path}")
        item.update(sha256=checksum, old_result=old["result"], model="dinov3-" + original["model"])
        item["domain_shift"] = item["reference_anomalous"] is None
        items.append(item)
    for category in ("hazelnut", "metal_nut"):
        folder = ROOT / "data" / "mvtec" / category / "test"
        normal = sorted((folder / "good").glob("*.png"))
        defects = sorted(path for path in folder.glob("*/*.png") if path.parent.name != "good")
        if not normal or not defects:
            raise ValueError(f"Missing complete trial images for {category}")
        for path, label in ((normal[0], False), (defects[0], True)):
            items.append({
                "image": path.relative_to(ROOT).as_posix(), "model": "dinov3-" + category,
                "reference_anomalous": label, "reference_type": path.parent.name,
                "source": "MVTec AD test (already included in full frontier evaluation)",
                "source_page": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
                "sha256": sha256(path), "domain_shift": False, "old_result": None,
            })
    if len({item["image"] for item in items}) != 14:
        raise ValueError("Selected trial images contain duplicates")
    return {
        "prior_selection_source": SELECTION.relative_to(ROOT).as_posix(),
        "prior_results_source": OLD_RESULTS.relative_to(ROOT).as_posix(),
        "rule": "Keep the exact previous ten photos; add the lexicographically first normal and first defect PNG for hazelnut and metal_nut. Freeze paths and SHA256 before inference. Never select by predictions.",
        "items": items,
    }


def decode_url(value):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")


def verdict(reference, anomalous):
    if reference is None:
        return {"correct": None, "error_type": None, "check": "domain_shift / no comparable industrial label"}
    if anomalous == reference:
        return {"correct": True, "error_type": None, "check": "correct"}
    return {"correct": False, "error_type": "FP" if anomalous else "FN", "check": "FP" if anomalous else "FN"}


def write_markdown(results, selection, summary):
    lines = [
        "# 新版 DINOv3：固定 14 张图片实际试用", "",
        "使用已发布的固定 seed 42 DINOSaur 主方案和正常验证集阈值，"
        "通过网页服务同一个 Engine 推理函数逐图运行，没有重建库、改阈值、按结果换图或重新下载外部照片。",
        "",
        "前 10 张与旧版试用完全相同，SHA256 与旧记录核对；"
        "另外按文件名排序各取 hazelnut、metal_nut 一张正常和一张缺陷图，路径在推理前冻结。"
        "12 张域内图来自已评估测试集，是直观试用，不能当作新独立验证。"
        "2 张 Wikimedia 普通照片标记 domain_shift，没有可比工业标签，不计检测成功率。",
        "",
        f"域内本次判断：{summary['correct']}/{summary['labeled_images']} 正确，"
        f"FP={summary['fp']}，FN={summary['fn']}。这是固定样本展示，不能代替完整类别成绩。",
        "",
        "| 图片 | 参考标签 | 旧版判断 | DINOv3 分数 | 阈值 | 新版判断 | 核对 | 预处理＋forward＋评分 ms |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for item in results:
        output, reference = item["result"], item["reference_anomalous"]
        truth = "无可比工业标签" if reference is None else ("异常" if reference else "正常")
        prediction = "异常" if output["anomalous"] else "正常"
        old = item["old_result"]
        old_prediction = "—" if old is None else ("异常" if old["anomalous"] else "正常")
        check = "场景外，不计成功率" if item["domain_shift"] else item["check"]
        lines.append(f"| {item['image']} | {truth} | {old_prediction} | {output['score']:.4f} | "
                     f"{output['threshold']:.4f} | {prediction} | {check} | {output['inference_ms']:.1f} |")
    lines += [
        "",
        "计时来自 Engine 的实际预处理＋骨干前向＋异常评分；不含模型加载、磁盘文件解码、热力图渲染和保存。"
        "线程数为 1，后台负载可能变化，不用旧记录计算公平加速比。"
        "两种方法的距离尺度不同，不根据裸分数高低判断优劣。",
        "",
        "![固定瓶口图片](frontier_image_trials/BOTTLE_TRIALS.png)", "",
        "![固定螺丝图片](frontier_image_trials/SCREW_TRIALS.png)", "",
        "![固定新增类别图片](frontier_image_trials/ADDITIONAL_TRIALS.png)", "",
        "![原两张外部照片](frontier_image_trials/EXTERNAL_TRIALS.png)", "",
        "热力图单图归一化；位置颜色不是概率。"
        "外部照片高距离只表示偏离正常工业特征库，不能解释为准确识别瓶子破损。", "",
        "## 来源与复查", "",
        "[路径、SHA256、逐图结果、发布模型哈希](FRONTIER_IMAGE_TRIALS.json)。",
        "",
        "- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)：原图与衍生可视化遵循 CC BY-NC-SA 4.0。",
        "- [DINOv3 模型卡](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)：权重使用官方自定义 DINOv3 License，与项目自编代码许可分开。",
    ]
    for item in selection["items"]:
        if item["domain_shift"]:
            lines.append(f"- [{Path(item['image']).name}]({item['source_page']})：{item['author']}，"
                         f"[{item['license']}]({item['license_url']})；缩放并叠加热力图。")
    (ROOT / "reports" / "FRONTIER_IMAGE_TRIALS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    manifest_path = ROOT / "release" / "frontier" / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Formal frontier release is missing. Complete experiments and run frontier_report.py first.")
    manifest = read_json(manifest_path)
    if set(manifest.get("models", {})) != set(CATEGORIES):
        raise ValueError("All four fixed released models are required")
    selection = select_images()
    record = {
        "schema_version": 1, "state": "selected_before_inference",
        "selection": selection, "release_manifest_sha256": sha256(manifest_path),
        "published_model_sha256": {category: manifest["models"][category]["model_sha256"] for category in CATEGORIES},
        "backbone_sha256": manifest["backbone"]["sha256"],
        "inference_path": "demo_server.Engine.predict (same backend used by /api/predict)",
    }
    write_json(REPORT, record)
    from PIL import Image
    from demo_server import Engine
    from try_images import draw_pairs

    engine = Engine()
    if set(engine.frontier_models) != set(CATEGORIES):
        raise ValueError("Loaded Engine is missing released frontier models")
    destination = ROOT / "reports" / "frontier_image_trials"
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for index, item in enumerate(selection["items"], 1):
        path = ROOT / item["image"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        output = engine.predict(item["model"], image)
        result = dict(item, result=output, **verdict(item["reference_anomalous"], output["anomalous"]))
        for kind in ("original", "heatmap"):
            decode_url(output[kind]).save(destination / f"{index:02d}-{kind}.png")
        results.append(result)
        print(f"{index:02d} {item['image']}: score={output['score']:.5f}, "
              f"threshold={output['threshold']:.5f}, prediction={output['anomalous']}, {result['check']}", flush=True)
    bottle = [item for item in results if item["model"] == "dinov3-bottle" and not item["domain_shift"]]
    screw = [item for item in results if item["model"] == "dinov3-screw"]
    additional = [item for item in results if item["model"] in ("dinov3-hazelnut", "dinov3-metal_nut")]
    external = [item for item in results if item["domain_shift"]]
    draw_pairs(bottle, destination / "BOTTLE_TRIALS.png", "DINOv3: exact previous bottle photos",
               "MVTec AD / CC BY-NC-SA 4.0. Fixed previously evaluated test photos; no new independent-validation claim.\n"
               "Each pair: input / heatmap. Red text = threshold error. Colours rescaled per image.")
    draw_pairs(screw, destination / "SCREW_TRIALS.png", "DINOv3: exact previous screw photos",
               "MVTec AD / CC BY-NC-SA 4.0. Same images as the old trial; no selection by current scores.")
    draw_pairs(additional, destination / "ADDITIONAL_TRIALS.png", "DINOv3: fixed hazelnut and metal nut samples",
               "MVTec AD / CC BY-NC-SA 4.0. Lexicographically first normal and defect per category.\n"
               "These are samples from the already evaluated test set.")
    draw_pairs(external, destination / "EXTERNAL_TRIALS.png", "DINOv3: same two outside-domain web photos",
               "Left: Miika Silfverberg / CC BY-SA 2.0. Right: Ildar Sagdejev / CC BY-SA 4.0. Wikimedia Commons.\n"
               "No comparable industrial ground truth; no detection-success-rate claim.")
    labeled = [item for item in results if not item["domain_shift"]]
    summary = {
        "images": len(results), "labeled_images": len(labeled), "domain_shift_images": len(external),
        "correct": sum(item["correct"] for item in labeled),
        "fp": sum(item["error_type"] == "FP" for item in labeled),
        "fn": sum(item["error_type"] == "FN" for item in labeled),
        "warning": "Fixed visual trial on previously evaluated samples, not an independent generalization estimate; external photos excluded from labeled counts",
    }
    write_markdown(results, selection, summary)
    for item in results:
        item["result"].pop("original")
        item["result"].pop("heatmap")
    record.update(state="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  threads=1, summary=summary, results=results)
    write_json(REPORT, record)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
