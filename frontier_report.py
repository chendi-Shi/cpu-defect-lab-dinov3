"""Validate recorded DINOv3 experiments, publish fixed-seed banks, and report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


ROOT = Path(__file__).resolve().parent
CATEGORIES = ("bottle", "screw", "hazelnut", "metal_nut")
SEEDS = (42, 43, 44)
MAIN_VARIANT = "spatial_kcenter_r3"
VARIANTS = (MAIN_VARIANT, "spatial_random_r3", "unrestricted_kcenter", "spatial_kcenter_r0")
FIELDS = (
    "category", "variant", "seed", "image_auroc", "pixel_auroc", "threshold",
    "threshold_metrics", "bank_mib", "bank_vectors", "scoring_median_ms",
    "scoring_p95_ms", "feature_extraction_mean_ms", "coreset_seconds",
    "fit_images", "calibration_images", "test_images", "smoke_run",
)
BEGIN = "<!-- FRONTIER_RESULTS_START -->"
END = "<!-- FRONTIER_RESULTS_END -->"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    return value


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stat(runs, key):
    values = [run["summary"][key] for run in runs if finite(run["summary"].get(key))]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def formatted(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "N/A"


def mean_std(result, digits=4):
    if result["n"] == 0:
        return "N/A"
    if result["n"] == 1:
        return f'{formatted(result["mean"], digits)} (n=1)'
    return f'{formatted(result["mean"], digits)} ± {formatted(result["std"], digits)}'


def verify_predictions(path, summary, config):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != summary["test_images"]:
        raise ValueError(f"Predictions/test count mismatch: {path}")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction"])
        score = float(row["score"])
        if label not in (0, 1) or prediction not in (0, 1) or not finite(score):
            raise ValueError(f"Invalid recorded prediction: {path}")
        if prediction != int(score > summary["threshold"]):
            raise ValueError(f"Prediction/threshold mismatch: {path}")
        key = ("tp" if prediction else "fn") if label else ("fp" if prediction else "tn")
        counts[key] += 1
    if any(counts[name] != summary["threshold_metrics"][name] for name in counts):
        raise ValueError(f"Predictions/confusion metrics mismatch: {path}")
    test_files = config.get("test_files")
    if test_files is not None:
        normalized = lambda value: str(value).replace("\\", "/")
        if {normalized(row["path"]) for row in rows} != {normalized(value) for value in test_files}:
            raise ValueError(f"Predictions/config test files mismatch: {path}")


def collect_frontier(output: Path):
    collected = {}
    for path in sorted(output.glob("*/seed-*/*/summary.json")):
        summary = read_json(path)
        if summary.get("smoke_run"):
            continue
        missing = [field for field in FIELDS if field not in summary]
        if missing:
            raise ValueError(f"{path}: missing summary fields {missing}")
        key = (summary["category"], int(summary["seed"]), summary["variant"])
        if key[0] not in CATEGORIES or key[2] not in VARIANTS:
            continue
        if (path.parent.name != key[2] or path.parent.parent.name != f"seed-{key[1]}"
                or path.parent.parent.parent.name != key[0]):
            raise ValueError(f"Summary identity does not match its path: {path}")
        for name in ("config.json", "predictions.csv"):
            if not (path.parent / name).is_file():
                raise ValueError(f"Missing recorded artifact: {path.parent / name}")
        metrics = summary["threshold_metrics"]
        if any(name not in metrics for name in ("tp", "fp", "fn", "tn", "f1")):
            raise ValueError(f"Missing threshold confusion metrics: {path}")
        if sum(metrics[name] for name in ("tp", "fp", "fn", "tn")) != summary["test_images"]:
            raise ValueError(f"Confusion matrix/test count mismatch: {path}")
        if key in collected:
            raise ValueError(f"Duplicate recorded run: {key}")
        config = read_json(path.parent / "config.json")
        if (config.get("category"), config.get("seed"), config.get("variant")) != key:
            raise ValueError(f"Config/summary identity mismatch: {path}")
        if config.get("split_seed") != 42:
            raise ValueError(f"Formal frontier split must remain seed 42: {path}")
        verify_predictions(path.parent / "predictions.csv", summary, config)
        collected[key] = {"path": path.parent, "summary": summary, "config": config}
    return collected


def check_split(reference, candidate, context):
    normalized_sets = {}
    for name in ("fit_files", "calibration_files", "test_files"):
        left, right = reference.get(name), candidate.get(name)
        if not isinstance(left, list) or not isinstance(right, list):
            raise ValueError(f"Missing explicit {name} for split audit: {context}")
        left_set = {str(item).replace("\\", "/") for item in left}
        right_set = {str(item).replace("\\", "/") for item in right}
        if len(left_set) != len(left) or len(right_set) != len(right):
            raise ValueError(f"Duplicate data paths in {name}: {context}")
        if left_set != right_set:
            raise ValueError(f"Data split {name} differs: {context}")
        normalized_sets[name] = right_set
    if normalized_sets["fit_files"] & normalized_sets["calibration_files"]:
        raise ValueError(f"Fit/threshold calibration overlap: {context}")
    if ((normalized_sets["fit_files"] | normalized_sets["calibration_files"])
            & normalized_sets["test_files"]):
        raise ValueError(f"Training/test overlap: {context}")


def collect_baselines(frontier):
    candidates = defaultdict(list)
    # The output directory is primary. Historical portable report copies are fallback.
    for parent in (ROOT / "outputs", ROOT / "reports" / "run_records"):
        for path in sorted(parent.glob("*/summary.json")):
            config_path = path.parent / "config.json"
            if not config_path.is_file():
                continue
            summary, cfg = read_json(path), read_json(config_path)
            category = cfg.get("category")
            if (category in CATEGORIES and not summary.get("smoke_run", True)
                    and cfg.get("method") == "local" and cfg.get("dims") == 64
                    and cfg.get("bank_size") == 2000 and cfg.get("size") == 224
                    and cfg.get("seed") == 42 and cfg.get("threshold_split", "all") == "all"
                    and not cfg.get("limit_train", 0) and not cfg.get("limit_test", 0)):
                candidates[category].append({"path": path.parent, "summary": summary, "config": cfg})
    selected = {}
    for category, runs in candidates.items():
        selected[category] = max(
            runs, key=lambda run: (run["path"].name, run["path"].parent == ROOT / "outputs")
        )
        main = frontier.get((category, 42, MAIN_VARIANT))
        if main and selected[category]["summary"]["test_images"] != main["summary"]["test_images"]:
            raise ValueError(f"Baseline/frontier test count differs for {category}")
        if main:
            for count in ("fit_images", "calibration_images", "test_images"):
                if selected[category]["summary"][count] != main["summary"][count]:
                    raise ValueError(f"Baseline/frontier {count} differs for {category}")
            check_split(selected[category]["config"], main["config"], f"baseline/frontier {category}")
            for (run_category, seed, variant), run in frontier.items():
                if run_category == category:
                    check_split(main["config"], run["config"], f"{category}/seed-{seed}/{variant}")
    return selected


def expected_missing(runs, baselines):
    missing = []
    for category in CATEGORIES:
        if category not in baselines:
            missing.append(f"baseline/{category}/local64/2000")
        for seed in SEEDS:
            if (category, seed, MAIN_VARIANT) not in runs:
                missing.append(f"{category}/seed-{seed}/{MAIN_VARIANT}")
        for variant in VARIANTS[1:]:
            if (category, 42, variant) not in runs:
                missing.append(f"{category}/seed-42/{variant}")
    return missing


def record_run(run, destination):
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name in ("summary.json", "config.json", "predictions.csv"):
        source = run["path"] / name
        if name.endswith(".json"):
            (destination / name).write_text(
                json.dumps(portable(read_json(source)), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            shutil.copyfile(source, destination / name)
        checksums[name] = digest(source)
    return {"source": relative(run["path"]), "record": relative(destination), "source_sha256": checksums}


def aggregates(runs):
    result = {}
    for category in CATEGORIES:
        result[category] = {}
        for variant in VARIANTS:
            chosen = [runs[(category, seed, variant)] for seed in SEEDS if (category, seed, variant) in runs]
            item = {name: stat(chosen, name) for name in
                    ("image_auroc", "pixel_auroc", "scoring_median_ms", "scoring_p95_ms",
                     "feature_extraction_mean_ms", "coreset_seconds", "bank_mib", "bank_vectors")}
            f1_runs = [{"summary": {"f1": run["summary"]["threshold_metrics"]["f1"]}} for run in chosen]
            item["f1"] = stat(f1_runs, "f1")
            item["seeds"] = [run["summary"]["seed"] for run in chosen]
            result[category][variant] = item
    return result


def publish(runs, provenance):
    models = {}
    for category in CATEGORIES:
        run = runs[(category, 42, MAIN_VARIANT)]
        bank = run["path"] / "model.pt"
        if not bank.is_file() or bank.stat().st_size == 0:
            raise ValueError(f"Missing nonempty release bank: {bank}")
        folder = ROOT / "release" / "frontier" / category
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "summary.json", "config.json"):
            shutil.copyfile(run["path"] / name, folder / name)
        models[category] = {
            "model": relative(folder / "model.pt"), "model_sha256": digest(bank),
            "run": relative(run["path"]), "seed": 42, "variant": MAIN_VARIANT,
            "summary": run["summary"],
        }
    manifest = {"schema_version": 1, "method": "DINOSaur core CPU adaptation",
                "published_seed_rule": "Fixed seed 42; no selection by test performance",
                "backbone": provenance, "models": models}
    manifest_path = ROOT / "release" / "frontier" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def copy_examples(runs):
    examples = {}
    for category in CATEGORIES:
        run = runs.get((category, 42, MAIN_VARIANT))
        if run is None:
            continue
        folder = ROOT / "reports" / "frontier" / "examples" / category
        folder.mkdir(parents=True, exist_ok=True)
        items = []
        for source in sorted(run["path"].glob("*.png")):
            target = folder / source.name
            shutil.copyfile(source, target)
            items.append({"source": relative(source), "record": relative(target),
                          "sha256": digest(source), "threshold_error_example": source.name.startswith("error-")})
        examples[category] = items
    return examples


def defect_counts(path):
    counts = defaultdict(lambda: {"images": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0})
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            item = counts[row["defect"]]
            label, prediction = int(row["label"]), int(row["prediction"])
            key = ("tp" if prediction else "fn") if label else ("fp" if prediction else "tn")
            item["images"] += 1
            item[key] += 1
    return dict(counts)


def all_defect_breakdowns(runs, baselines):
    result = {}
    for category in CATEGORIES:
        primary = runs.get((category, 42, MAIN_VARIANT))
        baseline = baselines.get(category)
        if not primary or not baseline:
            continue
        result[category] = {
            "primary_seed42": defect_counts(primary["path"] / "predictions.csv"),
            "baseline": defect_counts(baseline["path"] / "predictions.csv"),
        }
        new, old = result[category]["primary_seed42"], result[category]["baseline"]
        if set(new) != set(old) or any(new[name]["images"] != old[name]["images"] for name in new):
            raise ValueError(f"Baseline/frontier defect-type test counts differ: {category}")
    return result


def comparison_lines(stats, baselines):
    lines = [
        "| 类别 | ResNet18 局部基线 AUROC | DINOv3 主方案图像 AUROC，mean ± std | 差值 | 主方案 pixel AUROC，mean ± std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category in CATEGORIES:
        main = stats[category][MAIN_VARIANT]
        baseline = baselines.get(category, {}).get("summary", {}).get("image_auroc")
        delta = main["image_auroc"]["mean"] - baseline if finite(baseline) and finite(main["image_auroc"]["mean"]) else None
        lines.append(f"| {category} | {formatted(baseline)} | {mean_std(main['image_auroc'])} | "
                     f"{formatted(delta, 4)} | {mean_std(main['pixel_auroc'])} |")
    return lines


def continual_lines(continual):
    stages = continual.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("continual.json must contain nonempty stages")
    lines = [
        "正常库按类别依次加入，以最近 CLS 原型自动路由测试图。"
        "后续类别加入后，即使原有库向量完全不变，更多候选原型也可能改变路由和最终异常判断。",
        "",
        "| 阶段 | 新加入类别 | 评估类别 | 路由准确率 | 图像 AUROC | TP | FP | FN | TN | 测试数 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    histories = defaultdict(list)
    for stage in stages:
        for name in ("stage", "added_category", "categories", "results", "bank_hashes"):
            if name not in stage:
                raise ValueError(f"continual stage missing {name}")
        for result in stage["results"]:
            category = result["category"]
            if sum(result[name] for name in ("tp", "fp", "fn", "tn")) != result["test_images"]:
                raise ValueError(f"Continual confusion/test count differs: stage {stage['stage']}/{category}")
            histories[category].append((stage["stage"], result))
            lines.append(
                f"| {stage['stage']} | {stage['added_category']} | {category} | "
                f"{formatted(result['routing_accuracy'])} | {formatted(result['image_auroc'])} | "
                f"{result['tp']} | {result['fp']} | {result['fn']} | {result['tn']} | {result['test_images']} |"
            )
    unchanged = continual.get("bank_contents_unchanged")
    lines += ["", f"记录中的已有库内容保持不变检查：{unchanged}。", "",
              "| 类别 | 首次出现阶段 | 最后阶段 | AUROC 首次 | AUROC 最后 | 最后－首次 | 路由准确率 首次→最后 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for category in CATEGORIES:
        if not histories[category]:
            continue
        first_stage, first = histories[category][0]
        final_stage, final = histories[category][-1]
        delta = final["image_auroc"] - first["image_auroc"]
        lines.append(f"| {category} | {first_stage} | {final_stage} | "
                     f"{formatted(first['image_auroc'])} | {formatted(final['image_auroc'])} | "
                     f"{formatted(delta)} | {formatted(first['routing_accuracy'])}→{formatted(final['routing_accuracy'])} |")
    lines += [
        "",
        "上表差值只是本次固定 seed 42、同一测试集、该类别加入顺序下的实际观测。"
        "库哈希不变不能直接证明实际系统零遗忘；判断还依赖 CLS 路由。"
        "本序列也不替代原论文的连续漂移、逻辑异常或边缘硬件 benchmark。",
    ]
    return lines


def report_lines(runs, baselines, stats, provenance, missing, continual, examples, breakdowns):
    lines = [
        "# DINOv3／DINOSaur 本机实验报告", "",
        "实现 2026 年 DINOSaur 的冻结 DINOv3 特征、空间索引 coreset 与邻域限制评分核心，"
        "在本机 CPU 上与原有 ResNet18 局部特征基线对照。"
        "本实验是四个 MVTec AD 类别的核心方法适配，不能据此宣称复现整篇持续学习与边缘硬件 benchmark，也不能宣称全球 SOTA。",
        "", "## 协议与来源", "",
        "- 主方案：" + MAIN_VARIANT + "；固定 coreset seeds 42、43、44，数据划分保持不变。",
        "- 模型发布固定 seed 42；消融至少完成 seed 42，表中明确实际重复次数。",
        "- 仅正常训练图建库，互斥 held-out 正常训练图的分数第 95 百分位确定阈值；测试集或缺陷标签不用于选阈值。",
        "- 图像输入 224×224；pixel AUROC 在本实验 224 网格计算，不是 MVTec 原图尺度的官方定位指标。",
        "- 三次 coreset seed 是同一测试集上的随机建库重复，标准差不是独立数据集置信区间。",
        "- bottle、screw 测试集在早期基线实验中已经查看过，结果属于探索性复测；hazelnut、metal_nut 的首次评估可单独查记录。",
        f"- 权重：[{provenance.get('model_id', 'DINOv3')}](https://huggingface.co/{provenance.get('model_id', 'timm/vit_small_patch16_dinov3.lvd1689m')})，"
        f"revision {provenance.get('model_revision', 'N/A')}，SHA256 {provenance.get('sha256', 'N/A')}。",
        f"- 运行库 timm {provenance.get('runtime_version', 'N/A')}，CPU float32。"
        "token 使用最终 LayerNorm 输出，不另作 L2 归一化，以官方核心源码为准。",
        "- 本 pipeline 按模型卡对 RoPE periods 作 bf16→float32 精度截断；"
        "timm API 替代仍不宣称与 Meta 原运行库逐位一致。"
        "下载 provenance 中的 RoPE 差异说明描述 timm 默认构造行为，须与此适配设置一起解读。",
        f"- [权重许可证]({provenance.get('license_url', 'https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md')})；"
        "[DINOSaur 官方代码](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD)，"
        "[论文](https://arxiv.org/abs/2605.24251)。",
        "- DINOv3 权重使用官方自定义 DINOv3 License，与项目自编代码许可分开；"
        "不将模型权重许可写成 MIT 或 Apache。"
        "[MVTec AD 原图及衍生可视化](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)遵循 CC BY-NC-SA 4.0。",
        "", "## 与原有基线比较", "",
    ]
    if missing:
        lines += ["**未完成报告：缺少正式记录，禁止作为最终成绩或发布版本。**", "",
                  *[f"- {item}" for item in missing], ""]
    lines += comparison_lines(stats, baselines)
    deltas = []
    for category in CATEGORIES:
        main = stats[category][MAIN_VARIANT]["image_auroc"]["mean"]
        old = baselines.get(category, {}).get("summary", {}).get("image_auroc")
        if finite(main) and finite(old):
            deltas.append((category, main - old))
    if deltas:
        improved = [category for category, delta in deltas if delta > 0]
        declined = [category for category, delta in deltas if delta < 0]
        lines += ["", f"按图像 AUROC 的已记录均值，改善类别：{', '.join(improved) or '无'}；"
                  f"下降类别：{', '.join(declined) or '无'}。每类结果均保留，不以单类别高分代替通用有效性。"]
    lines += ["", "## 消融", "",
              "| 类别 | 方案 | 实际 seeds | 图像 AUROC | pixel AUROC | F1 |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for category in CATEGORIES:
        for variant in VARIANTS:
            item = stats[category][variant]
            lines.append(f"| {category} | {variant} | {', '.join(map(str, item['seeds'])) or '缺失'} | "
                         f"{mean_std(item['image_auroc'])} | {mean_std(item['pixel_auroc'])} | {mean_std(item['f1'])} |")
    lines += ["", "## 固定阈值下的实际判断", "",
              "| 类别 | seed | 阈值 | TP | FP | FN | TN | F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category in CATEGORIES:
        for seed in SEEDS:
            run = runs.get((category, seed, MAIN_VARIANT))
            if run:
                summary, metrics = run["summary"], run["summary"]["threshold_metrics"]
                lines.append(f"| {category} | {seed} | {formatted(summary['threshold'], 6)} | "
                             f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} | {metrics['tn']} | {formatted(metrics['f1'])} |")
    lines += ["", "## 每种缺陷与正常图的错误分布", "",
              "以下直接从固定 seed 42 的正式 predictions.csv 汇总，没有重跑、改图、改阈值或改参数。"
              "异常类型重点看漏检 FN，good 重点看误报 FP；所有已记录类型都保留。",
              "",
              "| 类别 | 类型 | n | 新 TP | 新 FP | 新 FN | 新 TN | 旧 FN | 旧 FP |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category in CATEGORIES:
        if category not in breakdowns:
            continue
        new = breakdowns[category]["primary_seed42"]
        old = breakdowns[category]["baseline"]
        for kind in sorted(new):
            item, previous = new[kind], old[kind]
            lines.append(f"| {category} | {kind} | {item['images']} | {item['tp']} | {item['fp']} | "
                         f"{item['fn']} | {item['tn']} | {previous['fn']} | {previous['fp']} |")
    if "screw" in breakdowns:
        new = breakdowns["screw"]["primary_seed42"]
        old = breakdowns["screw"]["baseline"]
        defect_types = [name for name in new if new[name]["tp"] + new[name]["fn"] > 0]
        if defect_types:
            worst = max(defect_types, key=lambda name: new[name]["fn"] / (new[name]["tp"] + new[name]["fn"]))
            improved = [f"{name}（FN {old[name]['fn']}→{new[name]['fn']}）"
                        for name in sorted(defect_types) if new[name]["fn"] < old[name]["fn"]]
            lines += ["", "螺丝漏检减少的类型：" + ("；".join(improved) or "无") + "。",
                      f"新版 seed 42 漏检率最高类型为 {worst}："
                      f"{new[worst]['fn']}/{new[worst]['tp'] + new[worst]['fn']}。"
                      "改善不表示生产可用；剩余漏检与误报仍需正面处理。"
                      "旋转、尺度或细小缺陷可能影响空间匹配，但这只是待验证假设，错误表不能单独证明因果。"]
    lines += ["", "## CPU 时间与特征库", "",
              "| 类别 | 新方案 threads | 基线 threads | 库 MiB | 库向量数 | 特征提取 mean ms | 评分 median ms | 评分 p95 ms | coreset 秒 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category in CATEGORIES:
        item = stats[category][MAIN_VARIANT]
        threads = sorted({run["config"].get("threads", "未知") for key, run in runs.items()
                          if key[0] == category and key[2] == MAIN_VARIANT}, key=str)
        baseline_threads = baselines.get(category, {}).get("config", {}).get("threads", "未知")
        lines.append(f"| {category} | {', '.join(map(str, threads))} | {baseline_threads} | "
                     f"{mean_std(item['bank_mib'], 2)} | {mean_std(item['bank_vectors'], 0)} | "
                     f"{mean_std(item['feature_extraction_mean_ms'], 1)} | {mean_std(item['scoring_median_ms'], 1)} | "
                     f"{mean_std(item['scoring_p95_ms'], 1)} | {mean_std(item['coreset_seconds'], 1)} |")
    lines += ["", "评分时间只包括已提取特征的异常评分，不能当作完整单图推理耗时；"
              "特征提取 mean 与评分 median 的相加也不是端到端 median。"
              "完整耗时应看网页 API 的 preprocess＋forward＋score 实测。库 MiB 是保存特征占用，不是进程内存峰值。"]
    lines += ["", "新方案与历史基线的线程数和运行时后台负载不同，"
              "这些时间记录只能描述各次实际运行，不能直接计算公平加速比或推断某方法必然更快。"]
    lines += ["", "## 持续加入类别", ""]
    if continual is None:
        lines += ["本报告未记录持续加入类别实验，不宣称完成原论文的持续学习 benchmark。"]
    else:
        lines += continual_lines(continual)
    lines += ["", "## 定位与错误样本", "",
              "保留固定 seed 42 主方案已生成的全部 PNG，包含程序保存的 error-* 失败图，"
              "无需重新推理或按新成绩换图。下图顺序为输入／标注／热力图；热力图颜色是单图缩放，不能当作异常概率。"]
    for category in CATEGORIES:
        items = examples.get(category, [])
        lines += ["", f"### {category}", ""]
        errors = [item for item in items if item["threshold_error_example"]]
        ordinary = [item for item in items if not item["threshold_error_example"]]
        shown = ordinary[:1] + errors[:2]
        for item in shown:
            path = item["record"].removeprefix("reports/")
            lines += [f"![{Path(path).name}]({path})", ""]
        if items:
            lines += ["全部已保存图（包含其余失败图）：", ""]
            for item in items:
                path = item["record"].removeprefix("reports/")
                lines.append(f"- [{Path(path).name}]({path})")
        else:
            lines += ["本运行未保存可视化 PNG；逐图分数和阈值错误以 predictions.csv 为准。"]
    lines += ["", "## 证据与可复现范围", "",
              "- 每次正式运行的 config、summary、predictions 保存在 reports/frontier/run_records。",
              "- reports/frontier/evidence.json 记录完整运行来源、输入统计、文件 SHA256、结果汇总与缺失检查。",
              "- release/frontier/<category>/model.pt 使用固定 seed 42 主方案；backbone 权重在 cache/frontier，本源码包不附权重。",
              "- 当前类别有限；无法将工业俯视样本检测结果推广到任意网络照片。"]
    return lines


def plot(stats, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
    xs = np.arange(len(CATEGORIES))
    colors = ("#738398", "#008a9c")
    for ax, key, title in zip(axes, ("image_auroc", "pixel_auroc"), ("Image AUROC", "Pixel AUROC at 224 grid")):
        old = [baselines.get(category, {}).get("summary", {}).get(key, math.nan) for category in CATEGORIES]
        old = [value if finite(value) else math.nan for value in old]
        new = [stats[category][MAIN_VARIANT][key]["mean"] for category in CATEGORIES]
        new = [value if finite(value) else math.nan for value in new]
        err = [stats[category][MAIN_VARIANT][key]["std"] or 0 for category in CATEGORIES]
        ax.bar(xs - .18, old, .34, label="ResNet18 baseline", color=colors[0])
        ax.bar(xs + .18, new, .34, yerr=err, capsize=4, label="DINOv3 spatial coreset", color=colors[1])
        ax.set_xticks(xs, [category.replace("_", " ") for category in CATEGORIES])
        ax.set_title(title, pad=12)
        ax.set_ylim(0, 1.07)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .13), ncol=2, frameon=False)
    fig.suptitle("Frozen DINOv3 / DINOSaur core: CPU adaptation", fontsize=15, y=.96)
    fig.text(.5, .055, "Mean of 3 coreset seeds; error bars = sample SD. Same test split. No claim of global SOTA.",
             ha="center", fontsize=9, color="#45505c")
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.28, wspace=.20)
    path = ROOT / "reports" / "frontier" / "RESULTS.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def update_interview(stats, baselines, runs, breakdowns):
    path = ROOT / "reports" / "FRONTIER_INTERVIEW.md"
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    if BEGIN not in content or END not in content:
        return
    left, rest = content.split(BEGIN, 1)
    _, right = rest.split(END, 1)
    body = "\n\n" + "\n".join(comparison_lines(stats, baselines)) + "\n\n"
    screw = runs.get(("screw", 42, MAIN_VARIANT))
    if screw:
        metrics = screw["summary"]["threshold_metrics"]
        body += (f"螺丝固定 seed 42：TP={metrics['tp']}、FP={metrics['fp']}、"
                 f"FN={metrics['fn']}、TN={metrics['tn']}，F1={formatted(metrics['f1'])}。"
                 "必须同时讲仍有多少漏检，不能把相对基线改善称为生产可用。\n\n")
        if "screw" in breakdowns:
            kinds = breakdowns["screw"]["primary_seed42"]
            anomalous = [kind for kind in kinds if kinds[kind]["tp"] + kinds[kind]["fn"] > 0]
            if anomalous:
                worst = max(anomalous, key=lambda kind: kinds[kind]["fn"] / (kinds[kind]["tp"] + kinds[kind]["fn"]))
                body += (f"seed 42 漏检率最高的螺丝类型为 {worst}，"
                         f"FN={kinds[worst]['fn']}/{kinds[worst]['tp'] + kinds[worst]['fn']}。"
                         "旋转或细小缺陷仅是可能解释，尚未通过控制变量实验确认。\n\n")
    path.write_text(left + BEGIN + body + END + right, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "frontier")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write clearly marked draft; never publish banks")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    runs = collect_frontier(args.output.resolve())
    baselines = collect_baselines(runs)
    missing = expected_missing(runs, baselines)
    if missing and not args.allow_incomplete:
        raise ValueError("Formal experiment matrix is incomplete:\n" + "\n".join(missing))
    provenance_path = ROOT / "cache" / "frontier" / "backbone_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else {}
    if not provenance and not args.allow_incomplete:
        raise ValueError("Missing verified backbone provenance")
    stats = aggregates(runs)
    report_dir = ROOT / "reports" / "frontier"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for (category, seed, variant), run in runs.items():
        key = f"{category}/seed-{seed}/{variant}"
        records[key] = record_run(run, report_dir / "run_records" / category / f"seed-{seed}" / variant)
    baseline_records = {category: record_run(run, report_dir / "run_records" / "baselines" / category)
                        for category, run in baselines.items()}
    examples = copy_examples(runs)
    breakdowns = all_defect_breakdowns(runs, baselines)
    continual_path = args.output / "continual.json"
    continual = read_json(continual_path) if continual_path.is_file() else None
    evidence = {
        "schema_version": 1, "complete": not missing, "missing": missing,
        "main_variant": MAIN_VARIANT, "coreset_seeds": list(SEEDS), "published_seed": 42,
        "backbone_provenance": provenance, "statistics": stats,
        "runs": records, "baselines": baseline_records, "examples": examples,
        "defect_breakdowns_seed42": breakdowns, "continual": portable(continual),
        "metric_note": "Pixel AUROC on 224 grid; scoring time excludes feature extraction; repeated seeds are not independent test sets",
    }
    if not missing:
        evidence["release_manifest"] = relative(ROOT / "release" / "frontier" / "manifest.json")
        publish(runs, provenance)
    (report_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = report_lines(runs, baselines, stats, provenance, missing, continual, examples, breakdowns)
    if not args.skip_plot:
        image_path = plot(stats, baselines)
        lines += ["", "![四类别比较](frontier/RESULTS.png)"]
    (ROOT / "reports" / "FRONTIER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_interview(stats, baselines, runs, breakdowns)
    print(json.dumps({"complete": not missing, "formal_runs": len(runs), "baseline_categories": list(baselines),
                      "report": relative(ROOT / "reports" / "FRONTIER.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
