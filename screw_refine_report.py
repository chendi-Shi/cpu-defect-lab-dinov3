"""Build reports from completed screw-refinement records, never from guesses.

This script performs no model inference and cannot select a new winner. It
requires both development_complete.json and evaluation.json, independently
checks recorded image metrics, and exports portable evidence without models,
feature caches or the raw dataset. Run after screw_refine.py has completed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/screw_refinement"
REPORTS = ROOT / "reports"
DESTINATION = REPORTS / "screw_refinement"
ORDER = ("raw224", "aligned224", "raw336", "multi224_336")
KINDS = ("scratch", "pit", "texture_shift")
DISPLAY = {"raw224": "原始224", "aligned224": "姿态对齐224",
           "raw336": "原始336", "multi224_336": "224＋336多尺度"}
KIND_DISPLAY = {"scratch": "细划痕", "pit": "暗斑/凹点", "texture_shift": "局部纹理位移"}


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"缺少真实完成产物：{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), ".").replace(ROOT.as_posix(), ".")
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(portable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        values = list(csv.DictReader(stream))
    for row in values:
        if "kind" not in row and "defect" in row:
            row["kind"] = row["defect"]
        for key in ("label", "prediction"):
            if key in row:
                row[key] = int(row[key])
        for key in ("score", "scoring_ms"):
            if key in row:
                row[key] = float(row[key])
                if not math.isfinite(row[key]):
                    raise ValueError(f"Non-finite {key}: {path}")
    return values


def auc_pairwise(values):
    positives = [row["score"] for row in values if row["label"] == 1]
    negatives = [row["score"] for row in values if row["label"] == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    wins = sum((positive > negative) + .5 * (positive == negative)
               for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def proxy_check(values, recorded, sources, context):
    normal = [row for row in values if row["label"] == 0]
    if {row["source"] for row in normal} != sources or len(normal) != len(sources):
        raise ValueError(f"Normal source split mismatch: {context}")
    computed = {}
    for kind in KINDS:
        defects = [row for row in values if row["kind"] == kind]
        if len(defects) != len(sources) or {row["source"] for row in defects} != sources:
            raise ValueError(f"Synthetic source/type mismatch: {context}/{kind}")
        computed[kind] = auc_pairwise(normal + defects)
        if abs(computed[kind] - recorded["by_kind_auroc"][kind]) > 1e-12:
            raise ValueError(f"Independent proxy AUROC mismatch: {context}/{kind}")
    if abs(float(np.mean(list(computed.values()))) - recorded["macro_auroc"]) > 1e-12:
        raise ValueError(f"Independent macro AUROC mismatch: {context}")
    return computed


def prediction_check(values, summary, context):
    if len(values) != 160 or sum(row["label"] == 1 for row in values) != 119:
        raise ValueError(f"Unexpected real test counts: {context}")
    if len({row["path"] for row in values}) != 160:
        raise ValueError(f"Duplicate real test path: {context}")
    counts = dict(tp=0, fn=0, fp=0, tn=0)
    for row in values:
        if row["label"] not in (0, 1) or row["prediction"] not in (0, 1):
            raise ValueError(f"Invalid binary labels: {context}")
        if row["prediction"] != int(row["score"] > summary["threshold"]):
            raise ValueError(f"Threshold/prediction mismatch: {context}")
        key = ("tp" if row["prediction"] else "fn") if row["label"] else ("fp" if row["prediction"] else "tn")
        counts[key] += 1
    for key, count in counts.items():
        if count != summary["threshold_metrics"][key]:
            raise ValueError(f"Confusion matrix mismatch: {context}/{key}")
    if abs(auc_pairwise(values) - summary["image_auroc"]) > 1e-12:
        raise ValueError(f"Independent real image AUROC mismatch: {context}")
    return counts


def historical_directory():
    options = [ROOT / "outputs/frontier/screw/seed-42/spatial_kcenter_r3",
               REPORTS / "frontier/run_records/screw/seed-42/spatial_kcenter_r3"]
    for candidate in options:
        if all((candidate / name).is_file() for name in ("config.json", "summary.json", "predictions.csv")):
            return candidate
    raise FileNotFoundError("Missing recorded historical 256-image DINOv3 screw run")


def load_evidence():
    # No output directory is created until both actual stages exist and validate.
    complete = read_json(OUTPUT / "development_complete.json")
    evaluation = read_json(OUTPUT / "evaluation.json")
    protocol = read_json(OUTPUT / "protocol.json")
    selection = read_json(OUTPUT / "selection.json")
    split = read_json(OUTPUT / "splits.json")
    normalization = read_json(OUTPUT / "normalization.json")
    manifest = read_json(OUTPUT / "synthetic_manifest.json")
    if complete["selection"] != selection or evaluation["selected"] != selection["selected"]:
        raise ValueError("Stage selection identities differ")
    if selection.get("test_used_for_selection") or selection.get("audit_used_for_selection"):
        raise ValueError("Recorded selection violates the frozen stage boundary")
    for name, expected in complete.get("artifacts", {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    for name, key in (("protocol.json", "protocol_sha256"), ("splits.json", "split_sha256"),
                      ("normalization.json", "normalization_sha256")):
        if sha(OUTPUT / name) != selection[key]:
            raise ValueError(f"Selection references a different {name}")
    source_files = {"proxy_generator_source_sha256": "screw_refine.py",
                    "geometry_source_sha256": "screw_geometry.py",
                    "runtime_source_sha256": "refine_features.py",
                    "coreset_source_sha256": "frontier_core.py",
                    "legacy_transform_source_sha256": "frontier.py",
                    "metrics_source_sha256": "metrics.py"}
    for key, name in source_files.items():
        if key in protocol and sha(ROOT / name) != protocol[key]:
            raise ValueError(f"Frozen source changed: {name}")
    dependency_path = ROOT / "delivery/github-refinement-source-audit.json"
    dependency_audit = read_json(dependency_path) if dependency_path.is_file() else None
    if dependency_audit is not None:
        for name, expected in dependency_audit["source_hashes"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Supplementary dependency audit no longer matches: {name}")
        if dependency_audit["protocol_sha256"] != sha(OUTPUT / "protocol.json"):
            raise ValueError("Supplementary source audit references another protocol")
    groups = {key: {entry["path"] for entry in values} for key, values in split["groups"].items()}
    if {key: len(value) for key, value in groups.items()} != {
            "fit": 192, "select": 48, "audit": 16, "calibration": 64}:
        raise ValueError("Expected 192/48/16/64 normal source split")
    if sum(map(len, groups.values())) != len(set.union(*groups.values())):
        raise ValueError("Normal source split overlaps")
    selected = selection["selected"]
    active = [candidate for candidate in ORDER if isinstance(selection["metrics"].get(candidate), dict)
              and isinstance(selection["metrics"][candidate].get("macro_auroc"), (int, float))
              and math.isfinite(selection["metrics"][candidate]["macro_auroc"])]
    if selected not in active or "raw224" not in active:
        raise ValueError("Invalid active/winner candidates")
    highest = max(selection["metrics"][candidate]["macro_auroc"] for candidate in active)
    best = next(candidate for candidate in active
                if highest - selection["metrics"][candidate]["macro_auroc"] <= 1e-9)
    expected = best if selection["metrics"][best]["macro_auroc"] >= (
        selection["metrics"]["raw224"]["macro_auroc"] + selection["minimum_gain"]) else "raw224"
    if selected != expected:
        raise ValueError("Winner does not follow the frozen minimum-gain/tie rule")
    development = {}
    for candidate in active:
        path = OUTPUT / candidate / "development_predictions.csv"
        if sha(path) != selection["development_csv_sha256"][candidate]:
            raise ValueError(f"Selection development CSV changed: {candidate}")
        development[candidate] = rows(path)
        proxy_check(development[candidate], selection["metrics"][candidate], groups["select"], candidate)
    evaluated = list(dict.fromkeys(("raw224", selected)))
    if set(evaluation["results"]) != set(evaluated):
        raise ValueError("Only baseline and the preselected winner may be real-test evaluated")
    audits, calibrations, predictions = {}, {}, {}
    for candidate in evaluated:
        audits[candidate] = rows(OUTPUT / candidate / "audit_predictions.csv")
        proxy_check(audits[candidate], complete["audit"][candidate], groups["audit"], f"audit/{candidate}")
        calibrations[candidate] = rows(OUTPUT / candidate / "calibration_predictions.csv")
        if len(calibrations[candidate]) != 64 or {
                row["path"] for row in calibrations[candidate]} != groups["calibration"]:
            raise ValueError(f"Calibration split mismatch: {candidate}")
        threshold = float(np.quantile([row["score"] for row in calibrations[candidate]], .95))
        if abs(threshold - evaluation["results"][candidate]["threshold"]) > 1e-12:
            raise ValueError(f"Calibration threshold mismatch: {candidate}")
        predictions[candidate] = rows(OUTPUT / candidate / "test_predictions.csv")
        prediction_check(predictions[candidate], evaluation["results"][candidate], candidate)
    history_dir = historical_directory()
    history = read_json(history_dir / "summary.json")
    history_predictions = rows(history_dir / "predictions.csv")
    prediction_check(history_predictions, history, "historical256")
    expected_paths = {row["path"] for row in predictions["raw224"]}
    if any({row["path"] for row in values} != expected_paths for values in predictions.values()):
        raise ValueError("New baseline/winner real test paths differ")
    if {row["path"] for row in history_predictions} != expected_paths:
        raise ValueError("Historical and new real test paths differ")
    return dict(complete=complete, evaluation=evaluation, protocol=protocol, selection=selection,
                split=split, groups=groups, normalization=normalization, synthetic=manifest,
                active=active, evaluated=evaluated, development=development, audits=audits,
                calibrations=calibrations, predictions=predictions, history=history,
                history_predictions=history_predictions, history_dir=history_dir,
                dependency_audit=dependency_audit)


def export_evidence(data, micro_probe=None, micro_source=None):
    directory = DESTINATION / "run_records"
    directory.mkdir(parents=True, exist_ok=True)
    exported = []

    def copy(source, target):
        source, target = Path(source), Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".json":
            write_json(target, read_json(source))
        else:
            shutil.copyfile(source, target)
        exported.append(dict(source=relative(source), exported=relative(target),
                             source_sha256=sha(source), exported_sha256=sha(target), bytes=target.stat().st_size))

    # Explicit evidence formats only: no .pt weights, cache tensor or raw images.
    for source in sorted(OUTPUT.rglob("*.json")):
        copy(source, directory / source.relative_to(OUTPUT))
    for source in sorted(OUTPUT.rglob("*.csv")):
        copy(source, directory / source.relative_to(OUTPUT))
    copy(ROOT / "SCREW_REFINEMENT_PROTOCOL.md", directory / "SCREW_REFINEMENT_PROTOCOL.md")
    for name in ("config.json", "summary.json", "predictions.csv"):
        copy(data["history_dir"] / name, directory / "historical_256_fit" / name)
    if micro_probe is not None:
        copy(micro_probe, directory / "resource" / Path(micro_probe).name)
    if micro_source is not None:
        copy(micro_source, directory / "resource" / Path(micro_source).name)
    if data.get("dependency_audit") is not None:
        copy(ROOT / "delivery/github-refinement-source-audit.json", directory / "source_dependency_audit.json")
    for candidate in data["evaluated"]:
        for source in sorted((OUTPUT / candidate).glob("*.png")):
            copy(source, DESTINATION / "examples" / candidate / source.name)
    identity = {key: value for key, value in data["protocol"].items()
                if "sha256" in key or "version" in key or key in ("retrieval", "weights_sha256")}
    identity.update(report_script_sha256=sha(Path(__file__)),
                    evidence_note="JSON copies replace local workspace prefixes with '.'; original and exported hashes are recorded separately.",
                    excluded=["model.pt", "pretrained weights", "feature caches", "raw dataset"])
    write_json(directory / "source_identity.json", identity)
    write_json(directory / "export_manifest.json", exported)
    return exported


def synthetic_examples(data):
    destination = DESTINATION / "examples/synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 23)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    names = {}
    for kind in KINDS:
        record = next(item for item in data["synthetic"] if item["partition"] == "select" and item["kind"] == kind)
        paths = (ROOT / record["source"], ROOT / record["mask"], ROOT / record["path"])
        for path, hash_key in ((paths[1], "mask_sha256"), (paths[2], "sha256")):
            if sha(path) != record[hash_key]:
                raise ValueError(f"Synthetic example content differs: {path}")
        panel = Image.new("RGB", (1224, 478), "white")
        draw = ImageDraw.Draw(panel)
        titles = ("Original normal", "Procedural defect mask", f"Synthetic {kind}")
        for index, (path, title) in enumerate(zip(paths, titles)):
            with Image.open(path) as opened:
                image = opened.convert("RGB").resize((392, 392), Image.Resampling.BILINEAR)
            x = 12 + index * 408
            panel.paste(image, (x, 50))
            draw.text((x, 15), title, fill="#182538", font=font)
        draw.text((12, 451), f"Fixed first selection source: {Path(record['source']).name}; seed={record['seed']}. Synthetic proxy only.",
                  fill="#334155", font=small)
        path = destination / f"{kind}.png"
        panel.save(path)
        names[kind] = path
    return names


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "未记录"


def rates(metrics):
    recall = metrics["tp"] / max(1, metrics["tp"] + metrics["fn"])
    false_positive = metrics["fp"] / max(1, metrics["fp"] + metrics["tn"])
    return recall, false_positive


def per_kind(values):
    result = {}
    for row in values:
        item = result.setdefault(row["kind"], dict(n=0, fn=0, fp=0))
        item["n"] += 1
        item["fn"] += int(row["label"] == 1 and row["prediction"] == 0)
        item["fp"] += int(row["label"] == 0 and row["prediction"] == 1)
    return result


def paired_changes(left, right):
    first = {row["path"]: row for row in left}
    improvements, regressions = [], []
    for row in right:
        old = first[row["path"]]
        before, after = old["prediction"] == old["label"], row["prediction"] == row["label"]
        if after and not before:
            improvements.append(row["path"])
        if before and not after:
            regressions.append(row["path"])
    return improvements, regressions


def fallback_table(data):
    path = OUTPUT / "geometry_development.json"
    geometry = read_json(path) if path.is_file() else {}
    result = []
    for role, label in (("fit", "正常建库"), ("select", "选择正常"), ("audit", "审计正常"), ("calibration", "阈值校准正常")):
        names = data["groups"][role]
        if all(name in geometry for name in names):
            failed = sum(bool(geometry[name]["fallback"]) for name in names)
            result.append((label, len(names), len(names) - failed, failed))
        else:
            failed = data["complete"].get("alignment_fallback_counts", {}).get(role)
            result.append((label, len(names), len(names)-failed if failed is not None else None, failed))
    for partition, label in (("select", "选择合成缺陷"), ("audit", "审计合成缺陷")):
        names = [item["path"] for item in data["synthetic"] if item["partition"] == partition]
        additional = OUTPUT / f"geometry_{'audit' if partition == 'audit' else 'development'}.json"
        details = read_json(additional) if additional.is_file() else geometry
        if all(name in details for name in names):
            failed = sum(bool(details[name]["fallback"]) for name in names)
            result.append((label, len(names), len(names)-failed, failed))
        else:
            result.append((label, len(names), None, None))
    if data["selection"]["selected"] == "aligned224":
        for label, values in (("真实测试正常", [row for row in data["predictions"]["aligned224"] if row["label"] == 0]),
                              ("真实测试缺陷", [row for row in data["predictions"]["aligned224"] if row["label"] == 1])):
            failed = sum(row.get("branch") == "raw224" for row in values)
            result.append((label, len(values), len(values)-failed, failed))
    else:
        result.append(("真实测试（winner未使用对齐）", 160, None, None))
    return result


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = data["selection"]["metrics"]
    active = data["active"]
    colors = ["#0f766e" if item == data["selection"]["selected"] else "#94a3b8" for item in active]
    bars = axes[0].bar(range(len(active)), [metrics[item]["macro_auroc"] for item in active], color=colors)
    axes[0].set_xticks(range(len(active)), [item.replace("_", "\n") for item in active])
    axes[0].set_ylim(0, 1.10)
    axes[0].set_title("Training-side synthetic selection")
    axes[0].set_ylabel("Macro image AUROC (3 proxy kinds)")
    for bar in bars:
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02, f"{bar.get_height():.4f}", ha="center")
    labels = ["Historical\n256 fit"] + [candidate.replace("_", "\n")+"\n192 fit" for candidate in data["evaluated"]]
    summaries = [data["history"]] + [data["evaluation"]["results"][candidate] for candidate in data["evaluated"]]
    x = np.arange(len(summaries))
    bars = axes[1].bar(x, [summary["image_auroc"] for summary in summaries], color=["#94a3b8"]+["#2563eb"]+["#0f766e"]*(len(summaries)-2))
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.10)
    axes[1].set_title("Previously exposed real-test retest")
    axes[1].set_ylabel("Image AUROC")
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02, f"{bar.get_height():.4f}", ha="center")
    false_negatives = [summary["threshold_metrics"]["fn"] for summary in summaries]
    false_positives = [summary["threshold_metrics"]["fp"] for summary in summaries]
    first = axes[2].bar(x-.18, false_negatives, width=.36, color="#dc2626", label="FN / 119 defects")
    second = axes[2].bar(x+.18, false_positives, width=.36, color="#d97706", label="FP / 41 normal")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, max(20, max(false_negatives+false_positives)+12))
    axes[2].set_title("Own 95% normal-calibration thresholds")
    axes[2].set_ylabel("Number of images")
    axes[2].legend(loc="upper right", fontsize=8)
    for bar in list(first)+list(second):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(int(bar.get_height())), ha="center")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    figure.suptitle("CPU screw refinement: frozen development selection, exploratory real-test comparison", fontsize=14)
    figure.text(.5, .015, "Synthetic proxies are not real defects. Historical run uses 256 fit images; new methods use the same 192-image split.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .06, 1, .94))
    path = DESTINATION / "RESULTS.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def micro_section(path):
    if path is None:
        return ["尚未提供独立 GEMM 微基准 JSON，本报告不填写加速倍数。实际检索实现见 [refine_features.py](../refine_features.py)。"]
    value = read_json(path)
    link = f"screw_refinement/run_records/resource/{Path(path).name}"
    measured = []
    if isinstance(value, dict) and isinstance(value.get("runs"), list):
        measured = ["", "| 尺寸 | 直接差分中位数ms | GEMM中位数ms | 局部评分中位数加速 | 最大距离差 |", "|---:|---:|---:|---:|---:|"]
        for run in value["runs"]:
            measured.append(f"| {run.get('size')} | {fmt(run.get('direct_median_ms'),2)} | {fmt(run.get('gemm_median_ms'),2)} | {fmt(run.get('median_speedup'),2)}倍 | {fmt(run.get('maximum_absolute_distance_error'),8)} |")
        measured += ["", "该微探针每个尺寸只有一个固定正常查询与20图参考库，预热后交替测三次；224和336都用半径3，而正式336候选用半径4。不能把这几个数字推广为整个测试集、正式336设置或其他CPU上的稳定加速倍数。"]
    # Preserve the actual record rather than inventing missing measurements.
    return [f"实际微基准原始记录见 [JSON]({link})。", "",
            *measured, "", "```json", json.dumps(portable(value), ensure_ascii=False, indent=2), "```", "",
            "这份记录只测缓存特征上的局部距离检索；不含图像解码、预处理、DINOv3前向、模型载入、网页传输或完整两尺度流程。它不能证明端到端同比加速。实现采用 float64 矩阵乘法选择最近候选，再对选中的参考做直接 float32 距离计算，以保留自匹配零距离并控制临时张量；32 MiB是目标临时预算，不是进程峰值内存。"]


def acceptance_section():
    """Summarize actual QA files without copying them into artifact manifests."""
    result = ["## 实际工程与网页验收", ""]
    qa_path = REPORTS / "SCREW_REFINEMENT_QA.json"
    if qa_path.is_file():
        qa = read_json(qa_path)
        stages = qa["stages"]
        unit = stages["unit_tests"]["result"]
        engine = stages["real_engine_and_cli"]["result"]
        http = stages["real_http"]["result"]
        frozen = stages["frozen_artifacts"]["result"]
        dependencies = stages["algorithm_dependency_sources"]["result"]
        if qa["status"] != "passed" or any(stage["status"] != "passed" for stage in stages.values()):
            raise ValueError("The recorded engineering QA is not fully passed")
        release = ROOT / "release/screw_refinement/model.pt"
        if not release.is_file() or sha(release) != frozen["model_sha256"]:
            raise ValueError("Published model SHA no longer matches the actual QA record")
        for name, expected in dependencies["source_hashes"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Engineering QA source identity changed: {name}")

        def numeric_values(value):
            if isinstance(value, dict):
                return [number for child in value.values() for number in numeric_values(child)]
            if isinstance(value, list):
                return [number for child in value for number in numeric_values(child)]
            return [float(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []

        def differences(value):
            if isinstance(value, dict):
                return [number for key, child in value.items()
                        for number in (numeric_values(child) if "difference" in key else differences(child))]
            if isinstance(value, list):
                return [number for child in value for number in differences(child)]
            return []

        metric_errors = differences(stages["independent_metrics"]["result"])
        inference_errors = differences(engine["predictions"] + engine["cli_checks"] + http["predictions"])
        if not metric_errors or not inference_errors:
            raise ValueError("Actual QA lacks recorded independent/inference difference values")
        metric_max = max(map(abs, metric_errors))
        inference_max = max(map(abs, inference_errors))
        result += [f"[实际验收记录](SCREW_REFINEMENT_QA.json) 状态为 `{qa['status']}`；没有在生成报告时重跑模型或测试。", "",
                   "| 检查 | 实际完成量或结果 |", "|---|---|",
                   f"| 单元测试 | {unit['tests']}项，进程退出码{unit['returncode']} |",
                   f"| 发布Engine实际推理 | {len(engine['predictions'])}次 |",
                   f"| 实际CLI | {len(engine['cli_checks'])}次 |",
                   f"| 实际HTTP推理 | {len(http['predictions'])}次 |",
                   f"| 独立图像指标与阈值/混淆重算 | 已记录差值最大绝对值{fmt(metric_max,8)} |",
                   f"| Engine/CLI/HTTP与离线分数及阈值 | 已记录差值最大绝对值{fmt(inference_max,8)} |",
                   f"| 冻结开发产物 | {frozen['artifact_count']}个指纹通过 |",
                   f"| 算法依赖源码 | {len(dependencies['source_hashes'])}个文件SHA256通过 |",
                   "| 实际发布模型 | SHA256与验收记录匹配 |", "",
                   f"发布模型SHA256为 `{frozen['model_sha256']}`。验收保证这些入口使用相同发布模型并与离线记录一致，不保证每张图的异常判断正确。", ""]
    else:
        result += ["尚无实际完成的工程QA JSON，不宣称单元测试或在线验收已经通过。", ""]
    browser_path = REPORTS / "SCREW_REFINEMENT_BROWSER_QA.json"
    if browser_path.is_file():
        browser = read_json(browser_path)
        if browser.get("status") != "passed":
            raise ValueError("Actual browser QA is not passed")
        result += [f"[实际浏览器验收](SCREW_REFINEMENT_BROWSER_QA.json) 使用 `{browser.get('model')}` / `{browser.get('candidate')}`，检查示例、真实文件上传、模型切换和结果展示。", "",
                   "| 实际网页样例 | 显示分数 | 显示阈值 | 网页判断 | 数据集答案 | 结果 |", "|---|---:|---:|---|---|---|"]
        failures = []
        for case in browser["cases"]:
            predicted, actual = bool(case["anomalous"]), bool(case["ground_truth_anomalous"])
            outcome = "正确" if predicted == actual else ("漏检" if actual else "误报")
            if predicted != actual:
                failures.append(case["path"])
            result.append(f"| `{case['path']}` | {case['displayed_score']} | {case['displayed_threshold']} | {'异常' if predicted else '正常'} | {'缺陷' if actual else '正常'} | **{outcome}** |")
        if failures:
            result += ["", "网页实际观察到的失败继续保留：" + "、".join(f"`{path}`" for path in dict.fromkeys(failures)) + "。浏览器验收通过表示流程与分数一致，不表示这些缺陷都检出。"]
        result += ["", "网页显示的耗时受系统负载影响，不能把这些交互观察当作受控速度比较；这些样例来自已经暴露的test，也不增加新的独立测试证据。", ""]
        screenshot = ROOT / browser.get("screenshot", "reports/SCREW_REFINEMENT_DEMO_SCREENSHOT.png")
        if screenshot.is_file():
            if browser.get("screenshot_sha256") and sha(screenshot) != browser["screenshot_sha256"]:
                raise ValueError("Browser screenshot SHA differs from the actual browser QA record")
            result += ["![实际螺丝336网页完整截图](SCREW_REFINEMENT_DEMO_SCREENSHOT.png)", ""]
    else:
        screenshot = REPORTS / "SCREW_REFINEMENT_DEMO_SCREENSHOT.png"
        if screenshot.is_file():
            result += ["现存网页截图如下；尚无浏览器QA JSON，不额外宣称验收已完成。", "",
                       "![现存螺丝网页截图](SCREW_REFINEMENT_DEMO_SCREENSHOT.png)", ""]
    return result


def build_reports(data, examples, micro_probe):
    selected = data["selection"]["selected"]
    baseline = data["evaluation"]["results"]["raw224"]
    winner = data["evaluation"]["results"][selected]
    old = data["history"]
    base_counts, win_counts = baseline["threshold_metrics"], winner["threshold_metrics"]
    delta_auc = winner["image_auroc"] - baseline["image_auroc"]
    changes = paired_changes(data["predictions"]["raw224"], data["predictions"][selected])
    winner_recall, winner_fpr = rates(win_counts)
    if selected == "raw224":
        finding = "开发侧没有候选达到预定接受门槛，因此保留新划分原始224对照作为winner；本轮不宣称复杂方法改进成功。"
    else:
        finding = (f"开发侧选中 **{DISPLAY[selected]}（`{selected}`）**。真实探索性复测相对同192张建库图对照，"
                   f"图像AUROC变化为 **{delta_auc:+.4f}**，漏检从 **{base_counts['fn']}** 变为 **{win_counts['fn']}**，"
                   f"误报从 **{base_counts['fp']}** 变为 **{win_counts['fp']}**。")
    historical_finding = (f"相对旧256张建库图的历史发布模型，漏检从 **{old['threshold_metrics']['fn']}** 变为 **{win_counts['fn']}**，"
                          f"误报从 **{old['threshold_metrics']['fp']}** 变为 **{win_counts['fp']}**。这是不同建库数量的历史比较，不能替代同192图对照。")
    lines = ["# 螺丝漏检改进：实际实验报告", "", finding, "",
             historical_finding, "",
             "本轮结果固定一个建库种子42，不是三个种子的均值；没有测量本轮多种子稳定性。实际winner是原始336分支，姿态对齐和多尺度仅作为未选中的开发候选，不应把它们描述为发布模型的组成。", "",
             "这是一轮CPU上的冻结特征实验，没有从零训练或微调DINOv3，也没有使用Ollama。官方螺丝test此前已全部查看，本轮真实结果均是**探索性复测**。合成审计只提供训练侧代理缺陷证据，不能称为未见真实工业缺陷验证。", "",
             "![实际开发选择与真实复测](screw_refinement/RESULTS.png)", "",
             "## 方案与数据边界", "",
             "旧256张正常建库图被重新分为192张fit、48张selection正常源图、16张audit正常源图；保留旧64张正常校准图。四组源图互斥。三个代理缺陷在原始分辨率生成，每个选择源图各三种，共144张；审计有16正常源图及48代理缺陷。所有候选使用冻结DINOv3 ViT-S/16，骨干与特征在CPU上为float32，单线程、固定种子42的逐位置k-center库；检索候选计算使用float64。", "",
             "| 候选 | 处理与检索 | 分数 |", "|---|---|---|",
             "| 原始224 | 14×14网格，半径3 | 最大patch距离 / 48开发正常图中位数 |",
             "| 原始336 | 21×21网格，半径4 | 最大patch距离 / 48开发正常图中位数 |",
             "| 姿态对齐224 | 前景主轴旋转居中，无crop和额外缩放 | 质量通过使用对齐库，否则使用原始224库和对应中位数 |",
             "| 224＋336 | 同一原图坐标，两尺度各自正常化后插值到224 | 两幅图等权平均，取融合图最大值 |", "",
             "336/半径4是相对空间范围的近似取整，与224/半径3的物理范围并不完全相同。融合插值可能平滑峰值，图像分数不是两个最大值的平均。对齐库只采用质量通过的fit图，因此与原始库相比还存在可用训练图数差别。低质量图绝不匹配对齐库。", "",
             "完整预先协议见 [冻结协议副本](screw_refinement/run_records/SCREW_REFINEMENT_PROTOCOL.md)，配置与源码身份见 [protocol.json](screw_refinement/run_records/protocol.json) 和 [source_identity.json](screw_refinement/run_records/source_identity.json)。", "",
             "## 开发侧选择：全部候选", "",
             "三种代理缺陷分别与同48张正常图计算AUROC，取三者平均。复杂方案至少比原始224高0.005才接受；精确并列按原始224、对齐224、原始336、多尺度顺序。正常尺度统计在代理评分前冻结；正常校准集和审计集均不参与选择。", "",
             "| 候选 | 划痕AUROC | 暗斑AUROC | 位移AUROC | 宏平均 | 是否选中 |", "|---|---:|---:|---:|---:|---|"]
    for candidate in ORDER:
        metric = data["selection"]["metrics"].get(candidate)
        if metric is None or candidate not in data["active"]:
            reason = (metric or {}).get("reason", data["selection"].get("disabled_candidates", {}).get(
                candidate, "未运行/已禁用，见冻结记录"))
            lines.append(f"| {DISPLAY[candidate]} | — | — | — | — | {reason} |")
        else:
            by_kind = metric["by_kind_auroc"]
            lines.append(f"| {DISPLAY[candidate]} | {fmt(by_kind['scratch'])} | {fmt(by_kind['pit'])} | {fmt(by_kind['texture_shift'])} | {fmt(metric['macro_auroc'])} | {'是' if candidate == selected else '否'} |")
    lines += ["", f"开发侧最佳代理候选是 `{data['selection']['best_proxy_candidate']}`，最终冻结winner为 `{selected}`。两者可能因0.005接受门槛不同。候选没有根据官方test重新选择。", "",
              "## 合成审计与生成样例", "",
              "选定后，在16个不同正常源图及48代理缺陷上报告审计。审计不改变winner。代理缺陷共享源图，不能把48或144张代理图当成同样数量的独立真实缺陷。", "",
              "| 审计候选 | 划痕AUROC | 暗斑AUROC | 位移AUROC | 宏平均 |", "|---|---:|---:|---:|---:|"]
    for candidate in data["evaluated"]:
        metric = data["complete"]["audit"][candidate]
        by_kind = metric["by_kind_auroc"]
        lines.append(f"| {DISPLAY[candidate]} | {fmt(by_kind['scratch'])} | {fmt(by_kind['pit'])} | {fmt(by_kind['texture_shift'])} | {fmt(metric['macro_auroc'])} |")
    for kind in KINDS:
        lines += ["", f"{KIND_DISPLAY[kind]}示例：左为正常源图，中为生成操作mask，右为合成图。固定取清单中的第一张该类选择样例，不按检测成功挑图。", "",
                  f"![{KIND_DISPLAY[kind]}合成示例](screw_refinement/examples/synthetic/{examples[kind].name})"]
    lines += ["", "生成位置仅来自正常训练图的前景估计，参数、种子、mask和文件哈希见 [synthetic_manifest.json](screw_refinement/run_records/synthetic_manifest.json)。细划痕、暗斑、位移只是代理异常，并没有证明其统计性质等同于真实工业缺陷。", "",
              "## 相同真实图片上的探索性复测", "",
              "每个新方法在同64张正常校准图上计算自己的95%分位阈值。分数正常化不是概率，不能直接跨模型比较原始分数高低。阈值没有根据test调整，经验95%分位数也不保证未来5%误报。", "",
              "| 模型 | 建库图 | 图像AUROC | TP | FN/119 | FP/41 | TN | 缺陷召回 | 正常误报率 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    results = [("旧发布模型（历史）", 256, old)] + [(DISPLAY[candidate], 192, data["evaluation"]["results"][candidate]) for candidate in data["evaluated"]]
    for label, count, summary in results:
        metric = summary["threshold_metrics"]
        recall, fpr = rates(metric)
        lines.append(f"| {label} | {count} | {fmt(summary['image_auroc'])} | {metric['tp']} | {metric['fn']} | {metric['fp']} | {metric['tn']} | {recall:.2%} | {fpr:.2%} |")
    lines += ["", "主要方法比较是新划分原始224与winner：两者使用相同192张建库图及64张校准图。旧模型使用256张建库图，只作历史参考，不能把与旧模型的所有差值都归因于本轮算法组件。所有真实test图片和标签完全相同；它们此前已经暴露，所以不证明对新产品的泛化。", "",
              "| 新方法 | 自身阈值 | 224共同有效区域像素AUROC | 有效像素比例 |", "|---|---:|---:|---:|"]
    for candidate in data["evaluated"]:
        summary = data["evaluation"]["results"][candidate]
        lines.append(f"| {DISPLAY[candidate]} | {fmt(summary['threshold'],6)} | {fmt(summary.get('pixel_auroc_common_valid_224'))} | {fmt(summary.get('pixel_valid_fraction'))} |")
    lines += ["", "定位指标在原图的224×224框架上计算。若winner使用对齐，逆变换热图无法覆盖的像素同时从对照与winner中排除，避免只处罚某一方法；因此这不是官方原分辨率指标，也不能直接与旧报告未限制有效区域的像素AUROC作公平比较。热图颜色逐图缩放，仅表示相对可疑位置。", "",
              "## 每类失败与逐图变化", "",
              "| 缺陷/正常类型 | 数量 | 旧模型漏检或误报 | 新原始224 | winner |", "|---|---:|---:|---:|---:|"]
    kinds_old = per_kind(data["history_predictions"])
    kinds_baseline = per_kind(data["predictions"]["raw224"])
    kinds_winner = per_kind(data["predictions"][selected])
    for kind in sorted(kinds_baseline):
        key = "fp" if kind == "good" else "fn"
        lines.append(f"| {kind}（{'误报' if kind == 'good' else '漏检'}） | {kinds_baseline[kind]['n']} | {kinds_old[kind][key]} | {kinds_baseline[kind][key]} | {kinds_winner[kind][key]} |")
    historic_regressions = [f"`{kind}`漏检{count['fn']}→{kinds_winner[kind]['fn']}"
                           for kind, count in kinds_old.items()
                           if kind != "good" and kinds_winner[kind]["fn"] > count["fn"]]
    if historic_regressions:
        lines += ["", "与旧历史模型相比仍有局部退步：" + "；".join(historic_regressions) + "。整体漏检下降不代表每种缺陷都改善。"]
    lines += ["", f"按同一张图配对：winner相对新对照有 **{len(changes[0])}** 张从错误变正确，**{len(changes[1])}** 张从正确变错误。完整路径见 [paired_changes.json](screw_refinement/run_records/paired_changes.json)。这些是已经暴露数据上的描述统计，不应包装成独立确认性实验。", ""]
    if win_counts["fn"] >= base_counts["fn"]:
        lines.append("本轮winner没有减少真实缺陷漏检。即使某个AUROC或合成指标提高，也不能声称已改善这项部署目标。")
    if win_counts["fp"] > base_counts["fp"]:
        lines.append("本轮winner的正常误报增加，检出改善伴随误报代价，不能只展示漏检下降。")
    if delta_auc < 0:
        lines.append("本轮winner真实图像排序能力下降，合成选择没有迁移为真实AUROC提升。")
    if win_counts["fn"] > 0:
        lines.append(f"winner仍漏检 {win_counts['fn']}/119 个真实缺陷，召回率 {winner_recall:.2%}，不具备“缺陷均已找出”的证据。")
    lines.append(f"winner在本次41张正常test上的误报率为 {winner_fpr:.2%}。真实现场拍摄变化、未知产品和速度要求尚需新数据验证。")
    for candidate in data["evaluated"]:
        examples_dir = DESTINATION / "examples" / candidate
        failed = sorted(examples_dir.glob("error-*.png"))
        false_positive = next((example for example in failed if example.name.startswith("error-good-")), None)
        false_negative = next((example for example in failed if not example.name.startswith("error-good-")), None)
        shown = [(false_positive, "正常产品误报"), (false_negative, "真实缺陷漏检")]
        for example, caption in shown:
            if example is None:
                continue
            lines += ["", f"{DISPLAY[candidate]}固定顺序保存的{caption}案例：", "",
                      f"![{candidate}失败例图](screw_refinement/examples/{candidate}/{example.name})"]
        if not failed:
            lines += ["", f"`{candidate}` 本次没有保存失败PNG；是否存在失败以逐图CSV及混淆矩阵为准，不能据此推定零失败。"]
    lines += ["", "## 对齐可靠性与回退覆盖", "",
              "| 数据角色 | 数量 | 对齐质量通过 | 原始分支回退 |", "|---|---:|---:|---:|"]
    for role, total, passed, failed in fallback_table(data):
        lines.append(f"| {role} | {total} | {passed if passed is not None else '未执行/未记录'} | {failed if failed is not None else '未执行/未记录'} |")
    fit_sources = data["complete"].get("fit_sources", {})
    if "aligned224" in fit_sources:
        lines += ["", f"实际对齐库采用 **{len(fit_sources['aligned224'])}/192** 张质量通过fit图。回退图使用raw224库和raw224中位数，对齐质量检查衡量几何稳定性，不等于异常判断正确。"]
    lines += ["", "## CPU资源与计时口径", "",
              "资源探针只采用固定8张fit正常图，不看缺陷结果。实际记录见 [resource_probe.json](screw_refinement/run_records/resource_probe.json)。"]
    probe_path = OUTPUT / "resource_probe.json"
    if probe_path.is_file():
        probe = read_json(probe_path)
        values = probe if isinstance(probe, list) else probe.get("records", [])
        lines += ["", "| 尺寸 | 正常图数 | 前向中位数ms | patch形状 |", "|---:|---:|---:|---|"]
        for record in values:
            count = len(record.get("normal_images", []))
            lines.append(f"| {record.get('size')} | {count} | {fmt(record.get('median_ms'),2)} | `{record.get('patch_shape')}` |")
    lines += ["", "| 新方法 | 实际所需库MiB | 记录的缓存评分中位数ms |", "|---|---:|---:|"]
    for candidate in data["evaluated"]:
        result = data["evaluation"]["results"][candidate]
        library = result.get("bank_mib", {})
        total = sum(library.values()) if isinstance(library, dict) else library
        lines.append(f"| {DISPLAY[candidate]} | {fmt(total,2)} | {fmt(result.get('scoring_median_ms'),2)} |")
    lines += ["", "评分计时来自已有特征，且单尺度与融合可能复用先前计算的距离图；它不含骨干前向，不是候选之间公平的完整推理测速。资源探针前向也不含完整在线流程，顺序执行与后台负载可能使尺寸耗时出现反常顺序，不能据此说高分辨率一定更快。最终端到端速度只能引用独立实际CLI/HTTP测量，不把以上数值相加伪造整图耗时。库字节数不包含骨干、缓存、运行时和临时张量，不能称为峰值内存。", ""]
    timing = data["complete"].get("feature_timings", data["complete"].get("feature_metadata"))
    if timing is not None:
        lines += ["实际特征提取/缓存记录：", "", "```json", json.dumps(portable(timing), ensure_ascii=False, indent=2), "```", ""]
    lines += micro_section(micro_probe)
    lines += ["", "## 证据核对与复现", "",
              "报告程序使用另一种正负样本逐对比较算法重新计算开发、审计和真实图像AUROC，并检查阈值来自同64张正常校准图、预测与阈值一致、混淆矩阵一致、test路径一致、源图划分互斥以及冻结artifact哈希。它没有重新运行骨干，也没有重新选择winner。像素AUROC引用runner的真实记录，不谎称报告程序独立重算了像素分数。", "",
              "可复现命令：", "", "```powershell", 
              ".\\.venv\\Scripts\\python.exe screw_refine.py --stage develop",
              ".\\.venv\\Scripts\\python.exe screw_refine.py --stage evaluate",
              ".\\.venv\\Scripts\\python.exe screw_refine_report.py", "```", "",
              "已有完整开发产物时，runner先校验身份再恢复；改动冻结参数或源码需要新实验版本，不能继续使用旧完成标记。报告程序缺真实完成JSON时直接失败，不创建占位成绩。", "",
              "可携带证据在 [run_records](screw_refinement/run_records/)。[export_manifest.json](screw_refinement/run_records/export_manifest.json) 分别记录原始文件与可携带副本的SHA256；JSON本机路径前缀被替换为`.`，因此副本哈希可能与原文件不同。保留模型文件的哈希记录，但不复制模型、预训练权重、特征缓存或原始数据集。", "",
              "## 来源与许可证", "",
              "检测方法沿用 [DINOSaur官方实现](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD) 的核心思路，骨干使用 [timm DINOv3 ViT-S/16模型](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)。具体权重及源码SHA见冻结记录；本轮高分辨率、对齐、融合、CPU检索优化和验证属于项目适配，不宣称原创DINOv3或DINOSaur。", "",
              "原始与合成面板、缺陷mask、失败热图均派生自 [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，按数据集的 **CC BY-NC-SA 4.0** 条件使用和分享，限非商业用途并保留署名及相同许可。项目原创代码按根目录MIT许可；DINOv3预训练权重有独立自定义许可，不能因为项目MIT就视为MIT权重。详见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。"]
    if data.get("dependency_audit") is not None:
        position = lines.index("可复现命令：")
        lines[position:position] = ["[独立源码依赖审核](screw_refinement/run_records/source_dependency_audit.json) 核对六个实际算法文件的SHA256。前三个已有冻结protocol指纹，`frontier_core.py`、`frontier.py`、`metrics.py`的额外指纹来自补充审计，没有被倒填进原冻结协议。", ""]
    position = lines.index("## 证据核对与复现")
    lines[position:position] = acceptance_section()
    (REPORTS / "SCREW_REFINEMENT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(DESTINATION / "run_records/paired_changes.json", dict(baseline="raw224", selected=selected,
                wrong_to_correct=changes[0], correct_to_wrong=changes[1],
                note="Previously exposed real test; descriptive paired changes, not independent confirmatory evidence."))
    interview = ["# 螺丝改进：面试解释与真实边界", "",
                 "## 为什么继续做", "",
                 f"旧DINOv3发布模型在螺丝真实test上漏检{old['threshold_metrics']['fn']}/119、误报{old['threshold_metrics']['fp']}/41。AUROC较旧ResNet方案提高，并没有解决固定阈值漏检。因此本轮围绕细小缺陷分辨率和姿态变化做受控候选比较。", "",
                 "## 我具体怎么做", "",
                 "在CPU上保持DINOv3冻结，重新划出192张正常建库图、48张候选选择图、16张审计图，保留64张正常阈值校准图。只用选择正常图生成原分辨率划痕、暗斑、纹理位移。比较原始224、原始336、姿态对齐224与两尺度融合，按预先固定的代理macro-AUROC和0.005接受门槛选择。最后才看审计与既有真实test。", "",
                 "正常库是每空间位置的代表特征集合，不是重新训练一个大模型。对齐只旋转居中，质量不稳定时回退原始库；多尺度在同一原图坐标上融合正常化距离图。CPU检索通过矩阵乘法确定候选，再直接计算最终距离，降低局部评分开销；局部微基准不等于完整推理速度。", "",
                 "## 实际结果怎么讲", "", finding, "", historical_finding, "",
                 "本轮只固定种子42，不能写成三个种子的平均；最终螺丝方案是原始336，对齐与多尺度没有被选入发布流程。", "",
                 f"winner真实test图像AUROC为{fmt(winner['image_auroc'])}，漏检{win_counts['fn']}/119、误报{win_counts['fp']}/41，召回率{winner_recall:.2%}。这些值来自固定正常校准阈值，AUROC不能当作准确率或召回率。历史256图库与本轮192图库不同，主要比较应使用本轮同划分原始224对照。", "",
                 "## 遇到什么问题与解决", "",
                 "- 小缺陷可能被缩放削弱：预先比较336，并把合成缺陷先画在原图，避免尺寸偏置。",
                 "- 产品姿态变化影响位置匹配：加入像素几何对齐与明确质量回退；对齐不可靠时使用原始库。",
                 "- 两尺度分数不同：仅用开发正常图统计尺度，再固定融合，校准图不参与选参。",
                 "- 距离计算慢且矩阵公式可能相消：使用float64候选搜索与直接最终距离，并用独立参考核对。",
                 "- 实验中断可能混用文件：冻结分组、配置、源码与关键产物哈希，完成标记校验后再恢复。", "",
                 "## 哪些话不能说", "",
                 "不能说从零训练了DINOv3、自己发明DINOSaur、合成缺陷证明真实工业能力、既有test是未见数据、微基准加速等于端到端加速，或模型已经满足生产质检。项目由AI助手协助实现，个人贡献按自己真正复跑、理解和修改的部分描述。", "",
                 f"winner仍有{win_counts['fn']}个漏检和{win_counts['fp']}个误报；未知产品拒绝、光照背景变化、新现场数据及稳定速度尚未验证。完整失败子类与逐图变化见 [实际报告](SCREW_REFINEMENT.md)。", "",
                 "## 能解释的关键问题", "",
                 "为什么分开选择与校准：选择比较方案，校准决定报警界限，混用会让评估偏乐观。为什么保留失败：漏检和误报决定实际可用性，高AUROC不能掩盖缺陷放行。为什么还要新数据：现有test已经帮助我们确定研究方向，复测不提供真正独立确认。"]
    (REPORTS / "SCREW_REFINEMENT_INTERVIEW.md").write_text("\n".join(interview)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-probe", type=Path, help="Actual optional GEMM benchmark JSON")
    parser.add_argument("--micro-source", type=Path, help="Optional corresponding microbenchmark source")
    args = parser.parse_args()
    default_probe = DESTINATION / "RETRIEVAL_BENCHMARK.json"
    if args.micro_probe is None and default_probe.is_file():
        args.micro_probe = default_probe
    try:
        data = load_evidence()
        if args.micro_probe is not None:
            read_json(args.micro_probe)
        if args.micro_source is not None and not args.micro_source.is_file():
            raise FileNotFoundError(args.micro_source)
        exported = export_evidence(data, args.micro_probe, args.micro_source)
        examples = synthetic_examples(data)
        figure = make_figure(data)
        build_reports(data, examples, args.micro_probe)
        print(json.dumps(dict(selected=data["selection"]["selected"],
                              independently_checked_image_metrics=True,
                              exported_files=len(exported), figure=relative(figure),
                              reports=["reports/SCREW_REFINEMENT.md", "reports/SCREW_REFINEMENT_INTERVIEW.md"]),
                         ensure_ascii=False, indent=2))
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"报告未生成：{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
