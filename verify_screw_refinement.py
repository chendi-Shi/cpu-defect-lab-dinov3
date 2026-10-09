"""Audit frozen screw refinement, independent metrics, real CLI and local HTTP.

Run only after development, exploratory evaluation and release are complete.
The temporary HTTP server uses an OS-assigned port, leaving port 18765 alone.
Errors are saved in reports/SCREW_REFINEMENT_QA.json, including partial results.
"""

from __future__ import annotations

import base64
import csv
import datetime
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/screw_refinement"
RELEASE = ROOT / "release/screw_refinement"
REPORT = ROOT / "reports/SCREW_REFINEMENT_QA.json"
CANDIDATES = ("raw224", "aligned224", "raw336", "multi224_336")
KINDS = ("scratch", "pit", "texture_shift")
SCORE_TOLERANCE = 2e-6
DEPENDENCY_FINGERPRINTS = {
    "screw_refine.py": "1226eee5064edb5a0f99b51af46c3a585ac714646537ccf22903254225c5c364",
    "screw_geometry.py": "a411cc6883f47ce0e5a1218184dca955bf19a72f8a8b17a8b52cc4315f167a60",
    "refine_features.py": "c36ea7dd1eb770b524cf9b5b90d35e12b8de8fda536c3a1260b86269aee864d2",
    "frontier_core.py": "921a1e9bf7b5ed394b747730bb333f7066a6d637dd70bfeacd9b39d06e2d63a4",
    "frontier.py": "703fe91eabb2c94b347a23102213f68bdaefa884dd13297da1dd8616d11ed3de",
    "metrics.py": "cbb43acab190bece5743a27277c823b638f2f703071eb3f3540ed0e049f13694",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def pairwise_auroc(rows):
    """Independent pair-counting oracle, with half credit for exact ties."""
    positive, negative = [], []
    for row in rows:
        label, score = int(row["label"]), float(row["score"])
        if label not in (0, 1) or not math.isfinite(score):
            raise ValueError("Metrics require binary labels and finite scores")
        (positive if label else negative).append(score)
    if not positive or not negative:
        return None
    wins = sum(1. if p > n else .5 if p == n else 0. for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def confusion(rows, threshold):
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in rows:
        label, prediction = int(row["label"]), float(row["score"]) > threshold
        key = ("tp" if prediction else "fn") if label else ("fp" if prediction else "tn")
        counts[key] += 1
        if "prediction" in row and int(row["prediction"]) != int(prediction):
            raise AssertionError(f"Saved prediction disagrees with score/threshold: {row.get('path')}")
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    counts.update(precision=tp/(tp+fp) if tp+fp else None,
                  recall=tp/(tp+fn) if tp+fn else None,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    return counts


def compare_number(actual, expected, label, tolerance=1e-12):
    if actual is None or expected is None:
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return 0.
    difference = abs(float(actual) - float(expected))
    if not math.isfinite(difference) or difference > tolerance:
        raise AssertionError(f"{label}: {actual!r} != {expected!r}; difference={difference}")
    return difference


def proxy_audit(rows, expected, label):
    by_kind, differences = {}, {}
    for kind in KINDS:
        value = pairwise_auroc([row for row in rows if row["kind"] in ("good", kind)])
        if value is None:
            raise AssertionError(f"Missing normal/proxy pairs for {label}/{kind}")
        by_kind[kind] = value
        differences[kind] = compare_number(value, expected["by_kind_auroc"][kind], f"{label}/{kind}")
    macro = sum(by_kind.values()) / len(KINDS)
    pooled = pairwise_auroc(rows)
    differences["macro"] = compare_number(macro, expected["macro_auroc"], f"{label}/macro")
    differences["pooled"] = compare_number(pooled, expected["pooled_auroc"], f"{label}/pooled")
    return {"by_kind_auroc": by_kind, "macro_auroc": macro, "pooled_auroc": pooled,
            "differences": differences, "rows": len(rows)}


def linear_quantile(values, quantile):
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("Cannot compute quantile of no scores")
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stage_units():
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = result.stdout + result.stderr
    print(output, flush=True)
    match = re.search(r"Ran (\d+) tests", output)
    record = {"returncode": result.returncode, "tests": int(match.group(1)) if match else None,
              "python": sys.executable, "output": output}
    if result.returncode:
        raise AssertionError("Unit discovery failed\n" + output)
    return record


def stage_frozen():
    import torch
    from refine_features import file_sha256
    from screw_refine import validate_development

    complete = validate_development()
    selection = read_json(OUT / "selection.json")
    if complete["selection"] != selection:
        raise AssertionError("Frozen development selection mismatch")
    if selection.get("test_used_for_selection") or selection.get("audit_used_for_selection"):
        raise AssertionError("Selection claims use of audit/test images")
    manifest = read_json(RELEASE / "manifest.json")
    model_path = RELEASE / "model.pt"
    expected_path = (ROOT / manifest["model"]).resolve()
    if expected_path != model_path.resolve():
        raise AssertionError("Released model path differs from manifest")
    actual_sha = file_sha256(model_path)
    if manifest["model_sha256"] != actual_sha:
        raise AssertionError("Release model SHA mismatch")
    if manifest["selected_candidate"] != selection["selected"]:
        raise AssertionError("Released model is not the frozen selected candidate")
    original = OUT / selection["selected"] / "model.pt"
    if file_sha256(original) != actual_sha:
        raise AssertionError("Released tensors differ from frozen output model")
    saved = torch.load(model_path, map_location="cpu", weights_only=True)
    from screw_refine_predict import validate_model
    validate_model(saved)
    if saved["config"]["selection_sha256"] != file_sha256(OUT / "selection.json"):
        raise AssertionError("Model selection SHA mismatch")
    groups = read_json(OUT / "splits.json")["groups"]
    sets = [set(entry["path"] for entry in values) for values in groups.values()]
    if sum(map(len, sets)) != len(set.union(*sets)) or len(set.union(*sets)) != 320:
        raise AssertionError("Training partitions overlap or fail to cover 320 images")
    if any("/test/" in name for values in sets for name in values):
        raise AssertionError("Real test image appears in a development partition")
    return {"selected": selection["selected"], "artifact_count": len(complete["artifacts"]),
            "model_sha256": actual_sha, "selection_sha256": file_sha256(OUT / "selection.json"),
            "training_partition_counts": {name: len(values) for name, values in groups.items()},
            "test_used_for_selection": False, "audit_used_for_selection": False}


def stage_dependency_sources():
    """Check three frozen protocol files plus three supplementary dependencies.

    Expected bytes are pinned from the separate read-only source audit, which
    also confirmed no change since its preceding audit. The additional files
    are not retroactively described as fields of the original protocol.
    """
    protocol = read_json(OUT / "protocol.json")
    fields = {"screw_refine.py": "proxy_generator_source_sha256",
              "screw_geometry.py": "geometry_source_sha256",
              "refine_features.py": "runtime_source_sha256"}
    checked = []
    for filename, expected in DEPENDENCY_FINGERPRINTS.items():
        actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Independent dependency fingerprint changed: {filename}")
        field = fields.get(filename)
        if field is not None and protocol[field] != actual:
            raise AssertionError(f"Dependency differs from frozen protocol: {filename}")
        checked.append({"path": filename, "sha256": actual, "expected_sha256": expected,
                        "match": True, "protocol_hash_key": field})
    return {"source_files": checked, "source_hashes": {item["path"]: item["sha256"] for item in checked},
            "scope": "Three files are explicit frozen protocol fields; three are supplementary independently pinned dependency fingerprints.",
            "audit_recorded_at_utc": "2026-10-09T07:40:27.054787+00:00"}


def stage_metrics():
    selection = read_json(OUT / "selection.json")
    complete = read_json(OUT / "development_complete.json")
    evaluation = read_json(OUT / "evaluation.json")
    manifest = read_json(RELEASE / "manifest.json")
    selected = selection["selected"]
    result = {"development": {}, "audit": {}, "test": {}, "calibration": {}}
    for candidate in CANDIDATES:
        expected = selection["metrics"][candidate]
        if expected.get("macro_auroc") is None:
            disabled = read_json(OUT / candidate / "disabled.json")
            if disabled != expected:
                raise AssertionError("Disabled candidate record differs from selection")
            result["development"][candidate] = {"status": "disabled", "reason": expected.get("disabled")}
            continue
        rows = read_rows(OUT / candidate / "development_predictions.csv")
        result["development"][candidate] = proxy_audit(rows, expected, f"development/{candidate}")
    enabled = [name for name in CANDIDATES if "macro_auroc" in result["development"][name]]
    highest = max(result["development"][name]["macro_auroc"] for name in enabled)
    best = next(name for name in enabled if highest-result["development"][name]["macro_auroc"] <= 1e-9)
    baseline_macro = result["development"]["raw224"]["macro_auroc"]
    oracle_selected = best if result["development"][best]["macro_auroc"] >= baseline_macro + selection["minimum_gain"] else "raw224"
    if best != selection["best_proxy_candidate"] or oracle_selected != selected:
        raise AssertionError("Independent candidate selection disagrees")
    result["selection_recomputed"] = oracle_selected
    for candidate in dict.fromkeys(("raw224", selected)):
        rows = read_rows(OUT / candidate / "audit_predictions.csv")
        result["audit"][candidate] = proxy_audit(rows, complete["audit"][candidate], f"audit/{candidate}")
        calibration = read_rows(OUT / candidate / "calibration_predictions.csv")
        threshold = linear_quantile([row["score"] for row in calibration], .95)
        result["calibration"][candidate] = {
            "rows": len(calibration), "threshold": threshold,
            "difference": compare_number(threshold, complete["calibration"][candidate]["threshold"], f"calibration/{candidate}")}
        test = read_rows(OUT / candidate / "test_predictions.csv")
        summary = read_json(OUT / candidate / "test_summary.json")
        auc = pairwise_auroc(test)
        counts = confusion(test, summary["threshold"])
        differences = {"image_auroc": compare_number(auc, summary["image_auroc"], f"test/{candidate}/auc"),
                       "threshold": compare_number(threshold, summary["threshold"], f"test/{candidate}/threshold")}
        for key, value in counts.items():
            differences[key] = compare_number(value, summary["threshold_metrics"][key], f"test/{candidate}/{key}")
        for kind in sorted({row["kind"] for row in test}):
            per_kind = confusion([row for row in test if row["kind"] == kind], summary["threshold"])
            for key, value in per_kind.items():
                compare_number(value, summary["by_kind"][kind][key], f"test/{candidate}/{kind}/{key}")
        if evaluation["results"][candidate] != summary:
            raise AssertionError("Evaluation summary differs from per-candidate record")
        if len(test) != 160:
            raise AssertionError("Incomplete real test predictions")
        result["test"][candidate] = {"rows": len(test), "image_auroc": auc,
                                     "threshold_metrics": counts, "differences": differences}
    chosen = result["test"][selected]
    compare_number(chosen["image_auroc"], manifest["test_metrics"]["image_auroc"], "manifest/auc")
    for key in ("tp", "fp", "fn", "tn"):
        compare_number(chosen["threshold_metrics"][key], manifest["test_metrics"][key], f"manifest/{key}")
    result["pixel_note"] = "Pixel AUROC is not independently recomputed here; this audit independently checks image AUROC and threshold outcomes."
    return result


def selected_examples():
    selected = read_json(OUT / "selection.json")["selected"]
    rows = read_rows(OUT / selected / "test_predictions.csv")
    choices = ["test/good/000.png", "test/good/001.png", "test/scratch_head/000.png", "test/thread_side/000.png"]
    if selected == "aligned224":
        fallback = next((row["path"] for row in rows if row["branch"] == "raw224"), None)
        if fallback is not None and fallback not in choices:
            choices.append(fallback)
    return selected, {row["path"]: row for row in rows}, choices


def compare_prediction(prediction, reference, threshold, label):
    score_diff = compare_number(prediction["score"], float(reference["score"]), label + "/score", SCORE_TOLERANCE)
    threshold_diff = compare_number(prediction["threshold"], threshold, label + "/threshold", SCORE_TOLERANCE)
    if bool(prediction["anomalous"]) != bool(int(reference["prediction"])):
        raise AssertionError(f"{label}: decision differs from offline prediction")
    return {"score_difference": score_diff, "threshold_difference": threshold_diff}


def stage_inference():
    import numpy as np
    from PIL import Image
    from refine_features import file_sha256
    from screw_refine_predict import RefinedScrewEngine

    selected, references, paths = selected_examples()
    threshold = float(read_json(OUT / selected / "test_summary.json")["threshold"])
    before = time.perf_counter()
    engine = RefinedScrewEngine(RELEASE / "model.pt")
    cold_ms = (time.perf_counter() - before) * 1000
    predictions, cli_checks = [], []
    for name in paths:
        source = ROOT / "data/mvtec/screw" / name
        with Image.open(source) as original:
            image = original.convert("RGB")
        for repetition in (1, 2):
            result = engine.predict(image)
            differences = compare_prediction(result, references[name], threshold, f"engine/{name}/{repetition}")
            if result["branch"] != references[name]["branch"] or result["candidate"] != selected:
                raise AssertionError("Live engine branch differs from offline result")
            if result["heatmap"].shape != (224, 224) or not np.isfinite(result["heatmap"]).all():
                raise AssertionError("Invalid live heatmap")
            predictions.append({"path": name, "repetition": repetition, "score": result["score"],
                                "threshold": result["threshold"], "anomalous": result["anomalous"],
                                "candidate": result["candidate"], "branch": result["branch"], **differences,
                                **{key: result[key] for key in ("preprocessing_ms", "forward_ms", "scoring_ms", "total_inference_ms")},
                                "geometry_diagnostics": result["geometry_diagnostics"]})
        directory = ROOT / "reports/screw_refinement/cli_qa" / (source.parent.name + "-" + source.stem)
        cli = subprocess.run([sys.executable, str(ROOT / "screw_refine_predict.py"), "--image", str(source),
                              "--model-path", str(RELEASE / "model.pt"), "--output-dir", str(directory)],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if cli.returncode:
            raise AssertionError("Actual CLI failed: " + cli.stdout + cli.stderr)
        saved = read_json(directory / "prediction.json")
        differences = compare_prediction(saved, references[name], threshold, f"cli/{name}")
        with Image.open(directory / "original.png") as delivered:
            expected = image.resize((224, 224), Image.Resampling.BILINEAR)
            if delivered.convert("RGB").tobytes() != expected.tobytes():
                raise AssertionError("CLI original pixels differ from input resize")
        for kind in ("heatmap", "overlay"):
            with Image.open(directory / f"{kind}.png") as delivered:
                if delivered.size != (224, 224):
                    raise AssertionError("Invalid CLI image dimensions")
        cli_checks.append({"path": name, "returncode": cli.returncode, "score": saved["score"],
                           "anomalous": saved["anomalous"], "branch": saved["branch"], **differences,
                           "output_directory": directory.relative_to(ROOT).as_posix(),
                           "source_sha256": file_sha256(source), "original_pixels_match": True})
    return {"candidate": selected, "cold_model_load_ms": cold_ms, "predictions": predictions,
            "cli_checks": cli_checks, "repeat_count_per_image": 2,
            "timing_note": "Actual resident engine timings; cold model loading listed separately. No fair historical speed comparison is claimed."}


def stage_http():
    from PIL import Image
    from demo_server import Engine, handler_for

    selected, references, _ = selected_examples()
    threshold = float(read_json(OUT / selected / "test_summary.json")["threshold"])
    before = time.perf_counter()
    engine = Engine(ROOT / "release")
    cold_ms = (time.perf_counter() - before) * 1000
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    predictions, rejection_checks, example_checks = [], [], []

    def request(path, payload=None, headers=None, expected=200):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(base + path, body, headers=headers or {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                status, data = response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            status, data = exc.code, json.load(exc)
        if status != expected:
            raise AssertionError(f"HTTP {path}: status {status}, expected {expected}: {data!r}")
        return data

    try:
        catalog = request("/api/models")
        ids = {entry["id"] for entry in catalog}
        if not {"screw-refined", "dinov3-screw", "dinov3-auto"}.issubset(ids):
            raise AssertionError("Required new/old HTTP modes are missing")
        for kind in ("normal", "defect"):
            example = request(f"/api/example?model=screw-refined&type={kind}")
            raw = base64.b64decode(example["image"].split(",", 1)[1], validate=True)
            candidates = sorted((ROOT / "data/mvtec/screw/test").rglob("*.png"))
            expected_path = next(path for path in candidates if (path.parent.name == "good") == (kind == "normal"))
            with Image.open(io.BytesIO(raw)) as served, Image.open(expected_path) as original:
                if served.size != original.size or served.convert("RGB").tobytes() != original.convert("RGB").tobytes():
                    raise AssertionError("HTTP example original pixels differ from the dataset")
            example_checks.append({"kind": kind, "path": expected_path.relative_to(ROOT).as_posix(), "raw_pixels_match": True})
        for name in ("test/good/000.png", "test/good/001.png", "test/scratch_head/000.png", "test/thread_side/000.png"):
            source = ROOT / "data/mvtec/screw" / name
            encoded = base64.b64encode(source.read_bytes()).decode("ascii")
            result = request("/api/predict", {"model": "screw-refined", "image": encoded})
            differences = compare_prediction(result, references[name], threshold, "HTTP/" + name)
            if result.get("candidate") != selected or result.get("selected_category") != "screw":
                raise AssertionError("New HTTP mode candidate/category mismatch")
            with Image.open(io.BytesIO(base64.b64decode(result["original"].split(",", 1)[1]))) as delivered, Image.open(source) as original:
                expected = original.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                if delivered.convert("RGB").tobytes() != expected.tobytes():
                    raise AssertionError("HTTP returned original pixels differ")
            predictions.append({"model": "screw-refined", "path": name, "score": result["score"],
                                "threshold": result["threshold"], "anomalous": result["anomalous"],
                                "inference_ms": result["inference_ms"], **differences, "original_pixels_match": True})
        old_rows = read_rows(ROOT / "outputs/frontier/screw/seed-42/spatial_kcenter_r3/predictions.csv")
        old = next(row for row in old_rows if row["path"] == "test/good/000.png")
        old_threshold = read_json(ROOT / "outputs/frontier/screw/seed-42/spatial_kcenter_r3/summary.json")["threshold"]
        encoded = base64.b64encode((ROOT / "data/mvtec/screw/test/good/000.png").read_bytes()).decode("ascii")
        manual = request("/api/predict", {"model": "dinov3-screw", "image": encoded})
        differences = compare_prediction(manual, old, old_threshold, "HTTP/legacy-manual")
        predictions.append({"model": "dinov3-screw", "path": old["path"], "score": manual["score"],
                            "threshold": manual["threshold"], "anomalous": manual["anomalous"], **differences})
        automatic = request("/api/predict", {"model": "dinov3-auto", "image": encoded})
        if automatic.get("selected_category") != "screw":
            raise AssertionError("Legacy automatic route changed for a known screw example")
        differences = compare_prediction(automatic, old, old_threshold, "HTTP/legacy-auto")
        predictions.append({"model": "dinov3-auto", "path": old["path"], "selected_category": "screw",
                            "score": automatic["score"], "anomalous": automatic["anomalous"], **differences})
        for name, payload, headers, status in (
            ("invalid_base64", {"model": "screw-refined", "image": "!!!!"}, None, 400),
            ("invalid_image", {"model": "screw-refined", "image": base64.b64encode(b"not-an-image").decode()}, None, 400),
            ("unknown_mode", {"model": "unknown-mode", "image": encoded}, None, 400),
            ("wrong_content_type", {}, {"Content-Type": "text/plain"}, 415),
            ("cross_origin", {}, {"Content-Type": "application/json", "Origin": "http://example.org"}, 403)):
            response = request("/api/predict", payload, headers, expected=status)
            rejection_checks.append({"case": name, "status": status, "error": response.get("error")})
        return {"temporary_port": server.server_port, "cold_engine_load_ms": cold_ms,
                "models": sorted(ids), "predictions": predictions,
                "original_example_checks": example_checks, "rejected_uploads": rejection_checks,
                "port_18765_untouched": True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def save_record(record):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".partial")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(REPORT)


def main():
    record = {"status": "running", "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "score_tolerance": SCORE_TOLERANCE, "stages": {}, "errors": [],
              "test_status": "Previously exposed MVTec test; exploratory verification, not an independent unseen benchmark."}
    save_record(record)
    for name, function in (("unit_tests", stage_units), ("frozen_artifacts", stage_frozen),
                           ("algorithm_dependency_sources", stage_dependency_sources),
                           ("independent_metrics", stage_metrics), ("real_engine_and_cli", stage_inference),
                           ("real_http", stage_http)):
        before = time.perf_counter()
        try:
            result = function()
            record["stages"][name] = {"status": "passed", "elapsed_seconds": time.perf_counter()-before, "result": result}
            print(f"PASS: {name}", flush=True)
        except Exception as exc:
            error = {"stage": name, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            record["errors"].append(error)
            record["stages"][name] = {"status": "failed", "elapsed_seconds": time.perf_counter()-before}
            print(f"FAIL: {name}: {exc}", flush=True)
        save_record(record)
    record["status"] = "passed" if not record["errors"] else "failed"
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_record(record)
    print(json.dumps({"status": record["status"], "errors": record["errors"],
                      "report": REPORT.relative_to(ROOT).as_posix()}, ensure_ascii=False, indent=2), flush=True)
    if record["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
