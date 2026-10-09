"""Create a readable report from the latest completed fixed baseline runs."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    runs = []
    for method in ['global', 'local']:
        candidates = []
        for path in (ROOT / 'outputs').glob('*/summary.json'):
            summary = json.loads(path.read_text(encoding='utf-8'))
            config = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
            if (summary['method'] == method and not summary['smoke_run']
                    and config['size'] == 224 and config['dims'] == 64
                    and config['bank_size'] == 2000 and config['seed'] == 42
                    and config.get('threshold_split', 'all') == 'all'
                    and summary['category'] == 'bottle'):
                candidates.append((path, summary, config))
        if not candidates:
            raise ValueError(f'No completed formal {method} baseline. Run run_baselines.ps1 first.')
        runs.append(max(candidates, key=lambda x: x[0].parent.name))
    if runs[0][2]['feature_cache_key'] != runs[1][2]['feature_cache_key']:
        raise ValueError('Baselines have different data/features; refusing direct comparison')
    lines = ['# 首轮 CPU 工业缺陷检测实验', '',
             '数据：MVTec AD bottle；两种方法使用相同训练／校准／测试划分。', '',
             '| 方案 | 图像 AUROC | 像素 AUROC（224 网格） | F1 | 评分 median ms | 特征库 MiB |',
             '|---|---:|---:|---:|---:|---:|']
    for path, s, _ in runs:
        pixel = f"{s['pixel_auroc']:.4f}" if s['pixel_auroc'] is not None else '—'
        f1 = s['threshold_metrics']['f1']
        lines.append(f"| {s['method']} | {s['image_auroc']:.4f} | {pixel} | {f1:.4f} | "
                     f"{s['scoring_median_ms']:.2f} | {s['bank_mib']:.3f} |")
    lines += ['', '## 固定阈值下的分类情况', '',
              '| 方案 | 缺陷类型 | 图片数 | 判为异常 |',
              '|---|---|---:|---:|']
    for _, s, _ in runs:
        for defect, values in s['defect_breakdown'].items():
            lines.append(f"| {s['method']} | {defect} | {values['images']} | {values['predicted_anomalous']} |")
    lines += ['', '## 如何理解', '',
              '这些是首轮固定配置的实测结果。两个完整方案使用不同层级和聚合方式；不能将差别全部归因于局部信息。',
              '评分耗时复用已缓存特征，不能称为完整推理耗时。像素指标在缩放后的网格计算，不能直接与论文的原图指标比较。',
              '耗时为本机当前运行环境下的观测值，未隔离后台负载，不作为严格性能基准。',
              '特征库占用不是进程峰值内存。阈值仅使用正常校准图片计算，F1 不使用测试集寻优。', '',
              '## 运行记录', '']
    for path, s, cfg in runs:
        lines += [f"- {s['method']}: [{path.parent.name}](../outputs/{path.parent.name}/summary.json)",
                  f"  建库 {s['fit_images']} 张、校准 {s['calibration_images']} 张、测试 {s['test_images']} 张；"
                  f"特征提取平均 {s['feature_extraction_mean_ms']:.2f} ms/图；缓存命中 {s['feature_cache_hit']}。"]
    lines += ['', '## 需要复查的误报和漏检', '']
    for path, s, _ in runs:
        with (path.parent / 'predictions.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        false_positive = sorted([r for r in rows if r['label'] == '0' and r['prediction'] == '1'],
                                key=lambda r: float(r['score']), reverse=True)[:5]
        false_negative = sorted([r for r in rows if r['label'] == '1' and r['prediction'] == '0'],
                                key=lambda r: float(r['score']))[:5]
        lines.append(f"### {s['method']}")
        lines.append('')
        lines.append('误报：' + (', '.join(r['path'] for r in false_positive) or '无'))
        lines.append('')
        lines.append('漏检：' + (', '.join(r['path'] for r in false_negative) or '无'))
        lines.append('')
    lines += ['## 下一轮实验', '',
              '先看失败案例；为后续参数选择单独设计验证集或训练侧合成缺陷验证方案。',
              '增加相同层级的全局／局部对比，控制层级因素；再研究特征维度与库大小。',
              '这些后续工作尚未完成，简历中不要写成已实现。', '']
    folder = ROOT / 'reports'
    folder.mkdir(exist_ok=True)
    dst = folder / 'BASELINE.md'
    dst.write_text('\n'.join(lines), encoding='utf-8')
    print(dst)


if __name__ == '__main__':
    main()
