"""Build final reports, release banks, and a source-only sharing archive."""
import csv
import json
import shutil
import zipfile
from pathlib import Path
from lab import ROOT


def find_run(category, method, dims=64, bank_size=2000, split='all'):
    candidates = []
    for path in (ROOT / 'outputs').glob('*/summary.json'):
        summary = json.loads(path.read_text(encoding='utf-8'))
        cfg = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
        if (not summary['smoke_run'] and cfg['category'] == category and cfg['method'] == method
                and cfg['dims'] == dims and cfg['bank_size'] == bank_size
                and cfg.get('threshold_split', 'all') == split and cfg['size'] == 224 and cfg['seed'] == 42):
            candidates.append((path.parent, summary, cfg))
    if not candidates:
        raise ValueError(f'Missing {category}/{method}/{dims}/{bank_size}/{split}')
    return max(candidates, key=lambda x: x[0].name)


def source_archive():
    def portable(value):
        if isinstance(value, str):
            return value.replace(str(ROOT), '.').replace(ROOT.as_posix(), '.')
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        return value

    delivery = ROOT / 'delivery'
    delivery.mkdir(exist_ok=True)
    archive = delivery / 'cpu-defect-lab-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            relative = p.relative_to(ROOT)
            if (p.is_file() and not set(relative.parts) & {'.venv', 'cache', 'data', 'outputs', 'release',
                                                         'delivery', '__pycache__', '.git'}
                    and p.suffix != '.pyc'):
                if p.suffix == '.json':
                    content = json.dumps(portable(json.loads(p.read_text(encoding='utf-8'))),
                                         ensure_ascii=False, indent=2)
                    z.writestr(str(Path('cpu-defect-lab') / relative), content)
                else:
                    z.write(p, Path('cpu-defect-lab') / relative)
    return archive


def main():
    selection = json.loads((ROOT / 'reports' / 'selection.json').read_text(encoding='utf-8'))
    chosen = selection['selected']
    baseline = find_run('bottle', 'local')
    bottle_selected = find_run('bottle', 'local', chosen['dims'], chosen['bank_size'], 'reserved')
    screw_global = find_run('screw', 'global_shared', chosen['dims'], chosen['bank_size'])
    screw_local = find_run('screw', 'local', chosen['dims'], chosen['bank_size'])
    if screw_global[2]['feature_cache_key'] != screw_local[2]['feature_cache_key']:
        raise ValueError('Screw comparison uses different data/features')
    if set(selection['tuning_normal_files']) & set(bottle_selected[2]['threshold_calibration_files']):
        raise ValueError('Tuning and threshold calibration overlap')
    if set(selection['reserved_threshold_files']) != set(bottle_selected[2]['threshold_calibration_files']):
        raise ValueError('Wrong reserved threshold split')
    recorded_runs = [find_run('bottle', 'global'), find_run('bottle', 'global_shared'),
                     baseline, bottle_selected, screw_global, screw_local]
    for run in recorded_runs:
        record_folder = ROOT / 'reports' / 'run_records' / run[0].name
        record_folder.mkdir(parents=True, exist_ok=True)
        for filename in ['config.json', 'summary.json', 'predictions.csv']:
            shutil.copyfile(run[0] / filename, record_folder / filename)
    releases = {}
    for category, run in [('bottle', baseline), ('screw', screw_local), ('bottle-validation', bottle_selected)]:
        folder = ROOT / 'release' / category
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run[0] / 'model.pt', folder / 'model.pt')
        shutil.copyfile(run[0] / 'summary.json', folder / 'summary.json')
        releases[category] = {'run': run[0].name, 'model': str((folder / 'model.pt').relative_to(ROOT)),
                              'summary': run[1]}
    (ROOT / 'release' / 'manifest.json').write_text(json.dumps(releases, indent=2), encoding='utf-8')
    lines = ['# 项目完整交付报告', '',
             '项目实现 CPU 工业异常检测、缺陷定位、三种表示策略、训练侧合成验证参数选择、独立类别检查、单图命令行与本机网页演示。', '',
             '## 训练侧验证与参数选择', '',
             '167 张正常 bottle 图片建库；42 张 held-out 正常训练图进一步分为互斥的 21 张调参正常图与 21 张阈值校准图。21 个合成划痕／斑点样本来自调参正常图。官方测试图或标签不用于参数选择。', '',
             '| 维度 | 库向量数 | 合成验证 AUROC | 库 MiB | 评分 median ms |',
             '|---:|---:|---:|---:|---:|']
    with (ROOT / 'reports' / 'VALIDATION_SWEEP.csv').open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            lines.append(f"| {row['dims']} | {row['bank_size']} | {float(row['validation_auroc']):.4f} | "
                         f"{float(row['bank_mib']):.3f} | {float(row['scoring_median_ms']):.2f} |")
    lines += ['', f"按预先固定的规则选择 **{chosen['dims']} 维、{chosen['bank_size']} 个库向量**。合成验证 AUROC 为 {chosen['validation_auroc']:.4f}；同分优先较小的库字节数。时间不参与选择。", '',
              '合成缺陷与真实缺陷分布不同，合成指标不替代真实测试。', '',
              '![训练侧参数对比](VALIDATION_SWEEP.png)', '',
              '## 完整数据结果', '',
              '| 实验 | 测试图片 | 图像 AUROC | 像素 AUROC（224 网格） | 漏检 | 误报 | 阈值校准图 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, run in [('bottle 首轮固定配置／默认演示', baseline),
                      ('bottle 参数选择后／探索性复测', bottle_selected),
                      ('screw 同层整图／首次评估', screw_global),
                      ('screw 局部／首次评估', screw_local)]:
        s = run[1]
        pixel = f"{s['pixel_auroc']:.4f}" if s['pixel_auroc'] is not None else '—'
        m = s['threshold_metrics']
        lines.append(f"| {name} | {s['test_images']} | {s['image_auroc']:.4f} | {pixel} | {m['fn']} | {m['fp']} | {s['calibration_images']} |")
    lines += ['', 'screw 的参数在其首次测试前已由 bottle 训练侧验证冻结。模型使用 screw 正常训练图片重新建库和校准，这是超参数跨类别检查，不是无需建库的零样本迁移。', '',
              'bottle 默认演示保留首轮固定配置与原 42 张正常校准阈值；参数选择后只使用另留的 21 张正常图片定阈值，作为独立对照保留。没有根据测试成绩替换默认模型。', '',
              '**跨类别检查未成功：screw 局部图像 AUROC 为 0.5190，接近随机排序，且漏检 108/119 张异常图。瓶口上的较好成绩不能推广到螺丝。网页中的螺丝模型仅作为失败研究对照。**', '',
              '## 演示与复现', '',
              '运行根目录 start_demo.cmd，等待 Demo ready 后在本机浏览器打开 http://127.0.0.1:18765。选择产品类别，再上传图片或点正常／缺陷示例。',
              '命令行、全流程复现、数据来源与许可证见 README.md；演示不需要云端服务。', '',
              '## 证据与边界', '',
              '- [首轮实验](BASELINE.md)', '- [同层对比与误报追溯](ROUND2.md)',
              '- [参数选择原始记录](selection.json)', '- [最终实验协议](../ROUND3_PROTOCOL.md)',
              '- [检查记录](QA.json)', '- [面试材料](INTERVIEW.md)', '',
              '局部特征库不是官方 PatchCore coreset。只检查两个类别；不能推广为完整工业部署。像素指标在缩放网格计算。分块库大小不是进程总内存。时间受后台负载影响，不声称严格加速。', '',
              '## 可复查的最终运行', '']
    for run in recorded_runs:
        lines.append(f'- [{run[0].name}](run_records/{run[0].name}/summary.json)')
    (ROOT / 'reports' / 'FINAL.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    evidence = {'selection': chosen, 'runs': {k: v['run'] for k, v in releases.items()},
                'screw_global_run': screw_global[0].name, 'screw_local': screw_local[1],
                'data_separation_verified': True}
    (ROOT / 'reports' / 'final_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('Release and report ready:', ROOT / 'reports' / 'FINAL.md', flush=True)
    print('Source archive:', source_archive(), flush=True)


if __name__ == '__main__':
    main()
