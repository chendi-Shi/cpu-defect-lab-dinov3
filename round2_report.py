"""Summarize matched-representation experiment and frozen-model diagnosis."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def latest(method):
    runs = []
    for path in (ROOT / 'outputs').glob('*/summary.json'):
        s = json.loads(path.read_text(encoding='utf-8'))
        c = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))
        if (s['method'] == method and not s['smoke_run'] and c['size'] == 224
                and c['dims'] == 64 and c['bank_size'] == 2000 and c['seed'] == 42
                and c.get('threshold_split', 'all') == 'all'
                and c['category'] == 'bottle'):
            runs.append((path, s, c))
    if not runs:
        raise ValueError(f'Missing completed {method} run')
    return max(runs, key=lambda x: x[0].parent.name)


def main():
    runs = [latest(method) for method in ['global', 'global_shared', 'local']]
    if len({cfg['feature_cache_key'] for _, _, cfg in runs}) != 1:
        raise ValueError('Data/features differ; refusing matched comparison')
    for field in ['fit_files', 'calibration_files', 'test_files']:
        if any(cfg[field] != runs[0][2][field] for _, _, cfg in runs):
            raise ValueError(f'Splits differ: {field}')
    diagnosis = json.loads((ROOT / 'reports' / 'error_diagnosis.json').read_text(encoding='utf-8'))
    if diagnosis['frozen_run'] != runs[-1][0].parent.name:
        raise ValueError('Diagnosis and local run differ')
    names = {'global': '原始整图：layer4 / 512 维',
             'global_shared': '同层整图：layer2+3 / 64 维',
             'local': '局部：layer2+3 / 64 维'}
    lines = ['# 第二轮实验：同层特征对比与误报分析', '',
             '本轮为首轮测试之后的探索性分析。所有方案的数据哈希、训练／校准／测试文件列表均核验一致；没有修改原有模型或基于测试标签调阈值。', '',
             '| 方案 | 图像 AUROC | F1 | 漏检 | 误报 | 库向量数 | 库 MiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for _, s, _ in runs:
        m = s['threshold_metrics']
        lines.append(f"| {names[s['method']]} | {s['image_auroc']:.4f} | {m['f1']:.4f} | "
                     f"{m['fn']} | {m['fp']} | {s['bank_vectors']} | {s['bank_mib']:.3f} |")
    lines += ['', '## 能支持的结论', '',
              '同层整图与局部方案使用完全相同的 layer2+layer3 原始特征和同一组 64 个通道。整图先空间平均再归一化，局部保留 patch 并逐个归一化与匹配。',
              '这排除了首轮对比中的特征层级与维度差异，但特征库构成、向量数量和分数聚合方式仍随方案改变。该实验比较的是表示与评分策略，不能归因于单一因素，也不能推广为所有产品上的结论。',
              '固定阈值只来自各方案的正常校准分数。高 AUROC 不代表任意阈值下都没有误报。', '',
              '## 冻结模型的误报追溯', '',
              f"正常图片 `{diagnosis['query_file']}` 的分数是 {diagnosis['score']:.4f}，阈值为 {diagnosis['threshold']:.4f}，超出 {diagnosis['score_above_threshold']:.4f}。",
              f"最高分 patch 位于 14×14 网格的行列 {diagnosis['max_patch_grid_row_col']}（从 0 开始），224 输入上的网格框为 {diagnosis['max_patch_box_at_224']}。", '',
              '| 最近邻正常训练图片 | patch 行列 | 特征距离 |',
              '|---|---|---:|']
    for ref in diagnosis['nearest_training_patches']:
        lines.append(f"| {ref['training_file']} | {ref['grid_row_col']} | {ref['distance']:.4f} |")
    lines += ['', '![误报图片及最近邻证据](FALSE_POSITIVE_TRACE.png)', '',
              '图中从左到右为正常测试图、高分热力图、最高分网格单元放大图、最近邻正常训练图及对应单元。框表示网格位置，不等于 CNN 的完整感受野。', '',
              '## 正常校准样本的高分尾部', '',
              '| 正常校准图片 | 最高分 | 最高分 patch 行列 |',
              '|---|---:|---|']
    for ref in diagnosis['highest_normal_calibration_samples']:
        lines.append(f"| {ref['file']} | {ref['score']:.4f} | {ref['grid_row_col']} |")
    same_cell = sum(x['grid_row_col'] == diagnosis['max_patch_grid_row_col']
                    for x in diagnosis['highest_normal_calibration_samples'])
    lines += ['', f'上述最高分的 5 张正常校准图中，有 {same_cell} 张的最高分网格位置与测试误报相同。这说明该位置的正常高分现象不只出现在该测试图片上，但仍不足以证明具体原因。']
    lines += ['', '## 分析边界与下一步', '',
              '该正常图片在瓶底内部出现较高分数，表明冻结模型将此处的正常外观判作与特征库不够相似。反光、内部纹理变化或正常库覆盖不足是待验证假设，最近邻距离本身不能确定具体原因。',
              '本轮没有提高阈值或裁掉该区域以消除测试误报。下一轮可用训练侧合成缺陷验证库采样与聚合策略；最终泛化效果需要新的独立类别或数据。',
              '耗时受后台负载影响，不能从两次单次运行声称严格加速。', '',
              '## 可复查的运行文件', '']
    for p, _, _ in runs:
        lines.append(f'- [{p.parent.name}](../outputs/{p.parent.name}/summary.json)')
    lines += ['- [诊断数字记录](error_diagnosis.json)',
              '- [实验运行前固定的第二轮协议](../ROUND2_PROTOCOL.md)', '']
    dst = ROOT / 'reports' / 'ROUND2.md'
    dst.write_text('\n'.join(lines), encoding='utf-8')
    print(dst)


if __name__ == '__main__':
    main()
