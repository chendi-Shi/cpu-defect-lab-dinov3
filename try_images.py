"""Run the preselected public images through the unchanged localhost demo model."""
import base64
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent


def read_image_url(value):
    return Image.open(io.BytesIO(base64.b64decode(value.split(',', 1)[1]))).convert('RGB')


def draw_pairs(results, destination, title, footer, columns=2):
    rows = (len(results) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns * 2, figsize=(columns * 6, rows * 4.15), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i, item in enumerate(results):
        row, col = divmod(i, columns)
        output = item['result']
        image_name = Path(item['image']).name
        reference = item['reference_anomalous']
        truth = 'unknown' if reference is None else ('defect' if reference else 'normal')
        prediction = 'anomaly' if output['anomalous'] else 'normal'
        colour = '#a04016' if reference is None else ('#168055' if output['anomalous'] == reference else '#b92f33')
        ax, heat = axes[row, col * 2:col * 2 + 2]
        ax.imshow(read_image_url(output['original']))
        ax.set_title(f"{item['model']} / {item['reference_type']}\n{image_name} | reference: {truth}", fontsize=10)
        heat.imshow(read_image_url(output['heatmap']))
        heat.set_title(f"prediction: {prediction}\nscore {output['score']:.4f} / threshold {output['threshold']:.4f}",
                       fontsize=10, color=colour)
    fig.suptitle(title, fontsize=16, y=.985)
    fig.text(.02, .012, footer, fontsize=9, color='#526070', va='bottom')
    fig.subplots_adjust(left=.025, right=.985, top=.86 if rows == 1 else .91,
                        bottom=.12 if rows == 1 else .075, hspace=.38, wspace=.20)
    fig.savefig(destination, dpi=145)
    plt.close(fig)


def main():
    selection = json.loads((ROOT / 'reports/IMAGE_TRIAL_SELECTION.json').read_text(encoding='utf-8'))
    destination = ROOT / 'reports/image_trials'
    destination.mkdir(exist_ok=True)
    results = []
    for i, item in enumerate(selection['items']):
        path = ROOT / item['image']
        payload = path.read_bytes()
        request = urllib.request.Request('http://127.0.0.1:18765/api/predict',
            json.dumps({'model': item['model'], 'image': base64.b64encode(payload).decode('ascii')}).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=90) as response:
            output = json.load(response)
        record = dict(item, sha256=hashlib.sha256(payload).hexdigest(), result=output)
        results.append(record)
        for kind in ['original', 'heatmap']:
            read_image_url(output[kind]).save(destination / f'{i+1:02d}-{kind}.png')
        print(f"{item['image']}: {output['score']:.4f}, anomalous={output['anomalous']}", flush=True)

    bottle = [r for r in results if r['model'] == 'bottle' and r['reference_anomalous'] is not None]
    screw = [r for r in results if r['model'] == 'screw']
    external = [r for r in results if r['reference_anomalous'] is None]
    draw_pairs(bottle, destination / 'BOTTLE_TRIALS.png', 'Bottle: fixed random samples, unchanged model',
               'Source: MVTec AD / CC BY-NC-SA 4.0. Previously evaluated test images; this is a visual trial, not new validation.\n'
               'Each pair: resized model input / heatmap. Colours rescaled per image; scores are distances, not probabilities.')
    draw_pairs(screw, destination / 'SCREW_TRIALS.png', 'Screw: research control',
               'Source: MVTec AD / CC BY-NC-SA 4.0. Red prediction text marks an error against the dataset label.')
    draw_pairs(external, destination / 'EXTERNAL_TRIALS.png', 'New web photos: outside bottle model capture conditions',
               'Left photo: Miika Silfverberg / CC BY-SA 2.0. Right photo: Ildar Sagdejev / CC BY-SA 4.0. Wikimedia Commons.\n'
               'Images resized and heatmap overlaid. No same-domain ground truth; no accuracy claim.')

    # Keep structured results compact: image pixels already have standalone PNGs.
    for result in results:
        result['result'].pop('original')
        result['result'].pop('heatmap')
    (ROOT / 'reports/IMAGE_TRIAL_RESULTS.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ['# 公开图片实际检测', '',
            '使用已发布的模型和原有阈值，通过本机网页同一 HTTP 接口逐图推理；没有重建特征库、改阈值或调参。', '',
            '8 张 MVTec 图片按 seed=20261009 固定抽取。它们属于此前已经评估的官方测试集，本次用于直观试验，不能算新的独立泛化证据。另有 2 张新下载的 Wikimedia 普通照片，用于检查拍摄场景不匹配时的表现。', '',
            '| 图片 | 参考情况 | 分数 | 阈值 | 模型判断 | 核对 |', '|---|---|---:|---:|---|---|']
    names = {'good':'正常', 'broken_large':'大破损', 'broken_small':'小破损', 'contamination':'污染',
             'scratch_neck':'螺丝颈部划痕', 'external_full_bottle':'完整瓶外观／工业标签未知',
             'external_shattered_bottle':'来源描述破碎瓶／工业标签未知'}
    for item in results:
        out = item['result']
        reference = item['reference_anomalous']
        check = '场景外，不计准确率' if reference is None else ('正确' if out['anomalous'] == reference else ('误报' if out['anomalous'] else '漏检'))
        rows.append(f"| {item['image']} | {names[item['reference_type']]} | {out['score']:.4f} | {out['threshold']:.4f} | "
                    f"{'异常' if out['anomalous'] else '正常'} | {check} |")
    rows += ['', '![瓶口样本](image_trials/BOTTLE_TRIALS.png)', '', '![螺丝对照](image_trials/SCREW_TRIALS.png)', '',
             '![外部照片](image_trials/EXTERNAL_TRIALS.png)', '',
             '普通侧视整瓶／碎玻璃照片与训练的瓶口俯视图不同。高异常距离表示偏离正常特征库，不能解释为成功识别具体缺陷。', '',
             '## 来源与复查', '', '[固定样本清单](IMAGE_TRIAL_SELECTION.json) · [逐图原始分数](IMAGE_TRIAL_RESULTS.json)', '']
    for item in external:
        rows.append(f"- [{Path(item['image']).name}]({item['source_page']})：{item['author']}，[{item['license']}]({item['license_url']})。图表中缩放并叠加热力图。")
    rows.append('- [MVTec AD 数据来源](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)：CC BY-NC-SA 4.0。')
    (ROOT / 'reports/IMAGE_TRIALS.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
