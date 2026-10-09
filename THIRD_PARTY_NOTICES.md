# 第三方来源与许可范围

根目录 MIT License 只覆盖本项目自编源码，不将第三方模型、数据、图片或它们的衍生图重新授权为 MIT。论文与原作者实现属于各自作者；本项目提供 CPU 适配、实验与工程验证。实现过程有 AI 编程助手参与，贡献说明见 [项目复盘](reports/PROJECT_STORY.md)。

## 方法与视觉骨干

- DINOSaur：[论文](https://arxiv.org/abs/2605.24251v2)，[原作者仓库固定版本](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD/tree/9574f14f2e5a99e605ed19f0ff78f0a496d29252)，上游源码采用 [Apache-2.0](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD/blob/9574f14f2e5a99e605ed19f0ff78f0a496d29252/LICENSE)。本项目未将原仓库源码作为 vendored 代码放入仓库；方法设计归原作者，CPU 实现与适配范围见 [协议](FRONTIER_PROTOCOL.md)。
- DINOv3：[Meta 官方仓库](https://github.com/facebookresearch/dinov3)，使用权重为 [timm DINOv3 ViT-S/16](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/tree/3bf4720a82ec2066db88137180ff1f83a675cef0)。权重受自定义 [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) 约束；仓库不附权重，下载记录保留版本及 SHA256。
- timm、PyTorch、torchvision 及其他安装依赖使用各自上游许可；仓库仅保留依赖声明，没有重新授权这些依赖。
- 原 ResNet18 局部正常匹配基线参考 [PatchCore 论文](https://arxiv.org/abs/2106.08265) 思路，具体简化设计见 [历史说明](reports/BASELINE_README.md)，并非官方 PatchCore coreset 的完整复现。

## MVTec AD 图片与衍生图

来源：[MVTec AD 官方数据集](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)，许可：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。本项目使用 bottle、screw、hazelnut、metal_nut 类别，原始数据不随仓库分发。

`reports/examples/`、`reports/frontier/examples/`、试用图目录、CLI 的原图与热力图、包含上述图片的网页截图，以及其他嵌入 MVTec AD 图片的图像，均保留原始来源并按 CC BY-NC-SA 4.0 分发。修改包括缩放、排版、标题、标注及模型热力图叠加；这些图像不是 MIT 授权的源码。报告中的独立数字、配置与预测表用于说明本次实验。

## Wikimedia Commons 试用图片

两个外部瓶子图片用于域偏移观察，不作为带工业标签的准确率测试。各张图片的作者、原始文件页、原许可及修改方式见 [旧版逐图试用](reports/IMAGE_TRIALS.md) 和 [新版逐图试用](reports/FRONTIER_IMAGE_TRIALS.md)。相应原图、叠加图与组合图沿用各自 Commons 页的许可和署名要求；未将它们改授 MIT。

仓库未附环境、缓存、预训练权重、原始数据或本机发布模型库。重新下载和使用第三方材料时，应遵守其相应许可。
