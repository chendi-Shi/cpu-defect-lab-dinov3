# DINOSaur / DINOv3 CPU 实验协议

固定于 2026-10-09，在新方法的测试推理之前。目标是复现近期论文的核心算法，并在本机完整评估，不声称排名全球第一。

## 依据

- [DINOSaur，ECCV 2026](https://arxiv.org/abs/2605.24251v2)：冻结 DINOv3、逐空间位置 greedy k-center 特征库、邻域约束检索、CLS 原型识别任务。
- [官方实现](https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD)，固定 commit `9574f14f2e5a99e605ed19f0ff78f0a496d29252`。
- 官方 timm DINOv3 ViT-S/16 权重：`timm/vit_small_patch16_dinov3.lvd1689m`，revision `3bf4720a82ec2066db88137180ff1f83a675cef0`。SHA256 与下载来源另存 provenance。

## 固定设置

1. CPU float32、1 线程、batch=2；正式测试前用固定随机张量确认本机单线程更适合当前资源，详见 RESOURCE_PROFILE.md；输入 RGB，以 torchvision v2 对 uint8 张量双线性 resize 224×224、antialias，再转 float32 和 ImageNet 归一化。
2. ViT-S/16 输出最终 LayerNorm 的 CLS 384 维与 196×384 patch；去掉 1 CLS＋4 register。维持官方核心源码的特征，不另作 L2。官方 README 关于 L2 的建议与源码不同，这里以源码为准。
3. timm RoPE periods 先转 bf16 再转回 float32，按照模型卡建议匹配原权重精度；timm 替代 Meta 接口仍属于适配，不能声称逐位一致。
4. 正常训练图按原 baseline seed=42 分为 80% 建库、20% held-out 正常校准；测试集保持官方原划分。coreset 随机首点种子 42、43、44，但三次实验的数据划分相同，以隔离库采样随机性。
5. 每空间位置库大小 `min(N,max(20,floor(0.1*N)))`；主方法邻域半径 3。率与半径使用论文固定设置，不看本次测试选参。对小 N 增加 min(N,...) 防止官方最少20点逻辑无限循环。
6. 四个固定类别依次 bottle、screw、hazelnut、metal_nut。前两类已见测试结果；后两类在此协议固定前未评估。新方法的所有类别结果完整报告。
7. 四个 DINOv3 方案：逐位置 k-center／半径3（主方法，3个库采样种子）；逐位置随机库／半径3（同预算采样消融）；相同 k-center 库全局检索（空间约束消融）；逐位置 k-center／半径0（邻域消融）。三组消融各固定seed42，不报告它们的多种子稳定性。主方案不随测试成绩改变。
8. 正常校准阈值固定取95%分位以与已有 baseline 一致。原论文97.5%训练分位、完整连续漂移/逻辑异常/边缘GPU协议不在本实验范围，明确记录差别。
9. 指标：图像 AUROC、224网格像素 AUROC、固定阈值混淆矩阵/F1、库字节数、建库时间、原始特征提取时间、缓存评分 median/p95；均不能混称完整推理或进程峰值内存。
10. 另做任务增量序列：每次加入一个类别的正常库，保留已有库和原型，使用最近 CLS 原型自动路由测试图；报告路由准确率、各阶段指标和原有类别变化。路由错误与库不变分开统计，不直接声称实测零遗忘。

## 工程验证

分块邻域检索与独立朴素参考一致；k-center无重复选点；小样本与全同向量不死循环；保存／加载模型一致；HTTP端到端分数和离线记录一致；网页可选择新版与旧版模型；逐图结果、失败案例、报告和源代码包齐全。

项目贡献属于论文核心算法 CPU 适配、可复现实验与消融分析，不包装为原创 DINOSaur 或论文完整 benchmark 复现。
