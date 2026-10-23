# 受试者水平 EEG 分类

[English](README.md)

本项目提供 M1/M2 参考的 EEG 预处理、手工特征、P1–P6 分类、完整流程置换和
独立的事后 SHAP 解释。

## 方法设置

- 双侧乳突参考：采用实际记录的 M1、M2 平均值。
- ICA：完整可用头皮通道；导出报告并盲态审核眼动、眼跳、心跳成分，保存同一分解。
- EEGNet：单独配置 12 个 ROI 通道，输入 0–1,000 ms。
- ERP：平均波幅及峰潜伏期；TF：先跨试次平均功率，再做基线 dB。
- PAC：完整分段 Hilbert 后取窗口；dwPLI²：先电极对计算，再 ROI 对平均。
- EEG 筛选：训练受试者均值差置换、家族 BH、绝对 d 排序及 0.70 相关去冗余。
- P3/P6：保留全部三项行为输入，共享 EEG 筛选，元学习器接收概率和处理后的原始特征。
- EEGNet：在外层训练集内部按受试者划分训练/验证子集并早停。
- 评价：六模型共享 10×5 外层划分，先平均每人的重复 OOF 概率再计算 AUC。
- 置换：真实标签重复性检查、3 次调试、1,000 次正式置换；每次重新拟合标签依赖步骤。

Stacking 内层概率依赖外层训练表征，完整流程性能仅由外层 OOF 评价。
短窗 PAC 和汇总时间点的 dwPLI² 仍有估计限制，详见[中文方法](docs/METHODS_ZH.md)。

## 安装与测试

```console
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

需要 Python >=3.11，默认 CPU。见[验证范围](docs/VALIDATION.md)。

## 模拟演示

```console
mci-eeg demo --output data/synthetic_demo
mci-eeg validate --data data/synthetic_demo --config configs/smoke.json --output results/demo
mci-eeg formal --data data/synthetic_demo --config configs/smoke.json --output results/demo
mci-eeg explain --data data/synthetic_demo --config configs/smoke.json --output results/demo_explanations
```

模拟信号和简化配置仅用于软件测试。

## 研究数据准备

在私有配置中填写真实事件编码、电极类型、位置和采样率。`scalp_channels=null`
表示全部可用头皮 EEG，排除 M1/M2，不强行假设具体电极数量。
`eegnet_channels` 是独立的 12 通道有序列表；`sfreq=null` 保留原采样率。
见[数据格式](docs/DATA_FORMAT.md)及[ICA 审核步骤](docs/ICA_REVIEW.zh-CN.md)。

```console
mci-eeg ica-review --manifest private/manifest.csv --config private/preprocessing.json --output private/ica_review
# 查看 report.html，填写每份 review.json 后再继续。
mci-eeg prepare --manifest private/manifest.csv --behavior private/behavior.csv --config private/preprocessing.json --ica-directory private/ica_review --output data/study
mci-eeg validate --data data/study --config configs/analysis.json --output results/study_v1
mci-eeg formal --data data/study --config configs/analysis.json --output results/study_v1
```

后续直接读取已审核 ICA，不重新拟合；源文件、参数、ICA 哈希和 MNE 版本均需一致。
变更后需重新审核。支持连续 FIF/SET；已分段 SET 需另行导入流程。

## 输出

保存受试者 OOF、逐折索引及特征、各次置换 AUC、错误、软件版本和哈希。
上尾 p=(1+置换 AUC≥观测 AUC 的次数)/(m+1)，调试置换单独保存。
空筛选输入预测训练集类别比例，拟合失败停止并记录。相同命令和未改变的协议可续跑。
异常锁文件须在确认对应进程停止后处理。

耗时应通过调试估算；每个外层折均重训 EEGNet。默认置换假设受试者标签可交换，
配对、家系或中心限制等设计需相应受限置换方法。

## 方法与发布

- [手稿对应表及待确认参数](docs/MANUSCRIPT_ALIGNMENT.md)
- [英文方法](docs/METHODS_EN.md) / [中文方法](docs/METHODS_ZH.md)
- [GitHub/Zenodo 发布](docs/RELEASE.md)

手稿未给出各 EEG 特征域的具体上限；当前 `null` 表示不设上限，按有上限的方案
分析前需补全。SHAP 使用独立全数据 Stacking 解释模型，不是 OOF 模型解释。
本包不含患者数据，采用 MIT 许可证。
