# M1/M2 参考与 ICA 审核

## 三套通道用途

- `reference_channels` 固定为 `M1`、`M2`；先对完整头皮信号减去两侧乳突的平均值。
- `scalp_channels` 控制预处理及 ICA；默认 `null` 表示从原始记录中选取全部 EEG
  类型的头皮通道，排除 M1/M2。可明确填写实际帽子的头皮通道，不能只填 12 个 ROI。
- `eegnet_channels` 独立设置为 F3、Fz、F4、FC3、FCz、FC4、P3、Pz、P4、PO7、Oz、PO8。
  ROI 由 `features.rois` 定义，在完成头皮预处理后提取。

头皮通道数量取决于记录中实际存在且类型正确的电极，软件不会假设“64 通道系统”
等于“64 个可用头皮 EEG 电极”。M1/M2 缺失时停止，不用全头平均参考替代。

## 第一阶段：生成盲态审核材料

```console
mci-eeg ica-review --manifest private/manifest.csv --config private/preprocessing.json --output private/ica_review
```

每位受试者保存：

- `decomposition-ica.fif`：拟合好的 ICA 分解。
- `report.html`、成分地形图、前 20 秒成分时间序列及功率谱。
- `artifact_candidates.json`：存在 EOG/ECG 通道时的候选成分和评分，仅作参考。
- `review.json`：待填写的人工审核结果。
- `binding.json`：原始文件、参数、MNE 版本、通道和 ICA 文件哈希。

报告不包含诊断标签。若首段时间序列不足以判断，可用 MNE 打开保存的分解和完整
记录进一步检查。`eog_channels`、`ecg_channels` 必须按实际记录填写；没有专用
辅助通道时，不会假装已经自动确认眼动或心跳成分。

在 `review.json` 中分别记录 `ocular`（眼动/眨眼）、`saccade`（眼跳）、
`cardiac`（心跳）和必要时的 `other` 成分索引。审核完毕填写审核人，设置
`approved: true`，并在 `notes` 中说明依据。无成分需排除也必须确认审核。
此步骤需要真正查看数据，命令不会自动批准。

ICA 在头皮通道的 1–30 Hz 连续副本上拟合，应用到 0.1–30 Hz 分析信号。
这项实现细节应在方法中说明。MNE 1.13+ 使用 `rng`；旧版本使用兼容参数
`random_state`。不同版本/随机流可能产生不同成分，因此必须使用已审核保存的分解。

## 第二阶段：应用同一分解

```console
mci-eeg prepare --manifest private/manifest.csv --behavior private/behavior.csv --config private/preprocessing.json --ica-directory private/ica_review --output data/study
```

`prepare` 核对绑定信息，读取已审核的 ICA，不重新拟合。源数据、手动坏段/坏导、
配置或 MNE 版本变更后，需要重新生成并审核材料。

清单中的 `bad_channels` 是 JSON 通道列表；`bad_segments` 是连续记录中
`[[起点秒, 持续秒], ...]`。坏导按配置的电极位置插值，坏段标记为 `BAD_manual`。
请检查标准电极模板是否适合实际布局。分析分段为 −0.5–1.0 秒，基线 −0.5–0 秒；
采用绝对幅度 ±80 μV 剔除，不把它替换成 160 μV 峰峰值阈值。
