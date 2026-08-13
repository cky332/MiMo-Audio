# MiMo-Audio 架构复现（from scratch）

依据技术报告《MiMo-Audio: Audio Language Models are Few-Shot Learners》
(arXiv:2512.23808) 的公式与结构描述，从零实现的最小可运行复现。
论文只开源了**推理**代码；本目录把论文声称的训练侧组件（两阶段 tokenizer
损失、GAN 判别器、A2T 联合 LLM、LM 的加权多通道损失）也补齐为可运行代码，
并用 CPU 小规模测试验证每个机制。

```
mimo_repro/
  config.py            # 论文超参（Table 2/3、2.1.1 节）+ tiny 测试配置
  transformer.py       # RoPE + 滑动窗口/因果注意力（替代官方 flash-attn 硬依赖）
  rvq.py               # EMA 残差向量量化（EnCodec 风格，同官方 quantization.py）
  tokenizer.py         # mel(100Hz) → conv/2 → encoder(+layer-3 skip) → pool/2
                       #   → RVQ(20 本) → causal decoder → Vocos 声码器 → wav
  tokenizer_losses.py  # Eq.1-7：A2T + 多尺度 mel 重建 + commit；MPD/MS-STFT
                       #   hinge GAN + feature matching（官方仓库完全没有）
  patch_lm.py          # Eq.8-15：patch encoder / 交错 LLM / delay patch decoder
                       #   + Table 3 损失权重 + 逐 patch 生成循环
tests/                 # 40 个测试：数值声称核验、因果性、delay 往返、
                       #   损失权重、toy 端到端训练收敛
```

运行：`cd reproduction && python -m pytest tests/ -q`（纯 CPU，约 6 秒）。

## 论文 ↔ 复现 ↔ 官方代码对照

| 论文 | 本复现 | 官方代码 |
|---|---|---|
| Eq.11 各码本 embedding 求和 | `PatchEncoder.frame_embeddings` | `_prepare_input_embeds` 循环累加 |
| 2.2.1 patch 内 Transformer + concat + 线性投影 | `PatchEncoder.forward` | `apply_input_local_transformer` + `speech_group_downcast` |
| Eq.14-15 delay 机制 | `build_delayed_patch` / `undelay_patch`（逐元素测试对拍） | `local_forward` 的 `cur_start <= t < cur_end` 窗口 |
| 2.2.3 R' 个输出头、causal patch decoder | `PatchDecoder` | `local_transformer` + `local_transformer_lm_heads` |
| Table 3 损失权重 100 / 12-8-6-4-2-2-1-1 | `MiMoLMConfig.text_loss_weight` / `audio_loss_weights` | 不在开源代码中（仅推理） |
| Eq.1 A2T 联合 LLM | `A2THead` | 不在开源代码中 |
| Eq.2 多尺度 mel 重建 e={5,6,7} | `multi_scale_mel_loss` | 不在开源代码中 |
| Eq.4-7 MPD+MS-STFT hinge GAN | `Discriminators` 等 | 不在开源代码中 |
| 2.1.1 layer-3 残差到输出 | `TransformerStack.skip_layer_id` | `encoder_skip_layer_id` |
| 2.1.1 声码器滑窗 [40,10] | `window_mask` | flash-attn `window_size=[40,10]` |

## 复现中发现的论文-代码差异（详见 ../PAPER_NOTES.md）

1. **empty token**：论文 Eq.15 写 "0 denotes an empty token"；官方代码实际在每个
   码本后**追加**一个 id（1025/129 号），并作为 `padding_idx` + 采样时禁采。
   本复现遵循代码。
2. **patch encoder 注意力方向**：论文正文说双向；官方代码
   `is_causal=not config.input_full_attention`，config 默认值下是**因果**。
   本复现默认双向（论文），提供开关复现代码行为。
3. **patch encoder 头数**：Table 2 写 16，论文正文与官方代码（`local_attn_heads`
   共享）都是 64。本复现取 64。
4. **训练侧全部缺失**：损失、判别器、A2T LLM、数据管线、训练循环，官方仓库
   一概没有；论文声称的 "full evaluation suite" 也不在本仓库。

## 与官方推理代码的已知差异

- 注意力用 PyTorch SDPA + 显式 mask，而非 flash-attn varlen（因此可在 CPU 跑）；
- LLM 主干是随机初始化的小型 TransformerStack，占位 MiMo-7B（Qwen2 架构）；
- 逐 patch 生成循环无 KV cache（O(T²)，小规模无碍）；
- 未实现官方的 streaming 分段解码与 batch 拼包。
