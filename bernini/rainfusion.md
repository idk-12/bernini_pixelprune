# Bernini DiT RainFusion v3

## Scope

This branch replaces only Wan2.2 DiT self-attention (`attn1`) with the
RainFusion v3 sparse student path. Text cross-attention, Qwen-VL attention, VAE,
scheduler, and sampling remain unchanged. Ulysses sequence parallel is
supported by running RainFusion after its gather-sequence/scatter-heads
all-to-all and before the inverse all-to-all.

The implementation intentionally stops at `output = RainFusion(Q, K, V)` and
does not include Wan v3's timestep-dependent features:

- no dense skip steps;
- no teacher attention;
- no compensation residual;
- no cross-timestep cache.

`rainfusion_start_step` remains only as an optional Bernini runtime switch for
choosing when to enable the backend. It is not passed into the v3 selector or
kernel.

## Full-visual-sequence data path

```text
packed visual Q/K/V [reference | target, H, D]
  -> under Ulysses: global sequence and local heads on every rank
  -> BSND [1, S, H, D]
  -> consecutive 128-token average pooling
  -> full visual-sequence block QK score + softmax
  -> per-head top-k for every reference and target query block
  -> dense target-first-frame query and KV blocks
  -> select_idx / select_num_idx
  -> BNSD [1, H, S, D]
  -> mindiesd sparse_flash_attn_rf_v2.rain_fusion_attention
  -> packed output [S, H, D]
  -> under Ulysses: gather heads and scatter sequence
```

Reference and target query blocks are both sparse-eligible and select KV blocks
from the complete packed visual sequence. The target video's first-frame query
and KV blocks remain dense. Text remains in the separate DiT cross-attention
path.

`rainfusion_sparsity=0.7` keeps 30% of KV blocks in the base top-k mask.
First-frame query/KV protection is applied afterwards, so the final effective
sparsity can be slightly lower than 0.7.

## Run

```bash
... existing Bernini command ... \
  --ulysses 2 \
  --dit_attention_backend rainfusion \
  --rainfusion_start_step 0
```

The runtime environment must provide:

```text
mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2.rain_fusion_attention
```

The known environment on this machine is:

```text
/home/xilab_program/envs/rainfusion
```
