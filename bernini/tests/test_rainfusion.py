import pytest
import torch

from bernini.attention import build_editing_rainfusion_selector


def test_v3_selector_has_kernel_layout_and_valid_counts():
    query = torch.randn(512, 2, 8)
    indices, counts = build_editing_rainfusion_selector(
        query,
        query,
        target_start=0,
        target_grid=(2, 16, 16),
        sparsity=0.7,
        block_size=128,
    )

    assert indices.shape == (4, 2, 4)
    assert counts.shape == (4, 2)
    assert indices.dtype == torch.int64
    assert counts.dtype == torch.int64
    assert torch.all((counts >= 1) & (counts <= 4))
    for query_block in range(4):
        for head in range(2):
            count = int(counts[query_block, head])
            chosen = indices[query_block, head, :count]
            padding = indices[query_block, head, count:]
            assert torch.all((chosen >= 0) & (chosen < 4))
            assert torch.all(chosen[:-1] <= chosen[1:])
            assert torch.all(padding == -1)


def test_v3_selector_protects_first_frame_blocks():
    query = torch.randn(512, 2, 8)
    indices, counts = build_editing_rainfusion_selector(
        query,
        query,
        target_start=0,
        target_grid=(2, 16, 16),
        sparsity=0.7,
        block_size=128,
    )

    # H*W=256 protects the target's first two query and KV blocks.
    assert torch.equal(counts[:2], torch.full_like(counts[:2], 4))
    for query_block in (2, 3):
        for head in range(2):
            chosen = indices[query_block, head, : counts[query_block, head]].tolist()
            assert 0 in chosen
            assert 1 in chosen


def test_editing_selector_sparsifies_reference_and_target_queries():
    # Block 0 is reference-only. The target's first frame [192, 384) overlaps
    # blocks 1 and 2, which remain dense as query and KV blocks.
    query = torch.randn(768, 2, 8)
    indices, counts = build_editing_rainfusion_selector(
        query,
        query,
        target_start=192,
        target_grid=(3, 12, 16),
        sparsity=0.75,
        block_size=128,
    )

    assert indices.shape == (6, 2, 6)
    assert torch.equal(counts[1:3], torch.full_like(counts[1:3], 6))
    for query_block in (0, 3, 4, 5):
        for head in range(2):
            chosen = indices[query_block, head, : counts[query_block, head]].tolist()
            # The target-first-frame KV blocks are protected for every query.
            assert {1, 2}.issubset(chosen)
            # Reference and later-target query blocks are sparse-eligible.
            assert int(counts[query_block, head]) < 6


def test_npu_single_block_v3_matches_dense_attention():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is not available")
    pytest.importorskip("mindiesd.layers.flash_attn.sparse_flash_attn")

    from bernini.attention import editing_rainfusion_attention

    device = torch.device("npu:0")
    sequence, heads, head_dim = 128, 2, 128
    query = torch.randn(sequence, heads, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    sparse = editing_rainfusion_attention(
        query,
        key,
        value,
        target_start=0,
        target_length=sequence,
        target_grid=(1, 8, 16),
        sparsity=0.7,
    )
    dense = torch_npu.npu_fusion_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        input_layout="BSND",
        scale=head_dim ** -0.5,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        head_num=heads,
    )[0].squeeze(0)
    torch.testing.assert_close(sparse.float().cpu(), dense.float().cpu(), rtol=1e-2, atol=1e-2)
