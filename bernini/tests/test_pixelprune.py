import argparse
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)

from bernini.cli import add_common_args, build_pipeline
from bernini.models.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)
from bernini.pixelprune_utils import (
    PixelPruneConfig,
    compress_planner_visual_inputs,
    compute_visual_keep_metadata,
    merged_to_raw_patch_indices,
)


def tiny_vision_model():
    config = Qwen2_5_VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        window_size=8,
        fullatt_block_indexes=[1],
        _attn_implementation="eager",
    )
    return Qwen2_5_VisionTransformerPretrainedModel(config).eval()


class PixelPruneConfigTest(unittest.TestCase):
    def test_cli_defaults_defer_to_environment(self):
        parser = argparse.ArgumentParser()
        add_common_args(parser)
        args = parser.parse_args([])
        self.assertIsNone(args.pixelprune)
        self.assertIsNone(args.pixelprune_profile)

        with mock.patch.dict(
            os.environ,
            {
                "PIXELPRUNE_ENABLED": "true",
                "PIXELPRUNE_PROFILE": "true",
                "PIXELPRUNE_THRESHOLD": "0.25",
            },
            clear=False,
        ):
            config = PixelPruneConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertTrue(config.profile)
        self.assertEqual(config.threshold, 0.25)

    @mock.patch("bernini.cli.PretrainedConfig.get_config_dict")
    def test_renderer_only_rejects_pixelprune(self, get_config_dict):
        get_config_dict.return_value = ({"model_type": "bernini_renderer"}, {})
        args = SimpleNamespace(
            config="renderer",
            pixelprune=True,
            pixelprune_threshold=0.0,
            pixelprune_method="pred_2d",
            pixelprune_verbose=False,
            pixelprune_profile=False,
            pixelprune_metrics_output=None,
            pixelprune_warmup=False,
        )
        with self.assertRaisesRegex(ValueError, "only supports full Bernini"):
            build_pipeline(args, torch.device("cpu"))


class PixelPruneIndexTest(unittest.TestCase):
    def test_merged_indices_expand_to_complete_patch_groups(self):
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
        raw = merged_to_raw_patch_indices(
            [torch.tensor([0, 3]), torch.tensor([1])],
            grid,
            spatial_merge_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(raw.tolist(), [0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23])

    def test_selector_runs_only_on_source_items(self):
        def fake_selector(pixel_values, grid_thw, **kwargs):
            self.assertEqual(tuple(grid_thw.shape), (1, 3))
            self.assertEqual(pixel_values.shape[0], 16)
            return [torch.tensor([0, 3], dtype=torch.long)]

        pixels = torch.randn(24, 6)
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
        config = PixelPruneConfig(enabled=True)
        with mock.patch(
            "bernini.pixelprune_utils._load_selector",
            return_value=fake_selector,
        ):
            metadata = compute_visual_keep_metadata(
                pixels,
                grid,
                "image",
                ["source-input", "target-output"],
                2,
                config,
            )
        self.assertEqual(metadata[0].kept_merged_token_count, 2)
        self.assertEqual(metadata[1].kept_merged_token_count, 2)
        self.assertIsNone(metadata[1].merged_keep_indices)

    def test_no_source_visual_input_does_not_load_selector(self):
        pixels = torch.randn(16, 6)
        grid = torch.tensor([[1, 4, 4]])
        with mock.patch(
            "bernini.pixelprune_utils._load_selector"
        ) as load_selector:
            metadata = compute_visual_keep_metadata(
                pixels,
                grid,
                "video",
                ["target-output"],
                2,
                PixelPruneConfig(enabled=True),
            )
        load_selector.assert_not_called()
        self.assertIsNone(metadata[0].merged_keep_indices)


class PlannerCompressionTest(unittest.TestCase):
    def test_dense_positions_are_selected_and_output_query_is_unchanged(self):
        length = 10
        visual_input = torch.zeros(length, dtype=torch.bool)
        visual_input[1:5] = True
        visual_output = torch.zeros(length, dtype=torch.bool)
        visual_output[6:9] = True
        token_types = torch.zeros(length, dtype=torch.int)
        token_types[visual_input] = 2
        token_types[visual_output] = 3
        segment_ids = torch.arange(length)
        segment_ids[1:5] = 1
        segment_ids[6:9] = 2
        example = {
            "input_ids": torch.arange(length),
            "attention_mask": torch.ones(length, dtype=torch.long),
            "labels": torch.arange(length),
            "position_ids": torch.stack(
                [
                    torch.arange(length),
                    torch.arange(length) + 100,
                    torch.arange(length) + 200,
                ]
            ),
            "visual_input_token_mask": visual_input,
            "visual_output_token_mask": visual_output,
            "token_types": token_types,
            "token_segment_ids": segment_ids,
            "flex_token_types": torch.full((length,), -1),
            "attention_mask_4d": torch.empty(1, length, length),
            "vision_start_indices": [],
        }
        dense_positions = example["position_ids"].clone()
        keep_mask = compress_planner_visual_inputs(
            example, [torch.tensor([0, 2])]
        )
        self.assertEqual(keep_mask.nonzero().flatten().tolist(), [0, 1, 3, 5, 6, 7, 8, 9])
        self.assertTrue(
            torch.equal(example["position_ids"], dense_positions[:, keep_mask])
        )
        self.assertEqual(int(example["visual_input_token_mask"].sum()), 2)
        self.assertEqual(int(example["visual_output_token_mask"].sum()), 3)
        self.assertEqual(tuple(example["attention_mask_4d"].shape), (1, 8, 8))


class SparseVisionForwardTest(unittest.TestCase):
    def test_full_keep_matches_dense_forward(self):
        torch.manual_seed(0)
        model = tiny_vision_model()
        pixels = torch.randn(16, 3 * 2 * 2 * 2)
        grid = torch.tensor([[1, 4, 4]])
        with torch.no_grad():
            dense = model(pixels, grid)
            full_keep = model(
                pixels,
                grid,
                merged_keep_indices=[torch.arange(4)],
            )
        self.assertEqual(tuple(dense.shape), (4, 32))
        torch.testing.assert_close(dense, full_keep, rtol=0, atol=0)

    def test_sparse_video_layout_preserves_frames_and_groups(self):
        model = tiny_vision_model()
        grid = torch.tensor([[2, 4, 4]])
        keep = [torch.tensor([0, 3, 4, 7])]
        raw_indices, window_index, cu_windows, cu_frames = model._build_sparse_layout(
            grid, keep, torch.device("cpu")
        )
        self.assertEqual(raw_indices.numel(), 4 * 4)
        self.assertEqual(window_index.numel(), 4)
        self.assertEqual(cu_frames.tolist(), [0, 8, 16])
        self.assertEqual(int(cu_windows[-1]), 16)
        for group in raw_indices.reshape(-1, 4):
            self.assertEqual(group.tolist(), list(range(int(group[0]), int(group[0]) + 4)))

    def test_multi_item_full_keep_matches_dense_forward(self):
        torch.manual_seed(1)
        model = tiny_vision_model()
        pixels = torch.randn(24, 3 * 2 * 2 * 2)
        grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
        with torch.no_grad():
            dense = model(pixels, grid)
            full_keep = model(
                pixels,
                grid,
                merged_keep_indices=[torch.arange(4), torch.arange(2)],
            )
            sparse = model(
                pixels,
                grid,
                merged_keep_indices=[torch.tensor([0, 3]), torch.tensor([1])],
            )
        torch.testing.assert_close(dense, full_keep, rtol=0, atol=0)
        self.assertEqual(tuple(sparse.shape), (3, 32))


if __name__ == "__main__":
    unittest.main()
