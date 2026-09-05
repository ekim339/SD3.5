from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from . import dataset as dataset_module
from .conditioning import encode_t5_target_conditioning
from .dataset import (
    DatasetFormatError,
    SRNetSelfPromptDataset,
    build_style_prompt,
    prepare_conditions,
    read_labels,
    render_glyph,
)
from .model import SelfPromptingSD35, expand_sd3_input_projection
from .train import checkpoint_step, resolve_resume_checkpoint


def _save_mask(path: Path, pixels: tuple[tuple[int, int], ...]) -> None:
    mask = Image.new("L", (8, 8), 0)
    for pixel in pixels:
        mask.putpixel(pixel, 255)
    mask.save(path)


def _make_cooldown_shard(root: Path) -> None:
    for directory in ("i_s", "mask_s", "t_f", "mask_t"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    # The changed record sorts after the equal-text record, and target labels
    # deliberately use the opposite line order to exercise keyed alignment.
    (root / "i_s.txt").write_text(
        "a.png same\nb.png source\n", encoding="utf-8"
    )
    (root / "i_t.txt").write_text(
        "b.png target\na.png same\n", encoding="utf-8"
    )
    for filename in ("a.png", "b.png"):
        Image.new("RGB", (8, 8), "red").save(root / "i_s" / filename)
        Image.new("RGB", (8, 8), "blue").save(root / "t_f" / filename)
        _save_mask(root / "mask_s" / filename, ((1, 1),))
        _save_mask(root / "mask_t" / filename, ((6, 6),))


def test_training_uses_source_reconstruction():
    source = Image.new("RGB", (100, 50), "red")
    mask = Image.new("L", source.size, 0)
    for x in range(20, 60):
        for y in range(10, 30):
            mask.putpixel((x, y), 255)
    sample = prepare_conditions(
        source, mask, "source", 64, include_style_prompt=False
    )
    assert torch.equal(sample["source_image"], sample["target_image"])
    assert torch.equal(sample["mask"], sample["loss_mask"])
    assert "style_image" not in sample
    assert sample["masked_image"][:, 20:40, 20:40].min() == -1
    assert build_style_prompt(source, mask, (64, 64)).size == (64, 64)
    assert render_glyph("target", (128, 64)).getbbox() is not None


def test_cooldown_uses_aligned_target_and_source_only_conditions(tmp_path: Path):
    root = tmp_path / "shard"
    _make_cooldown_shard(root)

    dataset = SRNetSelfPromptDataset(
        [root], resolution=8, limit=1, mode="cooldown"
    )
    sample = dataset[0]

    # a.png is removed before the limit because its source and target glyphs
    # are equal; b.png is paired by filename rather than label-file line order.
    assert len(dataset) == 1
    assert sample["filename"] == "b.png"
    assert sample["source_text"] == "source"
    assert sample["target_text"] == "target"
    assert torch.equal(
        sample["source_image"][:, 0, 0], torch.tensor([1.0, -1.0, -1.0])
    )
    assert torch.equal(
        sample["target_image"][:, 0, 0], torch.tensor([-1.0, -1.0, 1.0])
    )

    # Only the source mask enters the model and constructs the masked/style
    # inputs. The union with mask_t is exposed separately for loss weighting.
    assert sample["mask"].sum().item() == 1
    assert sample["mask"][0, 1, 1].item() == 1
    assert sample["mask"][0, 6, 6].item() == 0
    assert sample["loss_mask"].sum().item() == 2
    assert sample["loss_mask"][0, 1, 1].item() == 1
    assert sample["loss_mask"][0, 6, 6].item() == 1
    assert torch.equal(
        sample["masked_image"][:, 0, 0], torch.tensor([1.0, -1.0, -1.0])
    )
    assert torch.equal(
        sample["masked_image"][:, 1, 1], torch.tensor([-1.0, -1.0, -1.0])
    )
    assert sample["style_image"][0].max().item() == 1
    assert sample["style_image"][2].max().item() == -1
    expected_glyph = prepare_conditions(
        Image.new("RGB", (8, 8), "black"),
        Image.new("L", (8, 8), 0),
        "target",
        8,
    )["glyph_image"]
    assert torch.equal(sample["glyph_image"], expected_glyph)


def test_dataset_defaults_to_style_free_self_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "self"
    (root / "i_s").mkdir(parents=True)
    (root / "mask_s").mkdir()
    (root / "i_s.txt").write_text("one.png original\n", encoding="utf-8")
    Image.new("RGB", (8, 8), "green").save(root / "i_s" / "one.png")
    _save_mask(root / "mask_s" / "one.png", ((2, 2),))

    def reject_style_crop(*_args, **_kwargs):
        raise AssertionError("self-reconstruction must not build a style crop")

    monkeypatch.setattr(dataset_module, "build_style_prompt", reject_style_crop)
    sample = SRNetSelfPromptDataset([root], resolution=8)[0]
    assert sample["source_text"] == sample["target_text"] == "original"
    assert torch.equal(sample["source_image"], sample["target_image"])
    assert torch.equal(sample["mask"], sample["loss_mask"])
    assert "style_image" not in sample


def test_dataset_rejects_invalid_mode_misalignment_and_duplicate_labels(
    tmp_path: Path,
):
    root = tmp_path / "invalid"
    _make_cooldown_shard(root)
    with pytest.raises(ValueError, match="Unsupported training mode"):
        SRNetSelfPromptDataset([root], mode="other")

    (root / "i_t.txt").write_text("missing.png target\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="filenames differ"):
        SRNetSelfPromptDataset([root], mode="cooldown")

    labels = tmp_path / "duplicate.txt"
    labels.write_text("same.png first\nsame.png second\n", encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="Duplicate filename"):
        read_labels(labels)


def test_projection_is_65_channels_and_preserves_base():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = nn.Module()
            self.pos_embed.proj = nn.Conv2d(16, 32, 2, 2)
            self.config = type("Config", (), {"in_channels": 16})()
        def register_to_config(self, **values):
            for key, value in values.items():
                setattr(self.config, key, value)
    transformer = Transformer()
    original = transformer.pos_embed.proj.weight.detach().clone()
    projection = expand_sd3_input_projection(transformer, 49)
    assert projection.in_channels == 65
    assert torch.equal(projection.weight[:, :16], original)
    assert torch.count_nonzero(projection.weight[:, 16:]) == 0


def test_missing_style_prompt_uses_zero_latent_without_changing_layout():
    model = object.__new__(SelfPromptingSD35)
    nn.Module.__init__(model)
    model.latent_channels = 2
    noisy = torch.randn(1, 2, 4, 4)
    masked = torch.randn_like(noisy)
    glyph = torch.randn_like(noisy)
    mask = torch.ones(1, 1, 8, 8)

    without_style = model.composite_input(noisy, masked, glyph, None, mask)
    assert without_style.shape == (1, 9, 4, 4)
    assert torch.count_nonzero(without_style[:, 6:8]) == 0

    style = torch.randn_like(noisy)
    with_style = model.composite_input(noisy, masked, glyph, style, mask)
    assert torch.equal(with_style[:, 6:8], style)


def test_cooldown_can_resume_from_a_separate_self_training_directory(tmp_path: Path):
    self_checkpoint = tmp_path / "self_reconstruction" / "checkpoint-050000"
    self_checkpoint.mkdir(parents=True)
    cooldown_output = tmp_path / "cooldown"
    cooldown_output.mkdir()

    resolved = resolve_resume_checkpoint(str(self_checkpoint), cooldown_output)
    assert resolved == self_checkpoint
    assert checkpoint_step(resolved) == 50_000

    (cooldown_output / "checkpoint-052500").mkdir()
    (cooldown_output / "checkpoint-055000").mkdir()
    assert (
        resolve_resume_checkpoint("latest", cooldown_output).name
        == "checkpoint-055000"
    )


def test_target_text_is_encoded_only_by_t5():
    calls = []
    expected_sequence = object()
    expected_pooled = object()

    class Pipeline:
        def encode_prompt(self, **kwargs):
            calls.append(kwargs)
            return expected_sequence, None, expected_pooled, None

    for target_text, targets in (
        ("glyph", ["glyph"]),
        (["first", "second"], ["first", "second"]),
    ):
        sequence, pooled = encode_t5_target_conditioning(
            Pipeline(), target_text, device="cuda", max_sequence_length=256
        )
        call = calls[-1]
        assert call["prompt"] == [""] * len(targets)
        assert call["prompt_2"] == [""] * len(targets)
        assert call["prompt_3"] == targets
        assert call["device"] == "cuda"
        assert call["num_images_per_prompt"] == 1
        assert call["do_classifier_free_guidance"] is False
        assert call["max_sequence_length"] == 256
        assert sequence is expected_sequence
        assert pooled is expected_pooled
