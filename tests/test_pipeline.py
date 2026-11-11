"""Tests for the cartoonify pipeline and CLI."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cartoonify import cartoonify, read_image, save_image
from cartoonify.cli import main


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory) -> Path:
    """A synthetic image with sharp shapes so edges are detectable."""
    rng = np.random.default_rng(42)
    image = np.full((200, 300, 3), 220, dtype=np.uint8)
    image[40:110, 60:160] = (30, 120, 220)
    image[90:170, 180:260] = (200, 40, 90)
    image += rng.integers(-10, 10, image.shape, dtype=np.int16).astype(np.uint8)
    path = tmp_path_factory.mktemp("data") / "sample.png"
    save_image(image, path)
    return path


STYLES = ("classic", "sketch", "comic", "hdr")


class TestPipeline:
    @pytest.mark.parametrize("style", STYLES)
    def test_all_styles_produce_rgb_cartoon(self, sample_image, style):
        stages = cartoonify(read_image(sample_image), style=style)

        assert stages.original.shape == (200, 300, 3)
        assert stages.grayscale.ndim == 2
        assert stages.cartoon.shape == (200, 300, 3)

    def test_classic_returns_all_stages(self, sample_image):
        stages = cartoonify(read_image(sample_image), style="classic")

        assert stages.edges.ndim == 2
        assert stages.flattened.shape == (200, 300, 3)

    def test_sketch_has_no_edge_stage(self, sample_image):
        stages = cartoonify(read_image(sample_image), style="sketch")

        assert stages.edges is None
        assert stages.flattened is None

    def test_comic_posterizes_colors(self, sample_image):
        stages = cartoonify(read_image(sample_image), style="comic")

        assert len(np.unique(stages.flattened.reshape(-1, 3), axis=0)) <= 8

    def test_edges_are_binary(self, sample_image):
        stages = cartoonify(read_image(sample_image), style="classic")
        assert set(np.unique(stages.edges)).issubset({0, 255})

    @pytest.mark.parametrize("style", STYLES)
    def test_cartoon_differs_from_original(self, sample_image, style):
        stages = cartoonify(read_image(sample_image), style=style)
        assert not np.array_equal(stages.cartoon, stages.original)

    def test_styles_produce_different_results(self, sample_image):
        image = read_image(sample_image)
        cartoons = [cartoonify(image, style=s).cartoon for s in STYLES]

        for i in range(len(cartoons)):
            for j in range(i + 1, len(cartoons)):
                assert not np.array_equal(cartoons[i], cartoons[j])

    def test_rejects_unknown_style(self, sample_image):
        with pytest.raises(ValueError, match="Unknown style"):
            cartoonify(read_image(sample_image), style="watercolor")

    def test_rejects_grayscale_input(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-channel"):
            cartoonify(gray)

    def test_read_image_rejects_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="Could not read"):
            read_image(tmp_path / "missing.png")

    def test_read_image_rejects_non_image_file(self, tmp_path):
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Could not read"):
            read_image(fake)


class TestCli:
    def test_run_creates_output(self, sample_image, tmp_path):
        output = tmp_path / "out.png"
        rc = main([str(sample_image), "-o", str(output)])

        assert rc == 0
        assert output.exists()
        assert read_image(output).shape == (200, 300, 3)

    def test_default_output_path(self, sample_image):
        rc = main([str(sample_image)])
        default = sample_image.with_name("sample_classic.png")

        assert rc == 0
        assert default.exists()

    def test_stages_dir(self, sample_image, tmp_path):
        stages_dir = tmp_path / "stages"
        rc = main([str(sample_image), "-o", str(tmp_path / "o.png"), "--stages-dir", str(stages_dir)])

        assert rc == 0
        for name in ("original", "grayscale", "edges", "flattened", "cartoon"):
            assert (stages_dir / f"{name}.png").exists()

    def test_sketch_stages_dir_skips_missing_stages(self, sample_image, tmp_path):
        stages_dir = tmp_path / "stages"
        rc = main(
            [str(sample_image), "--style", "sketch", "-o", str(tmp_path / "o.png"),
             "--stages-dir", str(stages_dir)]
        )

        assert rc == 0
        assert (stages_dir / "cartoon.png").exists()
        assert not (stages_dir / "edges.png").exists()

    @pytest.mark.parametrize("style", STYLES)
    def test_all_styles_run_via_cli(self, sample_image, tmp_path, style):
        output = tmp_path / f"{style}.png"
        rc = main([str(sample_image), "--style", style, "-o", str(output)])

        assert rc == 0
        assert output.exists()

    def test_missing_input_returns_error(self, tmp_path, capsys):
        rc = main([str(tmp_path / "missing.png")])

        assert rc == 1
        assert "Could not read" in capsys.readouterr().err

    def test_module_entry_point(self, sample_image, tmp_path):
        output = tmp_path / "mod.png"
        result = subprocess.run(
            [sys.executable, "-m", "cartoonify", str(sample_image), "-o", str(output)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert output.exists()
