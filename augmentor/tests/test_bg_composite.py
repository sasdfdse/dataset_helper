import numpy as np
import pytest

from background_swap.stages.composite import alpha_map, composite, diff_image, mask_overlay
from background_swap.common.domain import Job, MaskSet
from background_swap.stages.inpaint import run_inpaint, working_size
from background_swap.common.settings import BackgroundSwapSettings


def test_composite_protect_is_bytewise_original():
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8); gen = 255 - orig
    protect = np.zeros((60, 80), bool); protect[10:30, 20:50] = True
    out = composite(orig, gen, protect, feather_px=4)
    assert out.dtype == np.uint8
    assert (out[protect] == orig[protect]).all()


def test_feather_only_on_inpaint_side():
    orig = np.zeros((40, 40, 3), np.uint8); gen = np.full_like(orig, 200)
    protect = np.zeros((40, 40), bool); protect[:, :20] = True
    out = composite(orig, gen, protect, feather_px=4)
    assert 0 < out[20, 21, 0] < 200          # just outside the edge is blended
    assert out[20, 30, 0] == 200             # outside the feather stays generated


def test_alpha_map_values_and_edge_cases():
    protect = np.zeros((10, 10), bool); protect[:, :5] = True
    a = alpha_map(protect, 4)
    assert a.dtype == np.float32
    assert (a[:, :5] == 1.0).all()
    assert a[0, 5] == pytest.approx(1 - 1 / 5)
    assert a[0, 9] == 0.0
    assert (alpha_map(protect, 0) == protect.astype(np.float32)).all()
    assert (alpha_map(np.zeros((4, 4), bool), 4) == 0).all()


def test_diff_image_marks_protected_changes_red():
    orig = np.zeros((10, 10, 3), np.uint8); out = orig.copy(); out[2, 2] = 50
    protect = np.zeros((10, 10), bool); protect[2, 2] = True
    d = diff_image(orig, out, protect)
    assert d.shape == (10, 10, 3) and d.dtype == np.uint8
    assert tuple(d[2, 2]) == (0, 0, 255)


def test_diff_image_gray_is_max_channel_diff():
    orig = np.zeros((4, 4, 3), np.uint8); out = orig.copy(); out[1, 1] = (10, 70, 30)
    protect = np.zeros((4, 4), bool)
    d = diff_image(orig, out, protect)
    assert tuple(d[1, 1]) == (70, 70, 70)
    assert tuple(d[0, 0]) == (0, 0, 0)


def test_mask_overlay_precedence():
    img = np.full((10, 30, 3), 100, np.uint8)
    core = np.zeros((10, 30), bool); core[:, :10] = True
    protect = core.copy(); protect[:, 10:20] = True
    keep = np.zeros_like(core); keep[:, 5:30] = True
    out = mask_overlay(img, core, protect, keep)
    def blend(c):
        return tuple(int(round(0.55 * 100 + 0.45 * v)) for v in c)
    assert tuple(out[0, 2]) == blend((0, 0, 255))      # core > keep
    assert tuple(out[0, 15]) == blend((0, 140, 255))   # protect ring > keep
    assert tuple(out[0, 25]) == blend((0, 200, 0))     # keep only
    img2 = img.copy()
    mask_overlay(img2, core, protect, keep)
    assert (img2 == 100).all()                         # input untouched
    none = np.zeros_like(core)
    assert (mask_overlay(img, none, none, none) == img).all()


@pytest.mark.parametrize("w,h,native,mx,mode,manual,expected", [
    (640, 480, 1024, 1280, "auto", 1024, (1024, 768)),    # small source -> upscaled to native
    (1920, 1080, 1024, 1280, "auto", 1024, (1280, 720)),  # large source -> downscaled to the cap
    (1100, 700, 1024, 1280, "auto", 1024, (1104, 704)),   # within range -> source size (multiple of 8)
    (640, 480, 512, 1280, "auto", 1024, (640, 480)),      # SD1.5
    (640, 480, 1024, 1280, "manual", 768, (768, 576)),
    (1024, 768, 1024, 768, "auto", 1024, (768, 576)),     # native > max -> max wins
])
def test_working_size(w, h, native, mx, mode, manual, expected):
    assert working_size(w, h, native, mx, mode, manual) == expected


def test_run_inpaint_returns_original_size_and_preserves_protect():
    class FakeInpainter:
        native_px = 64
        def __init__(self): self.sizes = []
        def inpaint(self, image_rgb, mask, prompt, negative_prompt, seed):
            self.sizes.append(image_rgb.shape[:2]); return np.full_like(image_rgb, 123)
    orig = np.random.default_rng(0).integers(0, 256, (30, 50, 3), dtype=np.uint8)
    protect = np.zeros((30, 50), bool); protect[5:15, 5:20] = True
    masks = MaskSet(protect, protect, np.zeros_like(protect), ~protect, [], None)
    job = Job("a", 1, 1, 0, {}, "p", "n"); settings = BackgroundSwapSettings(max_resolution_px=128)
    fake = FakeInpainter()
    out = run_inpaint(fake, orig, masks, job, settings)
    assert out.shape == orig.shape and fake.sizes == [(40, 64)]
    assert (out[masks.protect] == orig[masks.protect]).all()


def test_run_inpaint_passes_rgb_and_bool_mask_and_converts_back():
    seen = {}
    class Fake:
        native_px = 64
        def inpaint(self, image_rgb, mask, prompt, negative_prompt, seed):
            seen.update(img=image_rgb.copy(), mask=mask, args=(prompt, negative_prompt, seed))
            return np.full_like(image_rgb, (10, 20, 30))  # RGB
    orig = np.zeros((64, 64, 3), np.uint8); orig[...] = (1, 2, 3)  # BGR
    protect = np.zeros((64, 64), bool); protect[:8] = True
    masks = MaskSet(protect, protect, np.zeros_like(protect), ~protect, [], None)
    out = run_inpaint(Fake(), orig, masks, Job("a", 1, 1, 7, {}, "p", "n"),
                      BackgroundSwapSettings(max_resolution_px=64))
    assert tuple(seen["img"][0, 0]) == (3, 2, 1)
    assert seen["mask"].dtype == bool and seen["mask"].shape == (64, 64)
    assert seen["args"] == ("p", "n", 7)
    assert tuple(out[40, 40]) == (30, 20, 10)  # BGR
