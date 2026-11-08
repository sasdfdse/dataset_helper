import cv2
import numpy as np

from background_swap.stages.color_region import color_mask, hsv_in_range, hsv_range_from_click

GREEN = (40, 160, 40)   # BGR


def _field_with_line():
    img = np.full((100, 100, 3), (200, 200, 200), np.uint8)
    img[20:80, 10:90] = GREEN
    img[47:53, 20:80] = (255, 255, 255)            # 필드 안의 흰 라인 (6px 두께: 5x5 CLOSE가 메우지 못함; 양끝은 연결 유지)
    return img


def test_hsv_range_from_click_centered_on_green():
    lo, hi = hsv_range_from_click(_field_with_line(), 30, 30, 10, 60, 60)
    h = int(cv2.cvtColor(np.uint8([[GREEN]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
    assert lo[0] == h - 10 and hi[0] == h + 10 and 0 <= lo[1] <= hi[1] <= 255


def test_convex_hull_fills_line_inside_field():
    img = _field_with_line(); lo, hi = hsv_range_from_click(img, 30, 30, 10, 60, 60)
    m = color_mask(img, lo, hi, largest_only=True, convex_hull=True)
    assert m[50, 50] and not m[5, 5]


def test_without_hull_line_is_excluded():
    img = _field_with_line(); lo, hi = hsv_range_from_click(img, 30, 30, 10, 60, 60)
    assert not color_mask(img, lo, hi, largest_only=True, convex_hull=False)[50, 50]


def test_largest_only_drops_small_blob():
    img = _field_with_line(); img[86:98, 2:14] = GREEN       # 모폴로지로 지워지지 않는 12px 블롭
    lo, hi = hsv_range_from_click(img, 30, 30, 10, 60, 60)
    assert not color_mask(img, lo, hi, largest_only=True, convex_hull=False)[92, 8]
    assert color_mask(img, lo, hi, largest_only=False, convex_hull=False)[92, 8]


def test_red_hue_wraparound():
    hsv = np.array([[[2, 200, 200], [178, 200, 200], [90, 200, 200]]], np.uint8)
    assert hsv_in_range(hsv, [-5, 100, 100], [5, 255, 255]).tolist() == [[True, True, False]]
    assert hsv_in_range(hsv, [175, 100, 100], [185, 255, 255]).tolist() == [[True, True, False]]
