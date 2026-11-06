"""HSV 색상 영역: 클릭 → HSV 범위 → 유지(keep) 마스크. torch 비의존 (OpenCV/NumPy only).

OpenCV HSV 규약: H 0-179, S/V 0-255.
"""
import cv2
import numpy as np

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def hsv_range_from_click(img_bgr: np.ndarray, x: int, y: int,
                         h_tol: int, s_tol: int, v_tol: int) -> tuple[list[int], list[int]]:
    """클릭 주변 5x5(경계 클립) HSV 채널별 중앙값 ± tol.

    S/V는 0-255로 클립한다. H는 클립하지 않는다(음수 또는 179 초과로 wrap 표현).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    y0, y1 = max(0, y - 2), min(h, y + 3)
    x0, x1 = max(0, x - 2), min(w, x + 3)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
    mh, ms, mv = (int(round(float(np.median(patch[:, c])))) for c in range(3))
    lower = [mh - h_tol, max(0, ms - s_tol), max(0, mv - v_tol)]
    upper = [mh + h_tol, min(255, ms + s_tol), min(255, mv + v_tol)]
    return lower, upper


def hsv_in_range(hsv: np.ndarray, lower: list[int], upper: list[int]) -> np.ndarray:
    """HxWx3 uint8 HSV → HxW bool. H 경계가 [0, 179]를 벗어나면 wrap 처리. S/V는 경계 포함."""
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo_h, hi_h = int(lower[0]), int(upper[0])
    if lo_h < 0 and hi_h > 179:
        h_ok = np.ones(hue.shape, bool)
    elif lo_h < 0:
        h_ok = (hue <= hi_h) | (hue >= 180 + lo_h)
    elif hi_h > 179:
        h_ok = (hue >= lo_h) | (hue <= hi_h - 180)
    else:
        h_ok = (hue >= lo_h) & (hue <= hi_h)
    s_ok = (sat >= lower[1]) & (sat <= upper[1])
    v_ok = (val >= lower[2]) & (val <= upper[2])
    return h_ok & s_ok & v_ok


def color_mask(img_bgr: np.ndarray, lower: list[int], upper: list[int],
               largest_only: bool, convex_hull: bool) -> np.ndarray:
    """범위 내 픽셀 → OPEN 후 CLOSE(5x5 타원) → (최대 성분만) → (성분별 볼록 껍질 채움). HxW bool."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = hsv_in_range(hsv, lower, upper).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
    if not mask.any():
        return np.zeros(mask.shape, bool)

    if largest_only:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = ((labels == biggest) * 255).astype(np.uint8)

    if convex_hull:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_mask = np.zeros_like(mask)
        for c in contours:
            cv2.fillPoly(hull_mask, [cv2.convexHull(c)], 255)
        mask = hull_mask

    return mask > 0
