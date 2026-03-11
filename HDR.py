import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance

def read_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)

def save_image_rgb(path: str, rgb_u8: np.ndarray, quality: int = 95) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb_u8, mode="RGB").save(path, format="JPEG", quality=quality, optimize=True)

def shift_image_zero_pad(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    x_src0 = max(0, -dx)
    x_src1 = min(w, w - dx)
    y_src0 = max(0, -dy)
    y_src1 = min(h, h - dy)

    x_dst0 = max(0, dx)
    x_dst1 = min(w, w + dx)
    y_dst0 = max(0, dy)
    y_dst1 = min(h, h + dy)

    if x_src1 <= x_src0 or y_src1 <= y_src0:
        return out

    out[y_dst0:y_dst1, x_dst0:x_dst1, ...] = img[y_src0:y_src1, x_src0:x_src1, ...]
    return out

def compute_common_crop(shifts: List[Tuple[int, int]], w: int, h: int) -> Tuple[int, int, int, int]:
    dxs = [dx for dx, _ in shifts]
    dys = [dy for _, dy in shifts]

    left = max(0, max(dxs))
    right = min(w, min(w + dx for dx in dxs))
    top = max(0, max(dys))
    bottom = min(h, min(h + dy for dy in dys))

    left = int(left); right = int(right); top = int(top); bottom = int(bottom)
    if right - left < 10 or bottom - top < 10:
        return 0, 0, w, h
    return left, top, right, bottom

def rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray + 0.5, 0, 255).astype(np.uint8)

def resize_gray(gray_u8: np.ndarray, new_w: int) -> np.ndarray:
    h, w = gray_u8.shape
    if new_w >= w:
        return gray_u8
    new_h = max(1, int(round(h * (new_w / w))))
    im = Image.fromarray(gray_u8, mode="L")
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    return np.array(im, dtype=np.uint8)

def build_pyramid(gray_u8: np.ndarray, min_size: int = 64) -> List[np.ndarray]:
    pyr = [gray_u8]
    while min(pyr[-1].shape) > min_size:
        h, w = pyr[-1].shape
        new_w = max(min_size, w // 2)
        pyr.append(resize_gray(pyr[-1], new_w))
    return pyr[::-1]  # coarse -> fine

def mtb_and_exclusion(gray_u8: np.ndarray, tolerance: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(gray_u8)
    mtb = gray_u8 >= med
    excl = np.abs(gray_u8.astype(np.int16) - int(med)) > tolerance
    return mtb, excl

def mtb_error(mtb_ref: np.ndarray, excl_ref: np.ndarray,
              mtb_cur: np.ndarray, excl_cur: np.ndarray,
              dx: int, dy: int) -> int:

    mtb_cur_s = shift_image_zero_pad(mtb_cur.astype(np.uint8), dx, dy).astype(bool)
    excl_cur_s = shift_image_zero_pad(excl_cur.astype(np.uint8), dx, dy).astype(bool)

    mask = excl_ref & excl_cur_s
    diff = (mtb_ref ^ mtb_cur_s) & mask
    return int(diff.sum())


def align_mtb_translation(images_rgb: List[np.ndarray],
                          ref_index: int,
                          pyramid_min_size: int = 64,
                          tolerance: int = 6) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:

    grays = [rgb_to_gray_u8(im) for im in images_rgb]
    ref_gray = grays[ref_index]

    ref_pyr = build_pyramid(ref_gray, min_size=pyramid_min_size)
    pyrs = [build_pyramid(g, min_size=pyramid_min_size) for g in grays]

    shifts: List[Tuple[int, int]] = []
    aligned: List[np.ndarray] = []

    for i in range(len(images_rgb)):
        if i == ref_index:
            shifts.append((0, 0))
            aligned.append(images_rgb[i])
            continue

        dx, dy = 0, 0
        for lvl in range(len(ref_pyr)):
            ref_lvl = ref_pyr[lvl]
            cur_lvl = pyrs[i][lvl]

            mtb_ref, excl_ref = mtb_and_exclusion(ref_lvl, tolerance=tolerance)
            mtb_cur, excl_cur = mtb_and_exclusion(cur_lvl, tolerance=tolerance)

            if lvl > 0:
                dx *= 2
                dy *= 2

            best = (10**18, dx, dy)
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    cand_dx = dx + ddx
                    cand_dy = dy + ddy
                    err = mtb_error(mtb_ref, excl_ref, mtb_cur, excl_cur, cand_dx, cand_dy)
                    if err < best[0]:
                        best = (err, cand_dx, cand_dy)

            dx, dy = best[1], best[2]

        shifts.append((dx, dy))
        aligned.append(shift_image_zero_pad(images_rgb[i], dx, dy))

    return aligned, shifts

def weight_triangle(z_u8: np.ndarray) -> np.ndarray:
    z = z_u8.astype(np.int16)
    return np.minimum(z, 255 - z).astype(np.float32)

def pick_sample_points(mid_img_u8: np.ndarray,
                       n_samples: int = 300,
                       valid_range: Tuple[int, int] = (10, 245),
                       seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = mid_img_u8.shape
    mask = (mid_img_u8 >= valid_range[0]) & (mid_img_u8 <= valid_range[1])
    ys, xs = np.where(mask)
    if len(ys) == 0:
        ys = rng.integers(0, h, size=n_samples)
        xs = rng.integers(0, w, size=n_samples)
        return np.stack([ys, xs], axis=1)

    idx = rng.choice(len(ys), size=min(n_samples, len(ys)), replace=False)
    return np.stack([ys[idx], xs[idx]], axis=1)

def levels_stretch_u8(rgb_u8: np.ndarray, p_black: float = 0.5, p_white: float = 99.5) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0
    out = np.empty_like(x)

    for c in range(3):
        lo = np.percentile(x[..., c], p_black)
        hi = np.percentile(x[..., c], p_white)
        if hi <= lo + 1e-6:
            out[..., c] = x[..., c]
        else:
            out[..., c] = (x[..., c] - lo) / (hi - lo)

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)

def enhance_pil_pop(rgb_u8: np.ndarray,
                    color: float = 1.9,
                    contrast: float = 1.25,
                    brightness: float = 1.05,
                    sharpness: float = 1.2) -> np.ndarray:
    img = Image.fromarray(rgb_u8, mode="RGB")
    img = ImageEnhance.Color(img).enhance(color)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    return np.array(img, dtype=np.uint8)

def solve_debevec_response(Z_samples: np.ndarray,
                           ln_t: np.ndarray,
                           lambda_reg: float = 100.0) -> np.ndarray:
    P, N = Z_samples.shape
    nZ = 256
    n_eq = P * N + 1 + (nZ - 2)
    n_unknown = nZ + P  # g(0..255) + lnE_i

    A = np.zeros((n_eq, n_unknown), dtype=np.float64)
    b = np.zeros((n_eq,), dtype=np.float64)

    k = 0
    for i in range(P):
        for j in range(N):
            z = int(Z_samples[i, j])
            w = float(min(z, 255 - z))
            if w <= 0.0:
                continue
            A[k, z] = w
            A[k, nZ + i] = w
            b[k] = w * ln_t[j]
            k += 1
    A[k, 128] = 1.0
    b[k] = 0.0
    k += 1

    for z in range(1, nZ - 1):
        w = lambda_reg * min(z, 255 - z)
        A[k, z - 1] = w
        A[k, z] = -2.0 * w
        A[k, z + 1] = w
        b[k] = 0.0
        k += 1

    A = A[:k, :]
    b = b[:k]

    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    g = x[:nZ]
    return g


def merge_hdr_debevec(images_u8: np.ndarray,
                      times: np.ndarray,
                      g_channels: List[np.ndarray]) -> np.ndarray:
    N, H, W, _ = images_u8.shape
    ln_times = np.log(times.astype(np.float64) + 1e-12)  # (N,)

    hdr = np.zeros((H, W, 3), dtype=np.float32)

    for ch in range(3):
        g = g_channels[ch].astype(np.float64)
        num = np.zeros((H, W), dtype=np.float64)
        den = np.zeros((H, W), dtype=np.float64)

        for j in range(N):
            Z = images_u8[j, ..., ch]                   # (H,W) uint8
            Wgt = weight_triangle(Z).astype(np.float64) # (H,W) float64
            Gj = g[Z]                                   # (H,W) float64
            num += Wgt * (Gj - ln_times[j])
            den += Wgt

        lnE = num / (den + 1e-12)
        hdr[..., ch] = np.exp(lnE).astype(np.float32)

    return hdr

def to_uint8(rgb01: np.ndarray) -> np.ndarray:
    return np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)

def auto_expose_ldr(rgb01: np.ndarray, percentile: float = 99.0, target: float = 0.95) -> np.ndarray:
    rgb01 = np.clip(rgb01, 0.0, 1.0).astype(np.float32)
    L = 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]
    p = float(np.percentile(L, percentile))
    gain = target / (p + 1e-6)
    return np.clip(rgb01 * gain, 0.0, 1.0)

def tonemap_reinhard(hdr: np.ndarray, key: float = 0.36, gamma: float = 2.2) -> np.ndarray:
    hdr = np.maximum(hdr, 0.0).astype(np.float32)
    L = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
    delta = 1e-6
    Lavg = np.exp(np.mean(np.log(delta + L)))
    Lm = (key / (Lavg + 1e-12)) * L
    Ld = Lm / (1.0 + Lm)
    scale = Ld / (L + 1e-12)
    out = hdr * scale[..., None]
    out = np.clip(out, 0.0, 1.0)
    out = auto_expose_ldr(out, percentile=99.0, target=0.95)
    out = out ** (1.0 / gamma)
    return to_uint8(out)

def tonemap_drago_simple(hdr: np.ndarray, bias: float = 0.85, gamma: float = 2.2) -> np.ndarray:
    hdr = np.maximum(hdr, 0.0).astype(np.float32)
    L = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
    Lmax = float(np.max(L)) + 1e-12

    b = np.clip(bias, 1e-4, 0.9999)
    exponent = math.log(b) / math.log(0.5)

    Ld = np.log1p(L) / np.log1p(Lmax)
    Ld = np.power(np.clip(Ld, 0.0, 1.0), exponent)

    scale = Ld / (L + 1e-12)
    out = hdr * scale[..., None]
    out = np.clip(out, 0.0, 1.0)
    out = out ** (1.0 / gamma)
    return to_uint8(out)

def tonemap_mantiuk_simple(hdr: np.ndarray, scale: float = 0.7, saturation: float = 1.0, gamma: float = 2.2) -> np.ndarray:
    hdr = np.maximum(hdr, 0.0).astype(np.float32)
    L = (hdr[..., 0] + hdr[..., 1] + hdr[..., 2]) / 3.0
    logL = np.log(L + 1e-6)
    mu = float(np.mean(logL))
    sigma = float(np.std(logL)) + 1e-6

    z = (logL - mu) / sigma
    z = z * float(scale)
    Ld = np.exp(z)

    Rn = hdr[..., 0] / (L + 1e-12)
    Gn = hdr[..., 1] / (L + 1e-12)
    Bn = hdr[..., 2] / (L + 1e-12)

    sat = float(saturation)
    Rn = np.power(np.clip(Rn, 0, 10), sat)
    Gn = np.power(np.clip(Gn, 0, 10), sat)
    Bn = np.power(np.clip(Bn, 0, 10), sat)

    out = np.stack([Ld * Rn, Ld * Gn, Ld * Bn], axis=-1)
    p = np.percentile(out, 99.5)
    out = out / (p + 1e-12)
    out = np.clip(out, 0.0, 1.0)
    out = out ** (1.0 / gamma)
    return to_uint8(out)

def enhance_simple(ldr_u8: np.ndarray, saturation: float = 1.25, contrast: float = 1.05, sharpen: float = 0.4) -> np.ndarray:
    rgb = ldr_u8.astype(np.float32) / 255.0

    rgb = np.clip((rgb - 0.5) * contrast + 0.5, 0.0, 1.0)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin
    s = np.where(cmax > 1e-12, delta / (cmax + 1e-12), 0.0)
    s = np.clip(s * saturation, 0.0, 1.0)
    gray = (r + g + b) / 3.0
    rgb = gray[..., None] + (rgb - gray[..., None]) * (s[..., None] / (np.where(delta > 1e-12, s, 1.0)[..., None] + 1e-12))

    rgb = np.clip(rgb, 0.0, 1.0)

    if sharpen > 1e-6:
        pad = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blur = (
            pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:] +
            pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
            pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
        ) / 9.0
        rgb = np.clip(rgb + sharpen * (rgb - blur), 0.0, 1.0)

    return to_uint8(rgb)

@dataclass
class HDRParams:
    mtb_tolerance: int = 6
    mtb_min_size: int = 64
    samples: int = 350
    lambda_reg: float = 100.0

def run_pipeline(image_paths: List[str],
                 exposure_times: List[float],
                 out_dir: str,
                 params: HDRParams) -> None:
    if len(image_paths) != len(exposure_times):
        raise ValueError("Количество изображений должно совпадать с количеством выдержек")

    print(f"Загрузка {len(image_paths)} изображений...")
    images = [read_image_rgb(p) for p in image_paths]
    times = np.array(exposure_times, dtype=np.float32)

    shapes = [im.shape for im in images]
    if len(set(shapes)) != 1:
        raise ValueError(f"Все изображения должны быть одного размера. Сейчас: {set(shapes)}")

    h, w = images[0].shape[:2]
    ref_index = len(images) // 2
    print(f"Опорный кадр: #{ref_index+1} ({os.path.basename(image_paths[ref_index])})")

    print("Выравнивание MTB (сдвиг dx/dy)...")
    aligned, shifts = align_mtb_translation(
        images, ref_index=ref_index,
        pyramid_min_size=params.mtb_min_size,
        tolerance=params.mtb_tolerance
    )

    for i, (dx, dy) in enumerate(shifts):
        print(f"  кадр {i+1}: shift dx={dx}, dy={dy}")

    x0, y0, x1, y1 = compute_common_crop(shifts, w, h)
    aligned = [im[y0:y1, x0:x1, :] for im in aligned]
    images_stack = np.stack(aligned, axis=0)  # (N,H,W,3)
    Hc, Wc = images_stack.shape[1], images_stack.shape[2]
    print(f"После crop общий размер: {Wc}x{Hc}")

    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        save_image_rgb(os.path.join(out_dir, f"01_original_{i+1}.jpg"), im)
    for i, im in enumerate(aligned):
        save_image_rgb(os.path.join(out_dir, f"02_aligned_{i+1}.jpg"), im)

    print("Debevec: калибровка отклика g(z) по каналам...")
    ln_t = np.log(times.astype(np.float64) + 1e-12)
    mid = images_stack[ref_index]  # (H,W,3)

    g_channels = []
    for ch, name in enumerate(["R", "G", "B"]):
        print(f"  канал {name}: выборка {params.samples} точек...")
        pts = pick_sample_points(mid[..., ch], n_samples=params.samples, seed=42 + ch)
        Z_samples = np.stack([images_stack[:, y, x, ch] for (y, x) in pts], axis=0)  # (P,N)
        g = solve_debevec_response(Z_samples, ln_t, lambda_reg=params.lambda_reg)
        g_channels.append(g)
        print(f"    g({name}) готов")

    print("Слияние в HDR (радианс-карта)...")
    hdr = merge_hdr_debevec(images_stack, times, g_channels)

    print("Tone mapping: Reinhard / Drago / Mantiuk ...")
    reinh = tonemap_reinhard(hdr, key=0.18, gamma=2.2)
    drago = tonemap_drago_simple(hdr, bias=0.85, gamma=2.2)
    mant = tonemap_mantiuk_simple(hdr, scale=0.7, saturation=1.0, gamma=2.2)

    save_image_rgb(os.path.join(out_dir, "03_tonemap_reinhard.jpg"), reinh)
    save_image_rgb(os.path.join(out_dir, "04_tonemap_drago.jpg"), drago)
    save_image_rgb(os.path.join(out_dir, "05_tonemap_mantiuk.jpg"), mant)

    base = reinh
    base = levels_stretch_u8(base, p_black=0.5, p_white=99.5)
    final = enhance_pil_pop(base, color=2.0, contrast=1.28, brightness=1.06, sharpness=1.25)
    save_image_rgb(os.path.join(out_dir, "06_FINAL_enhanced.jpg"), final)

    print("\nГотово.")
    print(f"Результаты: {out_dir}")
    print("Главный файл: 06_FINAL_enhanced.jpg")

def parse_times(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    if not files:
        raise ValueError(f"В папке {folder} не найдено изображений")
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="images", help="Папка с входными кадрами")
    ap.add_argument(
        "--times",
        type=str,
        default="0.02,0.0333333333,0.04,0.05,0.0666666667",
        help="Выдержки в секундах через запятую (по умолчанию: 1/50,1/30,1/25,1/20,1/15)"
    )
    ap.add_argument("--output", type=str, default="hdr_results_no_cv2", help="Папка для результатов")
    ap.add_argument("--samples", type=int, default=350, help="Число точек для Debevec калибровки")
    ap.add_argument("--lambda_reg", type=float, default=100.0, help="Регуляризация гладкости g(z)")
    ap.add_argument("--mtb_tol", type=int, default=6, help="MTB exclusion tolerance")
    ap.add_argument("--mtb_min", type=int, default=64, help="Минимальный размер пирамиды MTB")
    args = ap.parse_args()

    image_paths = list_images(args.input)
    times = parse_times(args.times)

    if len(image_paths) < len(times):
        raise ValueError(f"В папке {args.input} изображений меньше, чем выдержек: {len(image_paths)} < {len(times)}")
    image_paths = image_paths[:len(times)]

    params = HDRParams(
        mtb_tolerance=args.mtb_tol,
        mtb_min_size=args.mtb_min,
        samples=args.samples,
        lambda_reg=args.lambda_reg
    )

    run_pipeline(image_paths, times, args.output, params)

if __name__ == "__main__":
    main()