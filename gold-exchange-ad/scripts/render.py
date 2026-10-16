"""Frame compositor for the Gold Exchange Lab 30 s ad (1080x1920, 30 fps).

Composites the Higgsfield/Kling image-to-video clips with the motion-graphics
layer: kinetic typography, the real logo (never redrawn), particle systems,
the XRF scan HUD, the location card, Roman-Urdu subtitles, light sweeps,
flashes, camera shake, vignette and grain.

It renders two picture-only streams at once:
  out/_video_subs.mp4    with burned-in Roman Urdu subtitles
  out/_video_clean.mp4   identical, without subtitles (for re-use)
build.sh then muxes the final audio.

Usage:
  python3 render.py <work_dir> [--frames a:b] [--workers N] [--stills t1,t2,...]
"""
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import timeline as TL  # noqa: E402
from timeline import FPS, H, N_FRAMES, SYNC, W  # noqa: E402

WORK = "."
GOLD = np.array([0.93, 0.76, 0.40], np.float32)
WARM_WHITE = np.array([1.0, 0.93, 0.80], np.float32)

# ----------------------------------------------------------------- fonts ---
FONT_DIRS = [os.environ.get("FONT_DIR", ""), os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"),
             "fonts"]
INTER = "/usr/share/fonts/opentype/inter/Inter-{}.otf"   # local preview fallback


@lru_cache(None)
def font(weight, size):
    for d in FONT_DIRS:
        p = os.path.join(d, f"Montserrat-{weight}.ttf") if d else ""
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype(INTER.format(weight), size)


# ---------------------------------------------------------------- easing ---
def clamp01(x):
    return max(0.0, min(1.0, x))


def prog(t, a, d):
    return clamp01((t - a) / d)


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


def ease_out_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in_out(x):
    return x * x * (3 - 2 * x)


def ease_out_back(x, s=1.9):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def smooth(a, b, x):
    return ease_in_out(clamp01((x - a) / (b - a)))


# ---------------------------------------------------------------- layers ---
def to_premul(img):
    """PIL RGBA -> float32 premultiplied HxWx4."""
    a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


class Layer:
    """An RGBA graphic that can be drawn at any scale (premultiplied resize)."""

    def __init__(self, img):
        self.img = img.convert("RGBa")
        self.w, self.h = self.img.size
        self._cache = {}

    def get(self, s=1.0):
        key = round(s, 3)
        if key not in self._cache:
            if abs(key - 1.0) < 1e-3:
                im = self.img
            else:
                im = self.img.resize((max(1, int(self.w * key)), max(1, int(self.h * key))), Image.BICUBIC)
            a = np.asarray(im, np.float32) / 255.0
            if len(self._cache) > 40:
                self._cache.clear()
            self._cache[key] = a
        return self._cache[key]


def over(F, L, cx, cy, alpha=1.0, add=False):
    """Composite premultiplied layer L centred at (cx, cy) onto frame F in place."""
    if alpha <= 0.002:
        return
    h, w = L.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = L[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    region = F[fy0:fy1, fx0:fx1]
    if add:
        region += sub[..., :3] * alpha
    else:
        region *= 1 - sub[..., 3:4] * alpha
        region += sub[..., :3] * alpha


def gradient_gold(h, w, dark=False):
    stops = [(0.0, (1.00, 0.96, 0.78)), (0.28, (0.98, 0.83, 0.48)), (0.52, (0.80, 0.59, 0.25)),
             (0.68, (0.97, 0.82, 0.48)), (1.0, (0.66, 0.46, 0.16))]
    ys = np.linspace(0, 1, h)
    col = np.zeros((h, 3), np.float32)
    for c in range(3):
        col[:, c] = np.interp(ys, [s[0] for s in stops], [s[1][c] for s in stops])
    g = np.repeat(col[:, None, :], w, axis=1)
    xs = np.linspace(-1, 1, w)[None, :, None]
    g *= (1.0 - 0.10 * xs ** 2)
    if dark:
        g *= 0.45
    return np.clip(g, 0, 1)


def text_mask(text, fnt, tracking=0, pad=24):
    """Grayscale mask of a text line with letter tracking (px)."""
    adv = [fnt.getlength(ch) for ch in text]
    width = int(sum(adv) + tracking * (len(text) - 1)) + 2 * pad
    asc, desc = fnt.getmetrics()
    img = Image.new("L", (max(width, 4), asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(img)
    if tracking == 0:
        d.text((pad, pad), text, font=fnt, fill=255)
    else:
        x = pad
        for ch, a in zip(text, adv):
            d.text((x, pad), ch, font=fnt, fill=255)
            x += a + tracking
    return img


def style(mask, fill="white", bevel=False, shadow=0.55, shadow_blur=10, shadow_dy=6, glow=0.0, stroke=0):
    """Turn a mask into a finished RGBA text graphic."""
    m = np.asarray(mask, np.float32) / 255.0
    h, w = m.shape
    if fill == "gold":
        rgb = gradient_gold(h, w)
    elif fill == "subgold":   # flatter, brighter gold that stays legible over footage
        rgb = np.ones((h, w, 3), np.float32) * np.array([1.0, 0.80, 0.36], np.float32)
        rgb *= np.linspace(1.08, 0.92, h, dtype=np.float32)[:, None, None]
        rgb = np.clip(rgb, 0, 1)
    elif fill == "white":
        rgb = np.ones((h, w, 3), np.float32) * np.array([1.0, 0.99, 0.97], np.float32)
    else:
        rgb = np.ones((h, w, 3), np.float32) * np.array(fill, np.float32)
    if bevel:
        up = np.zeros_like(m)
        up[3:] = m[:-3]
        dn = np.zeros_like(m)
        dn[:-3] = m[3:]
        hl = np.clip(m - up, 0, 1)
        sh = np.clip(m - dn, 0, 1)
        hl = np.asarray(Image.fromarray((hl * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
        sh = np.asarray(Image.fromarray((sh * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
        rgb = np.clip(rgb + hl[..., None] * 0.55 - sh[..., None] * 0.35, 0, 1)
    face = np.dstack([rgb * m[..., None], m])
    out = np.zeros((h, w, 4), np.float32)
    if shadow > 0:
        sm = mask.filter(ImageFilter.GaussianBlur(shadow_blur))
        s = np.asarray(sm, np.float32) / 255.0 * shadow
        s = np.roll(s, shadow_dy, axis=0)
        out[..., 3] = s
    if stroke > 0:
        sk = np.asarray(mask.filter(ImageFilter.MaxFilter(2 * stroke + 1)).filter(ImageFilter.GaussianBlur(1)),
                        np.float32) / 255.0 * 0.75
        out[..., :3] *= 1 - sk[..., None]
        out[..., 3] = 1 - (1 - out[..., 3]) * (1 - sk)
    if glow > 0:
        gm = np.asarray(mask.filter(ImageFilter.GaussianBlur(18)), np.float32) / 255.0 * glow
        out[..., :3] = out[..., :3] * (1 - gm[..., None]) + GOLD * gm[..., None]
        out[..., 3] = 1 - (1 - out[..., 3]) * (1 - gm)
    out[..., :3] = out[..., :3] * (1 - face[..., 3:4]) + face[..., :3]
    out[..., 3] = 1 - (1 - out[..., 3]) * (1 - face[..., 3])
    return premul_to_img(out)


def premul_to_img(a):
    a = np.clip(a, 0, 1)
    alpha = a[..., 3:4]
    rgb = np.where(alpha > 1e-4, a[..., :3] / np.maximum(alpha, 1e-4), 0)
    return Image.fromarray((np.dstack([rgb, alpha]) * 255 + 0.5).astype(np.uint8), "RGBA")


@lru_cache(None)
def text_layer(text, weight, size, fill="white", tracking=0, bevel=False, shadow=0.55, glow=0.0, max_w=1000,
               stroke=0):
    fnt = font(weight, size)
    m = text_mask(text, fnt, tracking)
    while m.size[0] - 48 > max_w and size > 20:
        size = int(size * 0.95)
        fnt = font(weight, size)
        m = text_mask(text, fnt, tracking)
    return Layer(style(m, fill, bevel, shadow, glow=glow, stroke=stroke))


def sheen(L, pos, width=0.10, strength=0.85, angle=0.35):
    """Add a diagonal specular band across a premultiplied layer (copy)."""
    h, w = L.shape[:2]
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    d = xs + ys * angle * h / max(w, 1) - pos
    band = np.exp(-(d / width) ** 2) * strength
    out = L.copy()
    out[..., :3] += out[..., 3:4] * band[..., None] * WARM_WHITE
    return out


def extruded_layer(text, weight, size, depth=16, tracking=0):
    """3D gold extrusion (face + stacked darker sides + soft cast shadow)."""
    fnt = font(weight, size)
    m = text_mask(text, fnt, tracking, pad=60)
    w, h = m.size
    mf = np.asarray(m, np.float32) / 255.0
    canvas = np.zeros((h + depth + 40, w + depth + 40, 4), np.float32)
    sh = np.asarray(m.filter(ImageFilter.GaussianBlur(16)), np.float32) / 255.0 * 0.75
    canvas[depth + 24:depth + 24 + h, 10:10 + w, 3] = sh
    side = gradient_gold(h, w, dark=True)
    for k in range(depth, 0, -1):
        dx, dy = int(k * 0.45), k
        shade = 0.55 + 0.45 * (1 - k / depth)
        a = mf
        reg = canvas[dy:dy + h, dx:dx + w]
        reg[..., :3] = reg[..., :3] * (1 - a[..., None]) + side * shade * a[..., None]
        reg[..., 3] = 1 - (1 - reg[..., 3]) * (1 - a)
    face = np.asarray(style(m, "gold", bevel=True, shadow=0), np.float32) / 255.0
    fa = face[..., 3:4]
    reg = canvas[0:h, 0:w]
    reg[..., :3] = reg[..., :3] * (1 - fa) + face[..., :3] * fa
    reg[..., 3] = 1 - (1 - reg[..., 3]) * (1 - fa[..., 0])
    return Layer(premul_to_img(canvas)), (w, h)


# ------------------------------------------------------------ sprites/fx ---
def disc(r, soft=0.6):
    s = int(r * 4 + 4)
    y, x = np.mgrid[-s:s + 1, -s:s + 1].astype(np.float32)
    d = np.sqrt(x * x + y * y)
    core = np.exp(-(d / max(r, 0.5)) ** 2 * (1 + 3 * (1 - soft)))
    halo = np.exp(-(d / (r * 3.2)) ** 2) * 0.25
    a = np.clip(core + halo, 0, 1)
    return np.dstack([a[..., None] * GOLD * 1.15, a])


SPRITES = {r: disc(r) for r in (1.5, 2.5, 4.0, 7.0, 12.0)}


class Particles:
    def __init__(self, n, seed, rise=(12, 45), sizes=(1.5, 2.5, 4.0, 7.0, 12.0), weights=(5, 5, 3, 1.5, 0.6)):
        r = np.random.default_rng(seed)
        self.x = r.uniform(0, W, n)
        self.y = r.uniform(0, H, n)
        self.vy = -r.uniform(*rise, n)
        self.sway = r.uniform(5, 30, n)
        self.ph = r.uniform(0, 2 * math.pi, n)
        self.tw = r.uniform(0.6, 2.2, n)
        p = np.array(weights) / sum(weights)
        self.size = r.choice(sizes, n, p=p)
        self.bright = r.uniform(0.25, 1.0, n)

    def draw(self, F, t, k=1.0):
        if k <= 0.01:
            return
        ys = (self.y + self.vy * t) % (H + 80) - 40
        xs = self.x + self.sway * np.sin(t * 0.7 + self.ph)
        tw = 0.55 + 0.45 * np.sin(t * self.tw * 2 + self.ph)
        for x, y, s, b, a in zip(xs, ys, self.size, self.bright, tw):
            over(F, SPRITES[s], x, y, k * b * a * (0.55 if s >= 7 else 1.0), add=True)


class Burst:
    def __init__(self, t0, cx, cy, n=90, speed=(300, 1400), seed=3, life=1.3):
        r = np.random.default_rng(seed)
        ang = r.uniform(0, 2 * math.pi, n)
        v = r.uniform(*speed, n)
        self.vx, self.vy = np.cos(ang) * v, np.sin(ang) * v * 0.8
        self.size = r.choice([1.5, 2.5, 4.0, 7.0], n, p=[0.35, 0.35, 0.22, 0.08])
        self.t0, self.cx, self.cy, self.life = t0, cx, cy, life

    def draw(self, F, t, k=1.0):
        dt = t - self.t0
        if dt < 0 or dt > self.life:
            return
        damp = (1 - math.exp(-3.2 * dt)) / 3.2
        fade = (1 - dt / self.life) ** 1.5
        for vx, vy, s in zip(self.vx, self.vy, self.size):
            over(F, SPRITES[s], self.cx + vx * damp, self.cy + vy * damp + 120 * dt * dt, k * fade, add=True)


def flash(F, amount, color=WARM_WHITE):
    if amount > 0.002:
        F += color * amount


def anamorphic(F, t, t0, cy=H / 2, dur=0.5, k=1.0):
    """Horizontal light streak used on transitions."""
    p = (t - t0) / dur + 0.5
    if not 0 < p < 1:
        return
    env = math.sin(math.pi * p) ** 2 * k
    ys = np.arange(H, dtype=np.float32)
    prof = np.exp(-((ys - cy) / 6) ** 2) * 1.6 + np.exp(-((ys - cy) / 90) ** 2) * 0.35
    xs = np.linspace(-1, 1, W, dtype=np.float32)
    spread = 0.25 + 0.9 * p
    horiz = np.exp(-(xs / spread) ** 2)
    F += (prof[:, None, None] * horiz[None, :, None]) * WARM_WHITE * env


def make_vignette(strength=0.55):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W * 0.62)) ** 2 + ((y - H * 0.48) / (H * 0.62)) ** 2)
    return (1 - strength * np.clip(d, 0, 1.3) ** 2.2)[..., None].astype(np.float32)


VIGNETTE = make_vignette(0.55)
VIGNETTE_SOFT = make_vignette(0.35)
_GR = np.random.default_rng(11)
GRAIN = [(_GR.standard_normal((H // 2, W // 2, 1)).astype(np.float32) * 0.012) for _ in range(6)]


def grain(F, f):
    g = GRAIN[f % 6]
    F += np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)


def shake_offset(t, t0, amp=18, dur=0.45):
    dt = t - t0
    if dt < 0 or dt > dur:
        return 0, 0
    e = (1 - dt / dur) ** 2 * amp
    return int(e * math.sin(dt * 90)), int(e * math.cos(dt * 71))


def apply_shake(F, dx, dy):
    if dx == 0 and dy == 0:
        return F
    return np.roll(np.roll(F, dy, axis=0), dx, axis=1)


# ------------------------------------------------------------------ clips ---
class Clip:
    """Decoded once to a raw RGB memmap at 1080x1920 / 30 fps."""

    def __init__(self, name):
        cfg = TL.CLIPS[name]
        self.start, self.src = cfg["start"], cfg["src"]
        path = os.path.join(WORK, cfg["file"])
        cache = os.path.join(WORK, "cache")
        os.makedirs(cache, exist_ok=True)
        raw = os.path.join(cache, f"{name}.rgb")
        if not os.path.exists(raw):
            vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                  f"crop={W}:{H},fps={FPS},format=rgb24")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", vf, "-f", "rawvideo",
                            raw + ".tmp"], check=True)
            os.replace(raw + ".tmp", raw)
        self.frames = np.memmap(raw, np.uint8, "r").reshape(-1, H, W, 3)

    def at(self, t, speed=1.0):
        i = int(round((self.src + (t - self.start) * speed) * FPS))
        i = max(0, min(len(self.frames) - 1, i))
        return self.frames[i].astype(np.float32) / 255.0


def push(F, s, cx=0.5, cy=0.5):
    """Digital camera push-in (s >= 1)."""
    if s <= 1.001:
        return F
    w, h = int(W / s), int(H / s)
    x0 = int((W - w) * cx)
    y0 = int((H - h) * cy)
    crop = F[y0:y0 + h, x0:x0 + w]
    im = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def grade(F, warm=(1.04, 1.0, 0.92), contrast=1.08, lift=0.0):
    F = (F - 0.5) * contrast + 0.5 + lift
    return np.clip(F * np.array(warm, np.float32), 0, 1.2)


# ------------------------------------------------------------ brand assets ---
def load_logo():
    src = Image.open(os.path.join(WORK, "refs", "logo.jpg")).convert("RGB")
    a = np.asarray(src, np.float32) / 255.0
    lum = a.max(axis=2)
    alpha = np.clip((lum - 0.07) / 0.22, 0, 1)
    ys, xs = np.where(alpha > 0.05)
    pad = 8
    y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    a, alpha = a[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    rgb = np.where(alpha[..., None] > 1e-3, a / np.maximum(alpha[..., None], 1e-3), 0)
    img = Image.fromarray((np.dstack([np.clip(rgb, 0, 1), alpha]) * 255).astype(np.uint8), "RGBA")
    return img


class Brand:
    def __init__(self):
        logo = load_logo()
        self.logo_src = logo

    @lru_cache(None)
    def logo(self, width):
        im = self.logo_src.resize((width, int(self.logo_src.height * width / self.logo_src.width)), Image.LANCZOS)
        return Layer(im)

    @lru_cache(None)
    def logo_glow(self, width):
        im = self.logo_src.resize((width, int(self.logo_src.height * width / self.logo_src.width)), Image.LANCZOS)
        a = im.split()[3]
        pad = 120
        big = Image.new("L", (im.width + 2 * pad, im.height + 2 * pad), 0)
        big.paste(a, (pad, pad))
        g = np.asarray(big.filter(ImageFilter.GaussianBlur(45)), np.float32) / 255.0
        return np.dstack([g[..., None] * GOLD, g]).astype(np.float32)


def draw_logo(F, brand, t, t0, cx, cy, width, scale_from=1.14, sheen_times=()):
    p = prog(t, t0, 1.25)
    if p <= 0:
        return
    s = scale_from + (1 - scale_from) * ease_out_cubic(p)
    L = brand.logo(width).get(s).copy()
    h, w = L.shape[:2]
    # left-to-right reveal with a soft edge + light band at the edge
    rv = -0.15 + 1.45 * ease_out_cubic(prog(t, t0, 1.0))
    xs = np.linspace(0, 1, w, dtype=np.float32)
    m = np.clip((rv - xs) / 0.12, 0, 1)[None, :, None]
    edge = np.exp(-((xs - rv + 0.04) / 0.05) ** 2)[None, :, None] * (rv < 1.15)
    L = L * m
    L[..., :3] += L[..., 3:4] * edge * WARM_WHITE * 1.1
    for ts in sheen_times:
        q = prog(t, ts, 0.9)
        if 0 < q < 1:
            L = sheen(L, -0.2 + 1.4 * ease_in_out(q), 0.08, 0.9)
    glow = brand.logo_glow(width)
    breathe = 0.55 + 0.15 * math.sin(t * 2.2)
    over(F, glow, cx, cy, ease_out_cubic(p) * breathe, add=True)
    over(F, L, cx, cy, 1.0)


def bg_charcoal():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / W) ** 2 + ((y - H * 0.40) / H) ** 2 * 1.4)
    c0 = np.array([0.15, 0.115, 0.07], np.float32)
    c1 = np.array([0.012, 0.010, 0.008], np.float32)
    k = np.clip(d / 0.75, 0, 1)[..., None] ** 0.9
    return (c0 * (1 - k) + c1 * k).astype(np.float32)


# ------------------------------------------------------- kinetic helpers ---
def words_line(words, weight, size, tracking=0, gap=None):
    """Lay out [(text, fill)] on one line; returns layers + x offsets + total width."""
    fnt = font(weight, size)
    gap = gap if gap is not None else fnt.getlength(" ") + tracking
    layers, widths = [], []
    for txt, fill in words:
        L = text_layer(txt, weight, size, fill, tracking, bevel=(fill == "gold"), max_w=2000)
        layers.append(L)
        widths.append(L.w - 48)
    total = sum(widths) + gap * (len(words) - 1)
    xs, x = [], -total / 2
    for wd in widths:
        xs.append(x + wd / 2)
        x += wd + gap
    return layers, xs, total


def kinetic_words(F, t, t0, words, weight, size, cy, stagger=0.11, rise=70, dur=0.55, tracking=0,
                  out_t=None, fit=980):
    layers, xs, total = words_line(words, weight, size, tracking)
    sc = min(1.0, fit / total)
    for i, (L, x) in enumerate(zip(layers, xs)):
        p = prog(t, t0 + i * stagger, dur)
        if p <= 0:
            continue
        e = ease_out_expo(p)
        a = min(1.0, p * 2.2)
        if out_t is not None:
            q = prog(t, out_t, 0.35)
            a *= 1 - q
        s = sc * (1.0 + 0.25 * (1 - e))
        over(F, L.get(s), W / 2 + x * sc, cy + rise * (1 - e), a)


def letters(F, t, t0, text, weight, size, cy, fill="white", tracking=10, stagger=0.035, dur=0.5,
            alpha=1.0, fit=980):
    fnt = font(weight, size)
    adv = [fnt.getlength(c) for c in text]
    total = sum(adv) + tracking * (len(text) - 1)
    sc = min(1.0, fit / total)
    x = W / 2 - total * sc / 2
    for i, (c, a) in enumerate(zip(text, adv)):
        if c != " ":
            p = prog(t, t0 + i * stagger, dur)
            if p > 0:
                L = text_layer(c, weight, size, fill, 0, bevel=(fill == "gold"), shadow=0.5)
                e = ease_out_cubic(p)
                over(F, L.get(sc), x + a * sc / 2, cy + 28 * (1 - e), min(1, p * 1.8) * alpha)
        x += (a + tracking) * sc
    return total * sc


def hline(F, cx, cy, length, thick=2, alpha=1.0, color=GOLD):
    if length < 1 or alpha <= 0:
        return
    x0, x1 = int(cx - length / 2), int(cx + length / 2)
    y0 = int(cy - thick / 2)
    x0, x1 = max(0, x0), min(W, x1)
    if x1 > x0:
        prof = np.linspace(0, 1, x1 - x0, dtype=np.float32)
        fade = np.minimum(1, np.minimum(prof, 1 - prof) * 8)[None, :, None]
        F[y0:y0 + thick, x0:x1] = F[y0:y0 + thick, x0:x1] * (1 - fade * alpha) + color * fade * alpha


def rounded_panel(w, h, r=26, fill=(0, 0, 0, 150), border=(214, 172, 92, 210), bw=2):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], r, fill=fill, outline=border, width=bw)
    return Layer(img)


def darken_region(F, y0, y1, k):
    """Vertical gradient darkening for text legibility (k = max darkness)."""
    ys = np.arange(H, dtype=np.float32)
    m = np.clip(1 - np.abs((ys - (y0 + y1) / 2) / ((y1 - y0) / 2)), 0, 1) ** 0.6 * k
    F *= (1 - m)[:, None, None]


# ------------------------------------------------------------- subtitles ---
SUB_SIZE = 54
SUB_MAXW = 940
SUB_BOTTOM = 1600
SUB_LH = 72


class Subtitles:
    def __init__(self):
        self.events = TL.subtitle_events()
        self.cards = [self._layout(ws) for _, _, ws in self.events]

    def _layout(self, ws):
        fnt = font("Bold", SUB_SIZE)
        space = fnt.getlength(" ")
        lines, cur, cur_w = [], [], 0
        for w, t, hl in ws:
            wd = fnt.getlength(w)
            if cur and cur_w + space + wd > SUB_MAXW:
                lines.append((cur, cur_w))
                cur, cur_w = [], 0
            cur.append((w, t, hl, wd))
            cur_w += wd + (space if len(cur) > 1 else 0)
        lines.append((cur, cur_w))
        lines = lines[-2:]
        items = []
        y = SUB_BOTTOM - SUB_LH * (len(lines) - 1) - SUB_LH / 2
        for words, lw in lines:
            x = W / 2 - lw / 2
            for w, t, hl, wd in words:
                L = text_layer(w, "Bold", SUB_SIZE, "subgold" if hl else "white", 0, shadow=0.9, stroke=3)
                items.append((L, x + wd / 2, y, t))
                x += wd + space
            y += SUB_LH
        return items

    def draw(self, F, t):
        for (a, b, _), items in zip(self.events, self.cards):
            if a <= t <= b:
                out = 1 - prog(t, b - 0.15, 0.15)
                for L, x, y, tw in items:
                    p = prog(t, tw - 0.04, 0.14)
                    if p > 0:
                        e = ease_out_cubic(p)
                        over(F, L.get(1.0 + 0.06 * (1 - e)), x, y + 12 * (1 - e), min(1, p * 1.5) * out)


# ---------------------------------------------------------------- scenes ---
class Ad:
    def __init__(self):
        self.brand = Brand()
        self.bg = bg_charcoal()
        self.clips = {k: Clip(k) for k in TL.CLIPS}
        self.dust = Particles(110, 1)
        self.dust_front = Particles(26, 2, rise=(40, 90), sizes=(4.0, 7.0, 12.0), weights=(3, 2, 1))
        self.bursts = [Burst(SYNC["to_brand"], W / 2, H * 0.45, 70, seed=4),
                       Burst(SYNC["to_free"], W / 2, H * 0.40, 80, seed=5),
                       Burst(SYNC["free_hit"], W / 2, 400, 140, (400, 1700), seed=6, life=1.6),
                       Burst(SYNC["to_endcard"], W / 2, 700, 90, seed=8)]
        self.subs = Subtitles()
        self.free_layer, _ = extruded_layer("FREE", "Black", 300, depth=18, tracking=6)
        self.label_panel = rounded_panel(640, 250)
        self.result_panel = rounded_panel(960, 240)
        self.card = rounded_panel(980, 380, r=32, fill=(8, 6, 4, 175))
        self.pin = self._pin()
        dm = Image.new("L", (22, 22), 0)
        ImageDraw.Draw(dm).polygon([(11, 0), (22, 11), (11, 22), (0, 11)], fill=255)
        self.diamond = Layer(style(dm, "gold", shadow=0))
        self.target = TL.XRF_TARGET

    # -- shared pieces
    def _pin(self):
        s = 4
        img = Image.new("RGBA", (90 * s, 130 * s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([5 * s, 5 * s, 85 * s, 85 * s], fill=(222, 180, 96, 255))
        d.polygon([(12 * s, 62 * s), (78 * s, 62 * s), (45 * s, 125 * s)], fill=(222, 180, 96, 255))
        d.ellipse([28 * s, 28 * s, 62 * s, 62 * s], fill=(20, 15, 8, 255))
        img = img.resize((90, 130), Image.LANCZOS)
        m = img.split()[3]
        return Layer(style(m, "gold", bevel=True, shadow=0.6))

    # -- scene 1
    def hook(self, t):
        c = self.clips["ring"]
        F = c.at(t)
        F = push(F, 1.0 + 0.05 * t / 4.0, 0.5, 0.42)
        F = grade(F, (1.05, 1.0, 0.9), 1.1)
        e = 0.04 + 0.96 * smooth(0.05, 1.7, t) ** 1.3
        F *= e
        F *= VIGNETTE
        out = SYNC["to_brand"] - 0.35
        kinetic_words(F, t, SYNC["hook_line1"], [("IS", "white"), ("YOUR", "white"), ("GOLD", "white")],
                      "ExtraBold", 92, 300, tracking=4, out_t=out)
        # "PURE OR NOT?"  PURE slams in metallic gold, "OR NOT?" in bright white
        hit = SYNC["hook_hit"]
        layers, xs, total = words_line([("PURE", "gold"), ("OR", "white"), ("NOT?", "white")], "Black", 132, 2)
        sc = min(1.0, 990 / total)
        fade = 1 - prog(t, out, 0.35)
        p = prog(t, hit, 0.32)
        if p > 0:
            s = sc * (1.0 + 0.9 * (1 - ease_out_back(p, 1.4)))
            L = layers[0].get(s)
            q = prog(t, hit + 0.55, 0.9)
            if 0 < q < 1:
                L = sheen(L, -0.2 + 1.4 * ease_in_out(q), 0.09, 0.9)
            over(F, L, W / 2 + xs[0] * sc, 445, min(1, p * 3) * fade)
        for i in (1, 2):
            p2 = prog(t, hit + 0.12 + 0.08 * i, 0.45)
            if p2 > 0:
                e2 = ease_out_expo(p2)
                over(F, layers[i].get(sc), W / 2 + xs[i] * sc + 160 * (1 - e2), 445, min(1, p2 * 2) * fade)
        flash(F, 0.35 * max(0, 1 - (t - hit) / 0.25) if t >= hit else 0)
        return F

    # -- scene 2
    def brand_scene(self, t, tl):
        F = self.bg.copy()
        self.dust.draw(F, t, 0.9 * smooth(3.9, 4.4, t))
        draw_logo(F, self.brand, t, SYNC["logo_reveal"], W / 2, 700, 560, sheen_times=(7.2,))
        # ISLAMABAD'S 1ST with gold rules either side
        t1 = SYNC["islamabad_1st"]
        tw = letters(F, t, t1, "ISLAMABAD'S 1ST", "SemiBold", 50, 1010, "white", tracking=12)
        ln = 150 * ease_out_cubic(prog(t, t1 + 0.2, 0.6))
        hline(F, W / 2 - tw / 2 - 30 - ln / 2, 1012, ln, 2, 1)
        hline(F, W / 2 + tw / 2 + 30 + ln / 2, 1012, ln, 2, 1)
        t2 = SYNC["gold_testing_lab"]
        p = prog(t, t2, 0.6)
        if p > 0:
            L = text_layer("GOLD TESTING LAB", "ExtraBold", 104, "gold", 2, bevel=True, glow=0.25, max_w=980)
            A = L.get(1.0 + 0.08 * (1 - ease_out_cubic(p)))
            h = A.shape[0]
            reveal = ease_out_expo(p)
            rows = np.clip((np.arange(h) / h - (1 - reveal)) / 0.12, 0, 1).astype(np.float32)
            A = A * rows[:, None, None]   # rises up out of an invisible baseline mask
            q = prog(t, t2 + 0.9, 0.9)
            if 0 < q < 1:
                A = sheen(A, -0.2 + 1.4 * ease_in_out(q), 0.08, 0.95)
            over(F, A, W / 2, 1125 + 30 * (1 - reveal), 1.0)
        self.dust_front.draw(F, t, 0.5)
        return F

    # -- scene 3
    def xrf(self, t):
        c = self.clips["xrf"]
        F = c.at(t)
        F = grade(F, (1.04, 1.0, 0.93), 1.06)   # the Kling clip already carries the push-in
        F *= 0.9
        F *= VIGNETTE_SOFT
        darken_region(F, -200, 760, 0.40 * (1 - prog(t, SYNC["analysis_done"], 0.4)))
        darken_region(F, 1350, 2000, 0.55 * smooth(SYNC["analysis_done"] - 0.3, SYNC["analysis_done"], t))
        self.hud(F, t)
        return F

    def target_at(self, t):
        ct = t - TL.CLIPS["xrf"]["start"] + TL.CLIPS["xrf"]["src"]
        keys = self.target
        return [float(np.interp(ct, [k[0] for k in keys], [k[i] for k in keys])) for i in range(1, 5)]

    def hud(self, F, t):
        cx, cy, bw, bh = self.target_at(t)
        # reticle corners
        p = ease_out_expo(prog(t, 9.55, 0.6))
        if p > 0:
            grow = 1 + 0.35 * (1 - p)
            hw, hh = bw * grow / 2, bh * grow / 2
            alpha = p * (1 - prog(t, 16.6, 0.3))
            arm = 46
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = cx + sx * hw, cy + sy * hh
                    hline(F, x - sx * arm / 2, y, arm, 3, alpha)
                    vline(F, x, y - sy * arm / 2, arm, 3, alpha)
            # scan beam
            a, b = SYNC["scan_start"], SYNC["scan_end"]
            if a <= t <= b + 0.2:
                period = 1.1
                ph = ((t - a) % period) / period
                y = cy - hh + 2 * hh * ease_in_out(ph)
                k = alpha * (1 - prog(t, b, 0.2)) * smooth(a, a + 0.25, t)
                scan_beam(F, cx, y, 2 * hw - 20, k, cy - hh, cy + hh)
                dot_grid(F, cx, cy, 2 * hw - 30, 2 * hh - 30, 0.16 * k)
            # progress readout under reticle
            pr = prog(t, a, b - a)
            if a <= t < SYNC["analysis_done"] + 2.0:
                label = "SCANNING" if t < SYNC["analysis_done"] else "ANALYSIS COMPLETE"
                fill = "white" if t < SYNC["analysis_done"] else "gold"
                L = text_layer(label, "SemiBold", 30, fill, 6, shadow=0.8)
                over(F, L.get(1.0), cx, cy + hh + 44, alpha)
                if t < SYNC["analysis_done"]:
                    barw = 2 * hw - 40
                    hline(F, cx, cy + hh + 82, barw, 4, 0.30 * alpha, np.array([1, 1, 1], np.float32))
                    hline(F, cx - barw / 2 + barw * pr / 2, cy + hh + 82, max(2, barw * pr), 4, alpha)
        # spec panel, top-left: labels appear as the narrator names them
        p_panel = ease_out_cubic(prog(t, SYNC["label_purity"] - 0.3, 0.45))
        fade = 1 - prog(t, SYNC["analysis_done"] + 0.6, 0.4)
        if p_panel > 0:
            px, py = 60 + 320, 250
            rows = [("GOLD PURITY ANALYSIS", SYNC["label_purity"]), ("XRF TECHNOLOGY", SYNC["label_xrf"]),
                    ("PROFESSIONAL TESTING", SYNC["label_pro"])]
            panel = self.label_panel.get(1.0)
            over(F, panel, px, py, p_panel * fade * 0.95)
            for i, (txt, ts) in enumerate(rows):
                y = py - 78 + i * 78
                q = prog(t, ts, 0.45)
                if q <= 0:
                    continue
                e = ease_out_cubic(q)
                over(F, self.diamond.get(1.0), 110, y, e * fade)
                L = text_layer(txt, "SemiBold", 34, "white", 3, shadow=0.6)
                A = L.get(1.0).copy()
                wd = A.shape[1]
                m = np.clip((np.linspace(0, 1, wd) - (1.15 * e - 0.15)) / -0.12, 0, 1).astype(np.float32)
                A *= m[None, :, None]
                over(F, A, 145 + wd / 2 - 24, y, fade)
                hline(F, 145 + (wd - 48) * e / 2, y + 28, (wd - 48) * e, 2, 0.8 * fade)
        # result card (illustrative, no assay numbers are invented)
        q = prog(t, SYNC["analysis_done"] + 0.05, 0.5)
        if q > 0:
            self.result_card(F, t, q * (1 - prog(t, 16.55, 0.35)))

    def result_card(self, F, t, a):
        """Horizontal card under the hand; no assay values are shown or invented."""
        cx, cy = W / 2, 1600
        e = ease_out_cubic(min(1, a * 1.2))
        dy = 40 * (1 - e)
        over(F, self.result_panel.get(1.0), cx, cy + dy, a * 0.95)
        L = text_layer("XRF SPECTRUM", "Bold", 26, "gold", 6, shadow=0)
        over(F, L.get(1.0), cx - 200, cy - 82 + dy, a)
        for i, (txt, fill) in enumerate((("ANALYSIS", "gold"), ("COMPLETE", "white"))):
            Lr = text_layer(txt, "ExtraBold", 44, fill, 4, bevel=(fill == "gold"), shadow=0.4)
            over(F, Lr.get(1.0), cx + 270, cy - 30 + 54 * i + dy, a)
        x0, y_base, width = cx - 430, cy + 52 + dy, 460
        grow = ease_out_cubic(prog(t, SYNC["analysis_done"] + 0.25, 0.9))
        xs = np.linspace(0, 1, 160)
        spec = (0.95 * np.exp(-((xs - 0.30) / 0.025) ** 2) + 0.55 * np.exp(-((xs - 0.36) / 0.02) ** 2)
                + 0.22 * np.exp(-((xs - 0.62) / 0.02) ** 2) + 0.16 * np.exp(-((xs - 0.75) / 0.02) ** 2)
                + 0.04 * np.sin(xs * 70) ** 2)
        for i, v in enumerate(spec):
            hgt = v * 105 * grow
            if hgt > 1:
                vline(F, x0 + i * width / 160, y_base - hgt / 2, hgt, 3, a * 0.9)
        hline(F, x0 + width / 2, y_base + 2, width, 2, a * 0.6, np.array([1, 1, 1], np.float32))
        for lab, xn in (("Au", 0.30), ("Ag", 0.62), ("Cu", 0.75)):
            Lb = text_layer(lab, "SemiBold", 22, "white", 1, shadow=0.5)
            over(F, Lb.get(1.0), x0 + xn * width, y_base + 22, a * grow)
        note = text_layer("Illustrative visual", "Medium", 20, (0.75, 0.72, 0.66), 2, shadow=0)
        over(F, note.get(1.0), cx + 270, cy + 82 + dy, a * 0.8)

    # -- scene 4
    def free(self, t):
        c = self.clips["polki"]
        F = c.at(t)
        F = push(F, 1.0 + 0.03 * prog(t, 17, 6), 0.5, 0.6)
        F = grade(F, (1.05, 1.0, 0.9), 1.1)
        F *= VIGNETTE_SOFT
        dk = smooth(SYNC["gold_testing"] - 0.4, SYNC["gold_testing"] + 0.3, t)
        darken_region(F, -350, 1000, 0.62 * dk)
        self.dust.draw(F, t, 0.35)
        hit = SYNC["free_hit"]
        # GOLD TESTING (white, letter cascade)
        letters(F, t, SYNC["gold_testing"], "GOLD TESTING", "ExtraBold", 96, 640, "white", tracking=8,
                stagger=0.04)
        # FREE: the hero element, 3D gold, slams in on the bass hit
        p = prog(t, hit, 0.28)
        if p > 0:
            s = 1.0 + 1.3 * (1 - ease_out_back(p, 1.6))
            A = self.free_layer.get(s)
            if p >= 1:
                for ts in (hit + 0.45, hit + 1.9):
                    q = prog(t, ts, 0.85)
                    if 0 < q < 1:
                        A = sheen(A, -0.25 + 1.5 * ease_in_out(q), 0.07, 1.0, 0.25)
            glow_k = 0.5 + 0.3 * math.exp(-(t - hit) * 3)
            over(F, self.brand_glow_free(), W / 2, 380, glow_k * min(1, p * 2), add=True)
            over(F, A, W / 2 + 6, 385, min(1, p * 3))
        flash(F, 0.55 * max(0, 1 - (t - hit) / 0.3) if t >= hit else 0)
        # VISIT OUR LAB TODAY pill
        q = prog(t, SYNC["visit_lab"], 0.5)
        if q > 0:
            e = ease_out_cubic(q)
            pill = self.visit_pill()
            over(F, pill.get(0.9 + 0.1 * e), W / 2, 780 + 20 * (1 - e), e)
        return F

    @lru_cache(None)
    def brand_glow_free(self):
        m = Image.new("L", (1100, 700), 0)
        ImageDraw.Draw(m).ellipse([150, 150, 950, 550], fill=255)
        g = np.asarray(m.filter(ImageFilter.GaussianBlur(110)), np.float32) / 255.0 * 0.55
        return np.dstack([g[..., None] * GOLD, g]).astype(np.float32)

    @lru_cache(None)
    def visit_pill(self):
        L = text_layer("VISIT OUR LAB TODAY", "Bold", 44, "white", 6, shadow=0)
        tw, th = L.w - 48, 100
        img = Image.new("RGBA", (tw + 120, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([1, 1, tw + 118, th - 2], 48, fill=(10, 8, 5, 170), outline=(226, 186, 104, 255), width=3)
        txt = premul_to_img(L.get(1.0))
        img.alpha_composite(txt, ((img.width - txt.width) // 2, (th - txt.height) // 2))
        return Layer(img)

    # -- scene 5
    def location(self, t):
        c = self.clips["shop"]
        F = c.at(t)
        F = grade(F, (1.03, 1.0, 0.95), 1.06)
        F *= VIGNETTE
        darken_region(F, 1000, 1800, 0.55 * smooth(23.3, 23.9, t))
        t0 = SYNC["pin_drop"]
        # card
        q = prog(t, t0 + 0.25, 0.55)
        if q > 0:
            e = ease_out_cubic(q)
            over(F, self.card.get(1.0), W / 2, 1335 + 50 * (1 - e), e)
            rows = [("GOLD EXCHANGE LAB & JEWELRY", "Bold", 44, "gold"),
                    ("SHOP #2, BLOCK 13", "SemiBold", 36, "white"),
                    ("F-7 MARKAZ, JINNAH SUPER MARKET", "SemiBold", 36, "white"),
                    ("NEXT TO METRO SHOES, ISLAMABAD", "SemiBold", 36, "white")]
            for i, (txt, wgt, sz, fill) in enumerate(rows):
                r = prog(t, t0 + 0.55 + 0.26 * i, 0.45)
                if r > 0:
                    er = ease_out_cubic(r)
                    L = text_layer(txt, wgt, sz, fill, 3 if i else 2, bevel=(fill == "gold"), shadow=0.5,
                                   max_w=900)
                    y = 1205 + (0 if i == 0 else 40 + 72 * i) + 50 * (1 - e)
                    over(F, L.get(1.0), W / 2 + 40 * (1 - er), y, er)
                    if i == 0:
                        hline(F, W / 2, y + 44, 820 * er, 2, 0.7)
        # pin drop with bounce + pulse rings
        p = prog(t, t0, 0.55)
        if p > 0:
            y = 1075 - 260 * (1 - ease_out_back(p, 2.2))
            for k in range(3):
                rp = ((t - t0 - 0.45 - k * 0.5) % 1.5) / 1.5 if t > t0 + 0.45 + k * 0.5 else -1
                if rp >= 0:
                    ring(F, W / 2, 1137, 30 + 110 * rp, (1 - rp) * 0.7)
            over(F, self.pin.get(1.0), W / 2, y, min(1, p * 2.5))
        return F

    # -- scene 6
    def endcard(self, t):
        F = self.bg.copy()
        self.dust.draw(F, t, 0.9)
        t0 = SYNC["to_endcard"]
        draw_logo(F, self.brand, t, t0 + 0.05, W / 2, 680, 540, 1.18, sheen_times=(t0 + 1.6,))
        letters(F, t, t0 + 0.55, "GOLD EXCHANGE LAB & JEWELRY", "SemiBold", 42, 985, "gold", tracking=6,
                stagger=0.018, fit=1000)
        ln = 300 * ease_out_cubic(prog(t, t0 + 0.9, 0.6))
        hline(F, W / 2, 1040, ln, 2, 0.9)
        letters(F, t, t0 + 0.85, "FREE GOLD TESTING", "Medium", 30, 1088, "white", tracking=10, stagger=0.02)
        q = prog(t, SYNC["final_cta"], 0.5)
        if q > 0:
            e = ease_out_back(q, 1.5)
            pill = self.cta_pill()
            A = pill.get(0.85 + 0.15 * e)
            r = prog(t, SYNC["final_cta"] + 0.6, 0.9)
            if 0 < r < 1:
                A = sheen(A, -0.2 + 1.4 * ease_in_out(r), 0.1, 0.8)
            over(F, A, W / 2, 1215, min(1, q * 2))
        a = prog(t, SYNC["final_cta"] + 0.3, 0.5)
        if a > 0:
            L = text_layer("F-7 Markaz  ·  Next to Metro Shoes, Islamabad", "Medium", 30, (0.85, 0.82, 0.76), 1,
                           shadow=0.4, max_w=980)
            over(F, L.get(1.0), W / 2, 1335, a)
        self.dust_front.draw(F, t, 0.45)
        return F

    @lru_cache(None)
    def cta_pill(self):
        L = text_layer("VISIT US TODAY", "Black", 60, (0.06, 0.045, 0.02), 6, shadow=0)
        tw = L.w - 48
        img = Image.new("RGBA", (tw + 150, 124), (0, 0, 0, 0))
        g = gradient_gold(124, tw + 150)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, 123], 62, fill=255)
        m = np.asarray(mask, np.float32)[..., None] / 255
        base = Image.fromarray((np.dstack([g, m]) * 255).astype(np.uint8), "RGBA")
        txt = premul_to_img(L.get(1.0))
        base.alpha_composite(txt, ((base.width - txt.width) // 2, (124 - txt.height) // 2))
        return Layer(base)

    # ---------------------------------------------------------- timeline ---
    def frame(self, f):
        t = f / FPS
        S = SYNC
        if t < S["to_brand"] - 0.2:
            F = self.hook(t)
        elif t < S["to_brand"] + 0.15:            # flash cross-dissolve
            u = smooth(S["to_brand"] - 0.2, S["to_brand"] + 0.15, t)
            F = self.hook(t) * (1 - u) + self.brand_scene(t, None) * u
        elif t < S["to_xrf"] - 0.3:
            F = self.brand_scene(t, None)
        elif t < S["to_xrf"] + 0.1:               # zoom-through into the lab
            u = smooth(S["to_xrf"] - 0.3, S["to_xrf"] + 0.1, t)
            A = push(self.brand_scene(t, None), 1 + 0.35 * u, 0.5, 0.37)
            F = A * (1 - u) + self.xrf(t) * u
        elif t < S["to_free"] - 0.15:
            F = self.xrf(t)
        elif t < S["to_free"] + 0.12:
            u = smooth(S["to_free"] - 0.15, S["to_free"] + 0.12, t)
            F = self.xrf(t) * (1 - u) + self.free(t) * u
        elif t < S["to_location"] - 0.2:
            F = self.free(t)
        elif t < S["to_location"] + 0.15:
            u = smooth(S["to_location"] - 0.2, S["to_location"] + 0.15, t)
            F = self.free(t) * (1 - u) + self.location(t) * u
        elif t < S["to_endcard"] - 0.2:
            F = self.location(t)
        elif t < S["to_endcard"] + 0.15:
            u = smooth(S["to_endcard"] - 0.2, S["to_endcard"] + 0.15, t)
            F = self.location(t) * (1 - u) + self.endcard(t) * u
        else:
            F = self.endcard(t)

        # transitions: light streaks, flashes, bursts
        anamorphic(F, t, S["to_brand"], H * 0.45, 0.55, 0.9)
        anamorphic(F, t, S["to_xrf"], H * 0.37, 0.45, 0.7)
        anamorphic(F, t, S["to_free"], H * 0.42, 0.5, 1.0)
        anamorphic(F, t, S["to_location"], H * 0.30, 0.45, 0.7)
        anamorphic(F, t, S["to_endcard"], H * 0.36, 0.55, 0.9)
        for k, amt in (("to_brand", 0.45), ("to_xrf", 0.3), ("to_free", 0.5), ("to_location", 0.3),
                       ("to_endcard", 0.45)):
            d = abs(t - S[k])
            if d < 0.18:
                flash(F, amt * (1 - d / 0.18) ** 2)
        for b in self.bursts:
            b.draw(F, t)
        dx, dy = 0, 0
        for k, amp in (("hook_hit", 10), ("free_hit", 22), ("to_endcard", 8)):
            ox, oy = shake_offset(t, S[k], amp)
            dx, dy = dx + ox, dy + oy
        F = apply_shake(F, dx, dy)
        grain(F, f)
        clean = F
        subbed = F.copy()
        self.subs.draw(subbed, t)
        return clean, subbed


def vline(F, cx, cy, length, thick=2, alpha=1.0, color=GOLD):
    if length < 1 or alpha <= 0:
        return
    y0, y1 = max(0, int(cy - length / 2)), min(H, int(cy + length / 2))
    x0 = int(cx - thick / 2)
    if y1 > y0 and 0 <= x0 < W - thick:
        F[y0:y1, x0:x0 + thick] = F[y0:y1, x0:x0 + thick] * (1 - alpha) + color * alpha


def scan_beam(F, cx, y, width, k, top, bottom):
    if k <= 0:
        return
    x0, x1 = max(0, int(cx - width / 2)), min(W, int(cx + width / 2))
    ys = np.arange(max(0, int(top)), min(H, int(bottom)))
    if len(ys) == 0 or x1 <= x0:
        return
    d = ys - y
    prof = (np.exp(-(d / 2.5) ** 2) * 1.3 + np.where(d < 0, np.exp(d / 70), 0) * 0.28).astype(np.float32)
    xs = np.linspace(0, 1, x1 - x0, dtype=np.float32)
    hor = np.minimum(1, np.minimum(xs, 1 - xs) * 10)
    F[ys[0]:ys[-1] + 1, x0:x1] += (prof[:, None, None] * hor[None, :, None]) * GOLD * k


def dot_grid(F, cx, cy, w, h, k, step=34):
    if k <= 0:
        return
    for gy in np.arange(cy - h / 2, cy + h / 2, step):
        for gx in np.arange(cx - w / 2, cx + w / 2, step):
            over(F, SPRITES[1.5], gx, gy, k, add=True)


def ring(F, cx, cy, r, a):
    if a <= 0 or r <= 1:
        return
    s = int(r + 6)
    y, x = np.mgrid[-s:s + 1, -s:s + 1].astype(np.float32)
    d = np.sqrt(x * x + (y * 3.2) ** 2)
    m = np.exp(-((d - r) / 2.2) ** 2) * a
    over(F, np.dstack([m[..., None] * GOLD, m]).astype(np.float32), cx, cy, 1.0, add=True)


# ------------------------------------------------------------------ main ---
def encoder(path):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p", "-g", "30", "-bf", "2", "-movflags", "+faststart", path],
        stdin=subprocess.PIPE)


def to8(F):
    return (np.clip(F, 0, 1) * 255 + 0.5).astype(np.uint8)


def render_range(a, b, tag):
    ad = Ad()
    out = os.path.join(WORK, "out")
    os.makedirs(out, exist_ok=True)
    e_clean = encoder(os.path.join(out, f"_seg_clean_{tag}.mp4"))
    e_subs = encoder(os.path.join(out, f"_seg_subs_{tag}.mp4"))
    for f in range(a, b):
        clean, subbed = ad.frame(f)
        e_clean.stdin.write(to8(clean).tobytes())
        e_subs.stdin.write(to8(subbed).tobytes())
        if f % 30 == 0:
            print(f"[{tag}] frame {f}", flush=True)
    for e in (e_clean, e_subs):
        e.stdin.close()
        e.wait()


def main():
    global WORK
    args = sys.argv[1:]
    WORK = args[0] if args else "."
    opts = dict(zip(args[1::2], args[2::2]))
    if "--stills" in opts:
        ad = Ad()
        os.makedirs(os.path.join(WORK, "stills"), exist_ok=True)
        for ts in opts["--stills"].split(","):
            f = int(round(float(ts) * FPS))
            _, s = ad.frame(f)
            Image.fromarray(to8(s)).save(os.path.join(WORK, "stills", f"f{f:04d}.png"))
            print("still", ts)
        return
    if "--frames" in opts:
        a, b = map(int, opts["--frames"].split(":"))
        render_range(a, b, opts.get("--tag", f"{a:04d}"))
        return
    workers = int(opts.get("--workers", "1"))
    for name in TL.CLIPS:           # decode caches once, before forking workers
        Clip(name)
    bounds = np.linspace(0, N_FRAMES, workers + 1).astype(int)
    procs = []
    for i in range(workers):
        procs.append(subprocess.Popen([sys.executable, __file__, WORK, "--frames",
                                       f"{bounds[i]}:{bounds[i + 1]}", "--tag", f"{i:02d}"]))
    rc = [p.wait() for p in procs]
    if any(rc):
        sys.exit("worker failed")
    out = os.path.join(WORK, "out")
    for kind in ("clean", "subs"):
        lst = os.path.join(out, f"_{kind}.txt")
        with open(lst, "w") as fh:
            for i in range(workers):
                fh.write(f"file '_seg_{kind}_{i:02d}.mp4'\n")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(out, f"_video_{kind}.mp4")], check=True)
    print("render ok")


if __name__ == "__main__":
    main()
