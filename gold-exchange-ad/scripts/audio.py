"""Original score + sound design + voice-over mix for the Gold Exchange Lab ad.

Everything musical here is synthesised from scratch (no samples, no third-party
loops), so the soundtrack is original and royalty-free for the client.

Outputs (48 kHz, 24-bit WAV) into out/:
  GOLD_EXCHANGE_LAB_VOICEOVER.wav  narration on the 30 s timeline
  stem_music.wav / stem_sfx.wav    unducked stems for re-editing
  mix_premaster.wav                ducked mix before loudness normalisation
The final loudness pass (-14 LUFS integrated, -1.5 dBTP) is done by
build.sh with ffmpeg's two-pass loudnorm.

Usage: python3 audio.py <work_dir>
"""
import os
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(__file__))
from timeline import DURATION, SYNC, VO  # noqa: E402

SR = 48000
N = int(DURATION * SR)
BPM = 120
BEAT = 60 / BPM
rng = np.random.default_rng(7)


# ------------------------------------------------------------- helpers ---
def t_axis(n):
    return np.arange(n) / SR


def place(buf, x, at, gain=1.0):
    i = int(round(at * SR))
    if i >= len(buf):
        return
    if i < 0:
        x, i = x[-i:], 0
    n = min(len(x), len(buf) - i)
    buf[i:i + n] += gain * x[:n]


def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output="sos")


def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x)


def env_ad(n, a, d, curve=4.0):
    t = t_axis(n)
    e = np.where(t < a, t / max(a, 1e-4), np.exp(-(t - a) * curve / max(d, 1e-4)))
    return e


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(freq, n, phase=0.0):
    ph = (phase + np.cumsum(np.full(n, freq) / SR)) % 1.0
    return 2 * ph - 1


def reverb(x, seconds=2.2, mix=0.25, pre=0.02):
    n = int(seconds * SR)
    t = t_axis(n)
    ir = rng.standard_normal(n) * np.exp(-t * 6.0 / seconds)
    ir = filt(ir, "lowpass", 6000)
    ir[: int(pre * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return (1 - mix) * x + mix * wet * 0.6


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


T = t_axis(N)

# ---------------------------------------------------------- harmony ---
# D minor, luxury / cinematic. One chord per bar (2 s).
CHORDS = {
    "Dm": [50, 57, 62, 65, 69], "Bb": [46, 53, 58, 62, 65], "F": [41, 53, 57, 60, 65],
    "C": [48, 55, 60, 64, 67], "Gm": [43, 55, 58, 62, 67], "A": [45, 52, 57, 61, 64],
}
PROG = ["Dm", "Dm", "Bb", "C", "Dm", "Bb", "F", "C", "Dm", "Bb", "Gm", "A",
        "Bb", "C", "Dm"]   # 15 bars x 2 s = 30 s


def chord_at(t):
    return PROG[min(int(t // 2), len(PROG) - 1)]


# ------------------------------------------------------------- music ---
def pad():
    out = np.zeros(N)
    bar = int(2 * SR)
    for b, name in enumerate(PROG):
        n = bar + int(0.6 * SR)
        seg = np.zeros(n)
        for m in CHORDS[name][1:]:
            for det in (-0.07, 0.0, 0.08):
                seg += saw(midi(m) * 2 ** (det / 12), n, rng.random())
        e = np.minimum(1, t_axis(n) / 0.35) * np.minimum(1, (n - np.arange(n)) / (0.6 * SR))
        place(out, seg * e / 12, b * 2.0)
    # filter opens with the energy of the ad
    lo = filt(out, "lowpass", 700, 2)
    hi = filt(out, "lowpass", 2600, 2)
    open_ = smoothstep(8.0, 9.5, T) * 0.6 + smoothstep(16.0, 17.0, T) * 0.4
    open_ *= 1 - 0.6 * smoothstep(23.0, 23.5, T) + 0.6 * smoothstep(26.6, 26.8, T)
    x = lo * (1 - open_) + hi * open_
    level = 0.35 + 0.35 * smoothstep(3.5, 5.0, T) + 0.3 * smoothstep(16.5, 17.0, T)
    return x * level


def bass():
    out = np.zeros(N)
    # whole notes 4-9 s, driving 8ths 9-23 s, whole notes after
    t = 4.0
    while t < 29.0:
        root = CHORDS[chord_at(t)][0] - 12
        driving = 9.0 <= t < 23.0
        step = BEAT / 2 if driving else 2.0
        n = int((step if driving else 2.2) * SR)
        f = midi(root)
        x = np.sin(2 * np.pi * f * t_axis(n)) + 0.3 * np.sin(4 * np.pi * f * t_axis(n))
        x = np.tanh(1.6 * x)
        e = env_ad(n, 0.005, step * (0.9 if driving else 2.0), 3.0 if driving else 1.2)
        place(out, x * e, t, 0.33 if driving else 0.26)
        t += step
    return filt(out, "lowpass", 420)


def kick_hit(level=1.0):
    n = int(0.45 * SR)
    tt = t_axis(n)
    f = 45 + 110 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sin(ph) * np.exp(-tt * 7.5)
    click = filt(rng.standard_normal(n), "highpass", 2500) * np.exp(-tt * 180) * 0.25
    return (np.tanh(2 * x) + click) * level


def drums():
    kick = np.zeros(N)
    hats = np.zeros(N)
    clap = np.zeros(N)
    kick_sc = np.zeros(N)   # sidechain envelope for bass/pad pumping
    k = kick_hit()
    beat = 9.0
    while beat < 26.6:
        if 23.0 <= beat:              # half-time under the address
            on = (beat - 23.0) % (2 * BEAT) < 1e-6
        elif 16.0 <= beat < 17.0:     # drop out before the FREE section
            on = False
        else:
            on = True
        if on:
            place(kick, k, beat, 0.9)
            place(kick_sc, np.exp(-t_axis(int(0.3 * SR)) * 12), beat)
        beat += BEAT
    # 8th-note off-beat hats 9-23 s
    n = int(0.06 * SR)
    hat = filt(rng.standard_normal(n), "highpass", 7000) * np.exp(-t_axis(n) * 70)
    t = 9.0 + BEAT / 2
    while t < 23.0:
        if not (16.0 <= t < 17.0):
            place(hats, hat, t, 0.16 if t < 17 else 0.22)
        t += BEAT
    # claps on 2 & 4 in the FREE section
    n = int(0.25 * SR)
    c = filt(rng.standard_normal(n), "bandpass", [900, 3500]) * np.exp(-t_axis(n) * 22)
    t = 17.0 + BEAT
    while t < 23.0:
        place(clap, c, t, 0.35)
        t += 2 * BEAT
    # snare roll build 15.5-17
    t, step = 15.5, BEAT / 2
    while t < 16.98:
        place(clap, c, t, 0.12 + 0.25 * (t - 15.5) / 1.5)
        step = max(BEAT / 8, step * 0.82)
        t += step
    return kick, hats, reverb(clap, 1.2, 0.3), kick_sc


def pluck_arp():
    """Harp / pizzicato-like arpeggio, the 'elegant' layer."""
    out = np.zeros(N)
    t = 4.0
    i = 0
    pattern = [0, 2, 1, 3, 2, 4, 3, 2]
    while t < 26.6:
        notes = CHORDS[chord_at(t)][1:]
        m = notes[pattern[i % 8] % len(notes)] + 12
        n = int(0.7 * SR)
        f = midi(m)
        tt = t_axis(n)
        x = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(4 * np.pi * f * tt)
             + 0.12 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt * 7)
        gain = 0.10 if t < 9 else 0.07
        if 16.0 <= t < 17.0:
            gain = 0.04
        place(out, x, t, gain)
        t += BEAT / 2
        i += 1
    return reverb(out, 2.5, 0.35)


def strings():
    """Saw-ensemble 'string' swell for the promotional peak."""
    out = np.zeros(N)
    for b, name in enumerate(PROG):
        t0 = b * 2.0
        if not (16.0 <= t0 < 24.0 or t0 >= 26.0):
            continue
        n = int(2.4 * SR)
        seg = np.zeros(n)
        for m in CHORDS[name][2:]:
            for det in (-0.1, 0.1):
                vib = 1 + 0.003 * np.sin(2 * np.pi * 5.2 * t_axis(n) + rng.random() * 6)
                ph = np.cumsum(midi(m + 12) * 2 ** (det / 12) * vib) / SR
                seg += 2 * (ph % 1) - 1
        e = np.minimum(1, t_axis(n) / 0.5) * np.minimum(1, (n - np.arange(n)) / (0.5 * SR))
        place(out, seg * e / 10, t0)
    out = filt(out, "lowpass", 3800)
    return reverb(out, 3.0, 0.4) * (0.55 * smoothstep(16.5, 17.2, T) + 0.4 * smoothstep(26.6, 27.0, T))


def final_chord():
    n = int(3.3 * SR)
    seg = np.zeros(n)
    for m in [38, 50, 57, 62, 65, 69, 74]:
        for det in (-0.06, 0.06):
            seg += saw(midi(m) * 2 ** (det / 12), n, rng.random())
    seg = filt(seg, "lowpass", 2200)
    e = np.minimum(1, t_axis(n) / 0.02) * np.exp(-t_axis(n) * 0.9)
    return reverb(seg * e / 14, 3.0, 0.45)


# ------------------------------------------------------------- SFX ---
def impact(size=1.0):
    n = int(3.0 * SR)
    tt = t_axis(n)
    f = 30 + 70 * np.exp(-tt * 9)
    boom = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.8)
    thump = filt(rng.standard_normal(n), "lowpass", 900) * np.exp(-tt * 14) * 0.6
    crack = filt(rng.standard_normal(n), "highpass", 3000) * np.exp(-tt * 45) * 0.25
    x = np.tanh(1.8 * (boom + thump)) + crack
    return reverb(x, 2.8, 0.3) * size


def riser(length, top=6000):
    n = int(length * SR)
    tt = t_axis(n)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    hop = 1024
    for i in range(0, n, hop):
        fc = 300 + (top - 300) * (i / n) ** 2
        blk = noise[max(0, i - 4096): i + hop]
        y = signal.sosfilt(sos("bandpass", [fc * 0.7, min(fc * 1.4, SR / 2 - 100)]), blk)
        out[i:i + hop] = y[-min(hop, n - i):]
    tone = np.sin(2 * np.pi * np.cumsum(110 + 440 * (tt / length) ** 2) / SR) * 0.15
    e = (tt / length) ** 2.2
    return (out * 0.6 + tone) * e


def whoosh(length=0.7, up=True):
    n = int(length * SR)
    tt = t_axis(n)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    hop = 512
    for i in range(0, n, hop):
        p = i / n
        fc = (400 + 4500 * p) if up else (4800 - 4200 * p)
        blk = noise[max(0, i - 2048): i + hop]
        y = signal.sosfilt(sos("bandpass", [fc * 0.6, min(fc * 1.6, SR / 2 - 100)]), blk)
        out[i:i + hop] = y[-min(hop, n - i):]
    e = np.sin(np.pi * np.clip(tt / length, 0, 1)) ** 1.5
    return out * e


def shimmer(length=1.2, density=26, base=2400):
    """Metallic gold shimmer: scattered inharmonic bell partials."""
    n = int(length * SR)
    out = np.zeros(n + SR)
    for _ in range(density):
        at = rng.random() * length * 0.8
        f = base * (1 + rng.random() * 2.5)
        m = int(0.6 * SR)
        tt = t_axis(m)
        x = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * f * 2.76 * tt)) * np.exp(-tt * 9)
        i = int(at * SR)
        out[i:i + m] += x * (0.4 + 0.6 * rng.random())
    return reverb(out[:n + SR // 2] / np.sqrt(density), 2.0, 0.45)


def blip(f=1800, length=0.09):
    n = int(length * SR)
    tt = t_axis(n)
    return np.sin(2 * np.pi * f * tt) * np.exp(-tt * 40) * np.minimum(1, tt / 0.002)


def scanner():
    """XRF scan bed: soft digital hum, sweeps and data ticks (10.0-14.4 s)."""
    a, b = SYNC["scan_start"], SYNC["scan_end"]
    n = int((b - a + 0.5) * SR)
    tt = t_axis(n)
    hum = (np.sin(2 * np.pi * 180 * tt) * 0.5 + np.sin(2 * np.pi * 360.7 * tt) * 0.25)
    hum *= 0.5 + 0.5 * np.sin(2 * np.pi * 1.1 * tt) ** 2
    sweep_f = 900 + 500 * np.sin(2 * np.pi * tt / 1.1)
    sweep = np.sin(2 * np.pi * np.cumsum(sweep_f) / SR) * 0.18
    e = smoothstep(0, 0.3, tt) * (1 - smoothstep(b - a - 0.2, b - a + 0.4, tt))
    out = (hum * 0.25 + sweep) * e
    out = filt(out, "highpass", 120)
    for k in range(int((b - a) / 0.11)):
        if rng.random() < 0.55:
            place(out, blip(2600 + 1400 * rng.random(), 0.03), k * 0.11, 0.18)
    return out


def chime():
    return np.concatenate([blip(1320, 0.18), blip(1760, 0.35)]) * 0.9


def pop():
    n = int(0.25 * SR)
    tt = t_axis(n)
    f = 300 + 900 * np.exp(-tt * 40)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 18)


# ------------------------------------------------------------ voice ---
def load_vo(work):
    vo = np.zeros(N)
    for v in VO:
        src = os.path.join(work, v["file"])
        tmp = os.path.join(work, "out", f"_vo_{os.path.basename(src)}.wav")
        af = f"aresample={SR},atempo={v['tempo']:.4f},highpass=f=80,acompressor=threshold=-20dB:ratio=3:attack=5:release=80"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af,
                        "-ac", "1", "-ar", str(SR), tmp], check=True)
        sr, x = wavfile.read(tmp)
        x = x.astype(np.float64) / 32768.0
        place(vo, x, v["at"])
        os.remove(tmp)
    peak = np.max(np.abs(vo)) + 1e-9
    return vo * (0.89 / peak)


def duck_env(vo, depth_db=-10.0):
    e = np.abs(vo)
    e = signal.sosfiltfilt(sos("lowpass", 6), e)
    e = np.clip(e / (np.max(e) * 0.25 + 1e-9), 0, 1)
    # attack fast, release slow
    out = np.zeros_like(e)
    a, r = np.exp(-1 / (0.02 * SR)), np.exp(-1 / (0.35 * SR))
    acc = 0.0
    for i in range(0, len(e), 64):          # block-rate follower (fast enough)
        x = e[i]
        acc = x + (acc - x) * (a ** 64 if x > acc else r ** 64)
        out[i:i + 64] = acc
    return 10 ** (depth_db * out / 20)


def write(path, x):
    """Write 48 kHz / 24-bit PCM WAV (via a float32 temp + ffmpeg)."""
    tmp = path + ".f32.wav"
    wavfile.write(tmp, SR, np.clip(x, -1, 1).astype(np.float32))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:a", "pcm_s24le", path], check=True)
    os.remove(tmp)


def main(work):
    os.makedirs(os.path.join(work, "out"), exist_ok=True)

    kick, hats, clap, kick_sc = drums()
    pump = 1 - 0.45 * kick_sc
    music = (pad() * pump + bass() * pump + kick + hats + clap
             + pluck_arp() + strings())
    music[: int(SYNC["to_endcard"] * SR)] *= 1.0
    place(music, final_chord(), SYNC["to_endcard"], 0.9)
    # opening: start mysterious, nearly silent for the first beat
    music *= 0.25 + 0.75 * smoothstep(0.0, 1.6, T)
    music *= 1 - smoothstep(29.3, 30.0, T)

    sfx = np.zeros(N)
    place(sfx, shimmer(1.4, 22, 2600), 0.25, 0.35)
    place(sfx, riser(1.5), SYNC["hook_hit"] - 1.5, 0.25)
    place(sfx, impact(1.0), SYNC["hook_hit"], 0.55)
    place(sfx, riser(1.6), SYNC["to_brand"] - 1.6, 0.3)
    place(sfx, whoosh(0.8), SYNC["to_brand"] - 0.4, 0.55)
    place(sfx, impact(0.6), SYNC["to_brand"], 0.4)
    place(sfx, shimmer(1.6, 30, 2200), SYNC["logo_reveal"], 0.45)
    place(sfx, whoosh(0.6, up=False), SYNC["islamabad_1st"] - 0.15, 0.25)
    place(sfx, shimmer(1.0, 16, 3000), SYNC["gold_testing_lab"], 0.3)
    place(sfx, riser(1.2), SYNC["to_xrf"] - 1.2, 0.25)
    place(sfx, whoosh(0.7), SYNC["to_xrf"] - 0.35, 0.5)
    place(sfx, scanner(), SYNC["scan_start"], 0.55)
    for k in ("label_purity", "label_xrf", "label_pro"):
        place(sfx, blip(2100, 0.07), SYNC[k], 0.35)
        place(sfx, blip(2800, 0.06), SYNC[k] + 0.07, 0.25)
    place(sfx, chime(), SYNC["analysis_done"], 0.35)
    place(sfx, riser(1.5, 8000), SYNC["to_free"] - 1.5, 0.35)
    place(sfx, whoosh(0.8), SYNC["to_free"] - 0.4, 0.55)
    place(sfx, impact(0.9), SYNC["to_free"], 0.5)
    place(sfx, shimmer(1.4, 24, 2400), SYNC["to_free"] + 0.3, 0.35)
    place(sfx, whoosh(0.5, up=False), SYNC["gold_testing"] - 0.1, 0.3)
    place(sfx, riser(0.9, 7000), SYNC["free_hit"] - 0.9, 0.3)
    place(sfx, impact(1.25), SYNC["free_hit"], 0.75)
    place(sfx, shimmer(1.8, 36, 2600), SYNC["free_hit"] + 0.05, 0.4)
    place(sfx, whoosh(0.6), SYNC["visit_lab"] - 0.2, 0.25)
    place(sfx, whoosh(0.8), SYNC["to_location"] - 0.4, 0.5)
    place(sfx, pop(), SYNC["pin_drop"] + 0.3, 0.45)
    for k in range(4):
        place(sfx, blip(1500 + 200 * k, 0.05), SYNC["pin_drop"] + 0.8 + 0.28 * k, 0.15)
    place(sfx, whoosh(0.9), SYNC["to_endcard"] - 0.45, 0.5)
    place(sfx, impact(1.1), SYNC["to_endcard"], 0.6)
    place(sfx, shimmer(2.0, 40, 2300), SYNC["to_endcard"] + 0.15, 0.45)
    place(sfx, blip(1760, 0.2), SYNC["final_cta"], 0.25)

    vo = load_vo(work)
    duck = duck_env(vo, -10.0)
    mix = music * 0.55 * duck + sfx * 0.6 * (0.55 + 0.45 * duck) + vo * 1.0
    peak = np.max(np.abs(mix))
    if peak > 0.98:
        mix *= 0.98 / peak

    out = os.path.join(work, "out")
    write(os.path.join(out, "GOLD_EXCHANGE_LAB_VOICEOVER.wav"), vo)
    write(os.path.join(out, "stem_music.wav"), music * 0.55)
    write(os.path.join(out, "stem_sfx.wav"), sfx * 0.6)
    write(os.path.join(out, "mix_premaster.wav"), mix)
    print("audio ok, premaster peak", round(float(np.max(np.abs(mix))), 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
