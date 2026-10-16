"""Frame-accurate master timeline for the Gold Exchange Lab 30-second ad.

Every other script (render.py, audio.py) imports this module, so changing a
time here moves the picture, the sound design and the subtitles together.
All times are seconds on the 30.000 s master timeline (30 fps -> 900 frames).
"""

W, H = 1080, 1920
FPS = 30
DURATION = 30.0
N_FRAMES = int(DURATION * FPS)

# ---------------------------------------------------------------- scenes ---
SCENES = {
    "hook":     (0.0, 4.0),    # ring macro + "IS YOUR GOLD / PURE OR NOT?"
    "brand":    (4.0, 9.0),    # particles + logo + "ISLAMABAD'S 1ST / GOLD TESTING LAB"
    "xrf":      (9.0, 17.0),   # XRF testing + scan HUD
    "free":     (17.0, 23.0),  # polki set + "FREE GOLD TESTING"
    "location": (23.0, 26.7),  # shop front + address card
    "endcard":  (26.7, 30.0),  # logo + VISIT US TODAY
}

# Source clips (Higgsfield / Kling 3.0 image-to-video renders) and the source
# time that lines up with the start of their scene.
CLIPS = {
    "ring":  {"file": "clips/ring.mp4",  "start": 0.0,  "src": 0.0},
    "xrf":   {"file": "clips/xrf.mp4",   "start": 9.0,  "src": 0.0},
    "polki": {"file": "clips/polki.mp4", "start": 17.0, "src": 0.0},
    "shop":  {"file": "clips/shop.mp4",  "start": 23.0, "src": 0.25},
}

# Scan-reticle track over the jewelry in the XRF clip, measured from frames
# of the final Kling render: (clip_time, centre_x, centre_y, box_w, box_h).
XRF_TARGET = [
    (0.0, 795, 1250, 430, 330),
    (4.0, 757, 1207, 525, 345),
    (8.0, 720, 1050, 690, 540),
]

# ------------------------------------------------------------ voice-over ---
# ElevenLabs v4 (via Higgsfield), voice "Mark" (preset 27c04473-...).
# "at" = where the line file starts on the master timeline,
# "tempo" = time-stretch factor applied without pitch change (>1 = faster).
VO = [
    {"file": "vo/line1.mp3", "at": 0.55,  "tempo": 1.00},
    {"file": "vo/line2.mp3", "at": 4.45,  "tempo": 1.00},
    {"file": "vo/line3.mp3", "at": 9.45,  "tempo": 1.00},
    {"file": "vo/line4.mp3", "at": 17.15, "tempo": 1.00},
    {"file": "vo/line5.mp3", "at": 23.15, "tempo": 1.07},
]

URDU_SCRIPT = [
    "کیا آپ کا سونا واقعی خالص ہے؟",
    "اسلام آباد کی پہلی گولڈ ٹیسٹنگ لیب!",
    "اب اپنے سونے کی پیورٹی چیک کروائیں، ایڈوانسڈ ایکس آر ایف ٹیکنالوجی کے ساتھ۔",
    "اور سب سے خاص بات، گولڈ ٹیسٹنگ بالکل فری!",
    "وزٹ کریں گولڈ ایکسچینج لیب، ایف سیون مرکز اسلام آباد، میٹرو شوز کے ساتھ۔",
]

# Word timings (seconds inside each line file, before tempo) measured with
# faster-whisper word timestamps on the generated narration.
# Each card = one subtitle event (max two lines on screen).
# Words wrapped in * are highlighted in metallic gold.
SUB_CARDS = [
    (0, [("Kya", 0.00), ("Aap", 0.28), ("Ka", 0.48), ("*Sona*", 0.66),
         ("Waqai", 1.04), ("*Khalis*", 1.64), ("Hai?", 2.10)], 2.42),
    (1, [("Islamabad", 0.00), ("Ki", 0.74), ("Pehli", 0.92), ("*Gold*", 1.48),
         ("*Testing*", 1.84), ("*Lab!*", 2.34)], 2.62),
    (2, [("Ab", 0.00), ("Apne", 0.20), ("Sonay", 0.50), ("Ki", 0.88),
         ("*Purity*", 1.00), ("Check", 1.52), ("Karwayein,", 1.74)], 2.18),
    (2, [("Advanced", 2.18), ("*XRF*", 3.22), ("*Technology*", 3.84),
         ("Ke", 4.46), ("Saath.", 4.54)], 5.20),
    (3, [("Aur", 0.00), ("Sab", 0.16), ("Se", 0.52), ("Khaas", 0.70),
         ("Baat,", 1.06)], 1.40),
    (3, [("*Gold*", 1.40), ("*Testing*", 2.18), ("Bilkul", 2.72),
         ("*FREE!*", 3.26)], 3.52),
    (4, [("Visit", 0.00), ("Karein", 0.40), ("*Gold*", 0.76),
         ("*Exchange*", 1.22), ("*Lab,*", 1.70)], 2.02),
    (4, [("*F-7*", 2.02), ("*Markaz*", 3.14), ("Islamabad,", 3.68)], 4.46),
    (4, [("Metro", 4.46), ("Shoes", 5.28), ("Ke", 5.58), ("Saath.", 5.70)], 6.24),
]


def vo_time(line_idx, t_in_file):
    v = VO[line_idx]
    return v["at"] + t_in_file / v["tempo"]


def subtitle_events():
    """Return [(start, end, [(word, t, highlighted), ...]), ...] on master time."""
    events = []
    for i, (line, words, end) in enumerate(SUB_CARDS):
        ws = [(w.strip("*"), vo_time(line, t), w.startswith("*")) for w, t in words]
        start = ws[0][1] - 0.08
        stop = vo_time(line, end) + 0.35
        # never overlap the next card
        if i + 1 < len(SUB_CARDS):
            nl, nw, _ = SUB_CARDS[i + 1]
            stop = min(stop, vo_time(nl, nw[0][1]) - 0.1)
        events.append((start, stop, ws))
    return events


# ------------------------------------------------------- key sync points ---
# Shared by the picture and the sound design so hits land on the same frame.
SYNC = {
    "hook_line1": 0.45,     # "IS YOUR GOLD"
    "hook_hit": 2.05,       # "PURE OR NOT?" + bass impact
    "to_brand": 4.0,
    "logo_reveal": 4.25,
    "islamabad_1st": vo_time(1, 0.0) + 0.75,
    "gold_testing_lab": vo_time(1, 1.48),
    "to_xrf": 9.0,
    "label_purity": vo_time(2, 1.00),
    "label_xrf": vo_time(2, 3.22),
    "label_pro": 14.05,
    "scan_start": 10.0,
    "scan_end": 14.9,
    "analysis_done": 15.1,   # after the line-3 subtitle has left the screen
    "to_free": 17.0,
    "gold_testing": vo_time(3, 1.40),
    "free_hit": vo_time(3, 3.26),
    "visit_lab": vo_time(3, 3.26) + 0.9,
    "to_location": 23.0,
    "pin_drop": 23.35,
    "to_endcard": 26.7,
    "final_cta": 27.6,
}


def fmt_srt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path):
    lines = []
    for n, (a, b, ws) in enumerate(subtitle_events(), 1):
        lines += [str(n), f"{fmt_srt(a)} --> {fmt_srt(b)}", " ".join(w for w, _, _ in ws), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import sys
    for k, v in SYNC.items():
        print(f"{k:18s} {v:6.2f}s  frame {round(v * FPS)}")
    for a, b, ws in subtitle_events():
        print(f"{a:6.2f}-{b:6.2f}  {' '.join(w for w, _, _ in ws)}")
    if len(sys.argv) > 1:
        write_srt(sys.argv[1])
