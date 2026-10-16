"""Build GOLD_EXCHANGE_LAB_STORYBOARD.pdf from frames of the final render.

Usage: python3 storyboard.py <work_dir>
"""
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
import timeline as TL  # noqa: E402
from render import font  # noqa: E402

BOARD = [
    ("01  THE HOOK", "00:00-00:04", [1.0, 2.6, 3.5],
     "Ring emerges from darkness (Kling 3.0 image-to-video from the client's ring reference). "
     "Kinetic type: IS YOUR GOLD / PURE (metallic gold) OR NOT? (white) slams on the bass impact.",
     "Kya aap ka sona waqai khalis hai?", "Deep impact, metallic shimmer, riser"),
    ("02  BRAND", "00:04-00:09", [4.6, 5.6, 7.6],
     "Gold particles on charcoal; the client's actual logo revealed with a light-sweep wipe (logo pixels "
     "untouched). ISLAMABAD'S 1ST (letter cascade) and GOLD TESTING LAB (bevelled gold, rising mask).",
     "Islamabad ki pehli Gold Testing Lab!", "Whoosh, shimmer, music builds"),
    ("03  XRF TESTING", "00:09-00:17", [10.6, 13.0, 15.6],
     "The client's real yellow handheld XRF analyzer testing a gold bridal set (photo edited with Nano "
     "Banana 2, animated with Kling 3.0). Overlay HUD: reticle, scan beam, labels synced to narration, "
     "illustrative spectrum card with no invented assay values.",
     "Ab apne sonay ki purity check karwayein, advanced XRF technology ke saath.",
     "Scanner bed, digital ticks, UI blips, completion chime"),
    ("04  FREE GOLD TESTING", "00:17-00:23", [17.8, 20.9, 22.2],
     "Polki necklace set on black velvet (from the client's photo). GOLD TESTING cascade, then 3D extruded "
     "gold FREE slams in exactly on the spoken 'free' with flash, shake and particle burst; VISIT OUR LAB TODAY.",
     "Aur sab se khaas baat, gold testing bilkul FREE!", "Strongest musical moment, bass impact on FREE"),
    ("05  LOCATION + CTA", "00:23-00:30", [24.6, 27.4, 29.5],
     "Real shopfront (client photo, Kling push-in) with animated pin and address card: Shop #2, Block 13, "
     "F-7 Markaz, Jinnah Super Market, next to Metro Shoes, Islamabad. Logo end card with VISIT US TODAY.",
     "Visit karein Gold Exchange Lab, F-7 Markaz Islamabad, Metro Shoes ke saath.",
     "Whoosh, pin pop, final impact + resolved chord"),
]

PW, PH = 1654, 2339  # A4 @ 200 dpi


def grab(work, t, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i",
                    os.path.join(work, "out", "GOLD_EXCHANGE_LAB_FINAL_AD.mp4"), "-frames:v", "1", path],
                   check=True)
    return Image.open(path).convert("RGB")


def wrap(d, text, fnt, width):
    out, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=fnt) > width:
            out.append(cur)
            cur = w
        else:
            cur = trial
    out.append(cur)
    return out


def main(work):
    tmp = os.path.join(work, "cache", "sb")
    os.makedirs(tmp, exist_ok=True)
    pages = []
    cover = Image.new("RGB", (PW, PH), (12, 10, 8))
    d = ImageDraw.Draw(cover)
    d.text((120, 200), "GOLD EXCHANGE LAB & JEWELRY", font=font("Bold", 64), fill=(226, 186, 104))
    d.text((120, 300), "30s Reel / TikTok advertisement  -  storyboard", font=font("Medium", 40), fill=(235, 230, 220))
    info = ["Format 1080 x 1920 (9:16), 30 fps, H.264 MP4, exactly 30.000 s",
            "Voice-over: Pakistani Urdu (ElevenLabs v4 via Higgsfield, voice 'Mark')",
            "Subtitles: Roman Urdu, burned in (subtitled master) + SRT; clean master without subtitles",
            "Visuals: Kling 3.0 image-to-video (Higgsfield) from client references + code-driven motion graphics",
            "Music & SFX: original, synthesised for this ad (royalty-free)",
            "", "Client-provided claim to verify before publishing: \"Islamabad's 1st Gold Testing Lab\"",
            "XRF reads surface composition; the spectrum card is illustrative, not a test result."]
    y = 420
    for line in info:
        d.text((120, y), line, font=font("Regular", 32), fill=(200, 195, 185))
        y += 56
    hero = grab(work, 28.6, os.path.join(tmp, "hero.png")).resize((540, 960))
    cover.paste(hero, ((PW - 540) // 2, 1050))
    pages.append(cover)
    for title, tc, times, visual, vo, sound in BOARD:
        pg = Image.new("RGB", (PW, PH), (245, 242, 236))
        d = ImageDraw.Draw(pg)
        d.text((100, 90), title, font=font("Bold", 60), fill=(30, 24, 14))
        d.text((100, 170), tc, font=font("SemiBold", 36), fill=(150, 110, 40))
        x = 100
        for t in times:
            im = grab(work, t, os.path.join(tmp, f"{t:.1f}.png")).resize((470, 836))
            pg.paste(im, (x, 260))
            d.text((x, 1110), f"{t:05.2f}s", font=font("Medium", 28), fill=(90, 80, 60))
            x += 500
        y = 1190
        for label, body in (("VISUAL", visual), ("VOICE-OVER (Roman Urdu)", vo), ("SOUND", sound)):
            d.text((100, y), label, font=font("Bold", 30), fill=(150, 110, 40))
            y += 50
            for ln in wrap(d, body, font("Regular", 34), 1450):
                d.text((100, y), ln, font=font("Regular", 34), fill=(30, 24, 14))
                y += 50
            y += 40
        pages.append(pg)
    out = os.path.join(work, "out", "GOLD_EXCHANGE_LAB_STORYBOARD.pdf")
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=200)
    print("storyboard ok", out)


if __name__ == "__main__":
    main(sys.argv[1])
