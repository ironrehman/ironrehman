# Gold Exchange Lab & Jewelry — 30 s cinematic ad

1080 × 1920 (9:16) · 30 fps · H.264 MP4 · exactly 30.000 s · Pakistani Urdu voice-over · Roman Urdu subtitles.

## Deliverables

| File | What it is |
|---|---|
| `GOLD_EXCHANGE_LAB_FINAL_AD.mp4` | Final ad, Roman Urdu subtitles burned in |
| `GOLD_EXCHANGE_LAB_FINAL_AD_NO_SUBTITLES.mp4` | Same cut without subtitles (for re-use / other languages) |
| `GOLD_EXCHANGE_LAB_VOICEOVER.wav` | Urdu narration alone, placed on the 30 s timeline (48 kHz / 24-bit) |
| `GOLD_EXCHANGE_LAB_MIX.wav` | Final stereo mix, −14 LUFS integrated, −1.5 dBTP |
| `GOLD_EXCHANGE_LAB_SUBTITLES.srt` | Roman Urdu subtitles (also in `deliverables/`) |
| `GOLD_EXCHANGE_LAB_STORYBOARD.pdf` | Scene-by-scene storyboard built from frames of the final render |
| `stem_music.wav`, `stem_sfx.wav` | Music and sound-design stems for re-editing |

The rendered media files are stored in the client's Higgsfield library (links are in the
production notes in the chat); this repository holds the editable source project.

## How it was made

**Picture.** Every live-action shot is Kling 3.0 Pro image-to-video, generated through
Higgsfield from the client's own reference photos:

| Scene | Source reference | Generation |
|---|---|---|
| 1 Hook (0–4 s) | flower-cluster ring from the jewelry-set photo | GPT Image 2.5 hero still (ring on black) → Kling 3.0 orbit/push |
| 3 XRF test (9–17 s) | the client's photo of their yellow handheld XRF analyzer | outpainted to 9:16 → Nano Banana 2 edit (technician holds a bridal set, machine untouched) → Kling 3.0 push-in |
| 4 FREE (17–23 s) | 21K polki necklace set photo | GPT Image 2.5 still on black velvet bust → Kling 3.0 push + light sweep |
| 5 Location (23–26.7 s) | the real shopfront photo (sign "GOLD EXCHANGE LAB & JEWELRY") | Kling 3.0 slow push-in |

Scenes 2 and 6 (brand reveal, end card) are pure motion graphics around the client's
**original logo file**, which is cut out by luminance and composited untouched (it is never
redrawn by an AI model).

All typography, the logo animation, particles, light sweeps, XRF scan HUD, location card,
subtitles and transitions are rendered frame by frame by `scripts/render.py` (Python, NumPy,
Pillow, ffmpeg) for pixel-exact text and spelling.

**Voice.** ElevenLabs v4 via Higgsfield, preset voice "Mark", written in Urdu script so the
pronunciation is natural Pakistani Urdu with English technical terms (gold testing, purity,
XRF). Each line was generated separately and placed on the timeline; word timings were
measured with faster-whisper and drive the word-by-word subtitle animation.

**Music & SFX.** Original score and sound design synthesised in `scripts/audio.py`
(D-minor cinematic electronic bed with pluck arpeggio, strings swell, bass, drums, risers,
whooshes, impacts, metallic shimmers, scanner sounds). Nothing is sampled, so there is no
third-party licence to clear. Music is side-chain ducked ~10 dB under the voice.

## Timeline (master: `scripts/timeline.py`)

| Time | Scene | On screen | VO (Roman Urdu) |
|---|---|---|---|
| 0:00–0:04 | Hook | IS YOUR GOLD / **PURE** OR NOT? | Kya aap ka sona waqai khalis hai? |
| 0:04–0:09 | Brand | Logo reveal · ISLAMABAD'S 1ST · GOLD TESTING LAB | Islamabad ki pehli Gold Testing Lab! |
| 0:09–0:17 | XRF | Scan HUD · GOLD PURITY ANALYSIS · XRF TECHNOLOGY · PROFESSIONAL TESTING | Ab apne sonay ki purity check karwayein, advanced XRF technology ke saath. |
| 0:17–0:23 | Offer | GOLD TESTING · **FREE** (3D, on the spoken "free") · VISIT OUR LAB TODAY | Aur sab se khaas baat, gold testing bilkul FREE! |
| 0:23–0:26.7 | Location | Pin + address card over the real shopfront | Visit karein Gold Exchange Lab, F-7 Markaz Islamabad, |
| 0:26.7–0:30 | End card | Logo · GOLD EXCHANGE LAB & JEWELRY · VISIT US TODAY | Metro Shoes ke saath. |

## Rebuild

```bash
# work_dir layout:
#   clips/ring.mp4 clips/xrf.mp4 clips/polki.mp4 clips/shop.mp4   (Kling renders)
#   vo/line1.mp3 … vo/line5.mp3                                   (ElevenLabs lines)
#   refs/logo.jpg                                                  (client logo)
#   fonts/Montserrat-*.ttf                                         (SIL OFL, github.com/JulietaUla/Montserrat)
pip install numpy scipy pillow
./build.sh <work_dir> <workers>
```

Edit `scripts/timeline.py` to retime anything; picture, sound hits and subtitles follow.
`python3 scripts/render.py <work_dir> --stills 2.5,20.6` renders preview frames.

## Compliance notes

- "Islamabad's 1st Gold Testing Lab" is a **client-provided claim**; confirm it can be
  substantiated before publishing.
- The XRF spectrum card is labelled "Illustrative visual" and shows no karat/purity values;
  no test result is invented. XRF measures surface composition, so the ad does not claim it
  proves purity throughout an item.
- The map is an animated location card with the exact address supplied; no street map was
  invented. The supplied reference image of a Thermo Niton XL2 was not used, because it is a
  different, third-party-branded model rather than the client's machine.
