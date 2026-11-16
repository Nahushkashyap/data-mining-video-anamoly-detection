> **Note (GitHub copy):** the media files themselves are not committed (videos are large, and the sound clips are
> FSD50K / UCF-Crime material with their own licences). To recreate them: take the FSD50K eval clips listed below
> (by FSD50K clip id in the file name) and the UCF-Crime videos named below, and mix/strip audio with `ffmpeg`
> as described. Put them in this folder.

# Demo media (made 22 Nov 2026)

All sound clips are from the **FSD50K eval split**. The audio head never saw them in training (it trained on the dev split).
FSD50K clips are Creative Commons licensed (per-clip licence in the FSD50K metadata).

| File | What it is | What the app does with it (checked 22 Nov) |
|---|---|---|
| `home_sounds_realistic.mp4` | UCF `Normal_Videos013` (a **training** video, normal street scene) + scream @0:08, glass break @0:18, door slam @0:28, shout @0:34 mixed into its own soundtrack (hum at about −27 dBFS) | Default τₐ 0.50: **burglary (glass break) 0:17–0:20, audio only**. Set τₐ to 0.35: also **fighting (scream) 0:07–0:12**. Door slam and shout are missed. |
| `home_sounds_clean.mp4` | Same, with the original soundtrack turned down 15 dB | Same alerts; glass break 95% |
| `*_fsd50k_*.wav` | The individual clips, for playing into the mic in **Live camera** mode | Scored alone: glass 1.0, siren 0.99, scream 0.78 → alerts; door slam 0.29, shout 0.12 and the gunshot clip (gunfire mixed with an aircraft engine, shooting 0.01) → no alert, so don't use that one |

Strongest demo: upload `home_sounds_realistic.mp4` with the audio slider at 0.35. In Live mode, play `glass_break_*.wav` or `siren_*.wav` near the mic.

## Explosion033_video_only.mp4 (added 22 Nov)
Unseen UCF-Crime test video Explosion033 with its (silent) audio track removed (`ffmpeg -an -c:v copy`). Video-only demo: two explosion alerts, 28.6-49.4 s (69%) and 59.8-105.1 s (70%); ground truth 32.3-45.0 s and 51.7-105.2 s. Same alerts as the original file.
