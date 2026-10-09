# Drop-test clips

These are web encodes of the eight hand-drop videos, one per morphology for four
release conditions, plus a longer hero clip. The footage is slow motion. The capture
frame rate is not recorded and the slow-down does not look uniform, so playback time
is not real time and timing should not be compared between clips.

## Encoding

The sources are 1920x1080, 30 fps phone exports (NTR45.mp4 ... WTP45.mp4, about 100 MB
in total); they are not tracked in this repository. Each one holds 24p content (one
duplicated frame in every five) recorded sideways, with a silent audio track. Each clip
was made upright and decimated to 24 fps, then trimmed by frame index and encoded:

```
-vf "transpose=1,decimate=cycle=5,setpts=N/24/TB,scale=720:1280:flags=lanczos"   # transpose=2 for WTP15
-c:v libx264 -preset slow -crf 24 -profile:v high -pix_fmt yuv420p -an -movflags +faststart
```

The hero clip uses CRF 22. All outputs are 720x1280, 24 fps H.264 with no audio stream
and the moov atom ahead of mdat.

## Trims

Every comparison clip is 100 frames (4.167 s) and starts 12 frames (0.5 s) before the
release frame, so the two clips of a pair play in step from release. Where the source
ends sooner, the last frame is held (`tpad=stop_mode=clone`) until the clip reaches
full length. Each poster `<name>.jpg` is the frame 62% of the way from release to
blanket contact, where the robot is clearly in the air.

Frame numbers are in the 24 fps stream: release is the first frame with the hands off
the robot, and contact is the first frame with the robot in the blanket.

| Clip | Source | Release | Contact | Release to contact | Held frames |
|---|---|---|---|---|---|
| `roll45_notail` | NTR45 | 62 | 143 | 3.375 s | 0 |
| `roll45_tail` | WTR45 | 23 | 97 | 3.083 s | 0 |
| `roll180_notail` | NTR180 | 49 | 99 | 2.083 s | 29 |
| `roll180_tail` | WTR180 | 43 | 121 | 3.250 s | 0 |
| `roll180_pitch15_notail` | NTP15 | 55 | 103 | 2.000 s | 29 |
| `roll180_pitch15_tail` | WTP15 | 49 | 127 | 3.250 s | 2 |
| `roll180_pitch45_notail` | NTP45 | 45 | 104 | 2.458 s | 21 |
| `roll180_pitch45_tail` | WTP45 | 80 | 133 | 2.208 s | 25 |

NTR45, WTR45, WTR180 and WTP15 were released from overhead. The other four were released
at rail height, so they fall a shorter distance on video and land sooner. Within the
roll 180 and roll 180 / pitch 15 pairs, the two clips therefore start together but
land about 1.2 s apart.

`hero.mp4` is WTP45, the roll 180 / pitch 45 drop with the tail and the condition of
the paper's Fig. 3. It is encoded from the source with the same chain plus a crop to the
fall corridor, `crop=810:1440:150:345` on the upright 1080x1920 frame, then scaled to
720x1280 at CRF 21. It runs from frame 70 to the last frame, 142, of the 24 fps stream:
73 frames, 3.04 s. Release is at 0.417 s and blanket contact at 2.625 s. The poster
`hero.jpg` is clip frame 40.
