# Drop-test clips

These are web encodes of the eight hand-drop videos, one per morphology for four
release conditions, plus a longer hero clip. The footage is slow motion. The capture
frame rate is not recorded and the slow-down does not look uniform, so playback time
is not real time and timing should not be compared between clips.

## Sources

The sources are 1920x1080, 30 fps phone exports (NTR45.mp4 ... WTP45.mp4, about 100 MB
in total). They are not tracked in this repository, and neither are the tracking and
encoding scripts. Each source holds 24p content (one duplicated frame in every five)
recorded sideways, with a silent audio track. Every clip starts from the upright 24 fps
stream:

```
-vf "transpose=1,decimate=cycle=5,setpts=N/24/TB"   # transpose=2 for WTP15
```

Frame numbers below are in this 24 fps stream. Release is the first frame with the hands
off the robot, and contact is the first frame with the robot in the blanket. WTP45 frame
120 repeats frame 119, so it is skipped.

## Height-synchronised comparison clips

Within each pair, the two robots are at the same height on screen in every frame from
release until the first of them reaches the blanket.

- **Tracking.** The robot's centroid was tracked in every frame by differencing against
  a clean plate (the per-pixel median of the clip) and keeping the dark connected
  component nearest the predicted position. Where the robot passes in front of the
  dropper, mostly in the overhead releases, its height was read by hand from zoomed frame
  grids. A weighted smoothing spline that never rises gives the height in each frame.
- **Scale.** The clips come from three camera framings: NTR45, WTR45 and WTR180; NTR180,
  NTP15, NTP45 and WTP45; and WTP15 on its own. Vertical scale was measured on the
  wall, from the bottom of the balcony cap to the soffit above the door and from the
  soffit to the door-handle plate. Relative to the second framing, the scales are 1.117
  for the first framing and 1.076 for WTP15. The absolute value comes from the no-tail
  robot's 44.7 cm length at its widest side-on extent; NTR45 and NTP15 agree to within
  0.7%. That gives 637, 570 and 613 px/m on the upright 1080x1920 frame.
- **Crop.** Each clip uses one static crop. The crop heights give both clips of a pair
  the same pixels per metre. The crops put both release points at the same output
  height, with the later robot's contact point inside the frame. Where a crop runs past
  the source frame, the strip is filled by mirroring the adjacent image, so the full
  9:16 frame can be shown without letterboxing.
- **Timing.** Output frame 12 is the release in both clips, and the 12 frames before it
  are native. After release, the reference clip plays its native frames. For each of
  those frames, the other clip shows the frame in which its robot has fallen the same
  distance, to the nearest frame. The reference is the clip whose robot falls faster per
  source frame, so the other clip skips frames and never repeats one. After the first
  contact, the reference keeps its native rate and the other clip keeps its last mapping
  rate. Both run until 7 frames (0.3 s) after the later contact. A source that ends
  sooner holds its last frame.
- **Encoding.** Each output frame is cropped on the upright 1080x1920 frame, then
  encoded:

```
-vf "scale=720:1280:flags=lanczos"
-c:v libx264 -preset slow -crf 23 -profile:v high -pix_fmt yuv420p -r 24 -an -movflags +faststart
```

  All outputs are 720x1280, 24 fps H.264 with no audio stream, and the moov atom is
  ahead of mdat. In each pair, the posters `<name>.jpg` (`-q:v 4`) are the same output
  frame, 55% of the way from release to the first contact.

| Pair | Frames | Output scale | Release at output y | Poster frame | Height gap, median / max |
|---|---|---|---|---|---|
| `roll45` | 98 (4.08 s) | 433 px/m | 134 | 52 | 2 / 11 px |
| `roll180` | 79 (3.29 s) | 439 px/m | 120 | 40 | 3 / 11 px |
| `roll180_pitch15` | 76 (3.17 s) | 452 px/m | 120 | 38 | 3 / 10 px |
| `roll180_pitch45` | 78 (3.25 s) | 450 px/m | 230 | 41 | 4 / 10 px |

Output heights are y values on the 1280-pixel-tall frame. The height gap is the vertical
distance between the two tracked robots, from release to the first contact. Re-detecting
the robots in the encoded files, in frames where both are clear of people, gives a median
of 3 to 5 px and a maximum of 15 px.

| Clip | Source | Release | Contact | Scale | Crop w×h at x,y | Contact at output frame |
|---|---|---|---|---|---|---|
| `roll45_notail` | NTR45 | 62 | 143 | 637 px/m | 1060×1884 at 20,12 | 89 |
| `roll45_tail` | WTR45, reference | 23 | 97 | 637 px/m | 1060×1884 at 20,30 | 86 |
| `roll180_notail` | NTR180, reference | 49 | 99 | 570 px/m | 936×1664 at 144,354 | 62 |
| `roll180_tail` | WTR180 | 43 | 121 | 637 px/m | 1046×1858 at 34,−20 | 70 |
| `roll180_pitch15_notail` | NTP15, reference | 55 | 103 | 570 px/m | 908×1616 at 110,424 | 60 |
| `roll180_pitch15_tail` | WTP15 | 49 | 127 | 613 px/m | 978×1738 at 102,−76 | 68 |
| `roll180_pitch45_notail` | NTP45 | 45 | 104 | 570 px/m | 912×1622 at 120,218 | 69 |
| `roll180_pitch45_tail` | WTP45, reference | 80 | 133 | 570 px/m | 912×1622 at 118,234 | 64 |

NTR45, WTR45, WTR180 and WTP15 were released from overhead. The other four were released
at rail height, so they fall a shorter distance and land sooner. In the roll 180 and
roll 180 / pitch 15 pairs, the rail-height robot reaches the blanket while the overhead
one is still about 0.45 m above its blanket. The overhead robot lands 8 frames later.
These overhead clips also start close to the top of their source frames. To give both
clips of each of these pairs the same scale and release height, the crops run past the
source frames. The mirrored strips are 75 px at the bottom of `roll180_notail`, 14 px at
the top of `roll180_tail`, 95 px at the bottom of `roll180_pitch15_notail` and 56 px at
the top of `roll180_pitch15_tail`.

`hero.mp4` is WTP45, the roll 180 / pitch 45 drop with the tail and the condition of
the paper's Fig. 3. It is encoded from the source with the same chain plus a crop to the
fall corridor, `crop=810:1440:150:345` on the upright 1080x1920 frame, then scaled to
720x1280 at CRF 21. It runs from frame 70 to the last frame, 142, of the 24 fps stream:
73 frames, 3.04 s. Release is at 0.417 s and blanket contact at 2.625 s. The poster
`hero.jpg` is clip frame 40.
