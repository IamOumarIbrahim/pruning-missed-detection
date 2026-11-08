# Data

The paper uses the RGB stream of the [Driver Monitoring Dataset (DMD)](https://dmd.vicomtech.org/).
DMD frames are not redistributed here. This repository holds only the 4-class annotations.

| Path | Contents | In git |
| :--- | :--- | :--- |
| `annotations/RGB/` | Source annotations and subject splits | yes |
| `processed/RGB/labels/` | YOLO labels, one file per frame | yes |
| `processed/RGB/yolo/` | `train.txt`, `val.txt`, `test.txt` image lists | yes |
| `processed/RGB/coco/` | The same annotations in COCO format | yes |
| `processed/RGB/images/` | 640 x 640 frames at 1 fps | no: request DMD |
| `partitions/` | Crop, collage and resized inputs | no: rebuilt by the script below |

## Rebuilding

1. Request DMD and extract the RGB video at 1 fps.
2. Center-crop each 1280 x 720 frame to 720 x 720, resize to 640 x 640, and save it as
   `processed/RGB/images/subject_XX/video_YY/subject_XX_video_YY_frame_NNNN.jpg`.
3. Run `python scripts/build_partitions.py`.

## Splits

Subjects never cross splits.

| Split | Subjects | Frames | Clips | Yawning | Hand-over-mouth | Drinking | Phone use |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 01, 04, 06, 07, 08, 09, 13, 14 | 9,087 | 40 | 94 | 83 | 154 | 1,417 |
| Val | 02, 03, 11 | 3,423 | 15 | 32 | 30 | 54 | 523 |
| Test | 05, 10, 12 | 3,213 | 13 | 33 | 28 | 56 | 497 |

A clip is one recording. Yawning and hand-over-mouth test frames each come from 5 clips, so the
effective test sample is far smaller than the frame counts suggest.
