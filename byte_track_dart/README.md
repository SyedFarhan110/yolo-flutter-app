# byte_track_dart

A pure-Dart, **detector-agnostic** multi-object tracker based on the
ByteTrack/SORT family of algorithms. Feed it bounding boxes from **any**
YOLO model — v5, v8, v9, v10, v11, a custom-trained checkpoint, or a
completely different detector entirely — and get back stable object IDs
across frames.

No native dependencies, no FFI, no platform channels. It's plain Dart, so
it runs the same way on Flutter mobile, Flutter web, and server-side Dart.

## Why

Object detectors tell you *what* and *where*, frame by frame, with no
memory between frames. Object **tracking** is what turns "a person" in
frame 1 and "a person" in frame 2 into "the same person, ID #7." This
package implements that layer:

- Kalman filter per object for constant-velocity motion prediction
  (survives brief occlusion by coasting on the last known velocity)
- Hungarian algorithm for globally-optimal frame-to-frame assignment
  (not greedy nearest-neighbor matching)
- Two-stage ByteTrack association: low-confidence detections that a naive
  tracker would discard as noise get a second chance to match against
  tracks that went unmatched in the first pass — this is what the
  ByteTrack paper found reduces broken/switched IDs the most

## Install

```yaml
dependencies:
  byte_track_dart: ^0.1.0
```

## Quickstart

```dart
import 'package:byte_track_dart/byte_track_dart.dart';

final tracker = ByteTracker();

// Call once per frame, in order, with that frame's detections.
for (final frameDetections in videoFrames) {
  final tracks = tracker.update(frameDetections);
  for (final t in tracks) {
    print('id=${t.id} class=${t.classId} box=${t.bbox}');
  }
}
```

A `Detection` is the only thing the tracker needs — it doesn't care where
it came from:

```dart
final detection = Detection.xyxy(
  x1: 120, y1: 80, x2: 210, y2: 340,
  score: 0.87,
  classId: 0,
);
```

## Using it with a YOLO model

Most on-device runtimes (tflite, ONNX Runtime, PyTorch Mobile) hand back a
raw output tensor, not decoded boxes. `YoloDecoder` turns that tensor into
`Detection`s regardless of which YOLO generation produced it:

```dart
import 'package:byte_track_dart/byte_track_dart.dart';

// 1. Run inference with whatever runtime you're using.
final rawOutput = await interpreter.run(inputTensor); // shape e.g. [1, 84, 8400]

// 2. Reshape/transpose to [numBoxes][attributes] if your runtime returns
//    a transposed tensor (very common for YOLOv8+ exports).
final rows = transpose(rawOutput); // your own helper, or use a package like ml_linalg

// 3. Decode. Only `layout` and `numClasses` change between YOLO versions.
final detections = YoloDecoder.decode(
  rawOutput: rows,
  numClasses: 80,
  layout: YoloOutputLayout.classesOnly, // YOLOv8/v9/v10/v11
  confThreshold: 0.25,
  iouThreshold: 0.45,
  // If you letterboxed the input frame to a square before inference,
  // pass the same scale/pad here to map boxes back to original pixels:
  scaleX: letterbox.scale,
  scaleY: letterbox.scale,
  padX: letterbox.padX,
  padY: letterbox.padY,
);

// 4. Track.
final tracks = tracker.update(detections);
```

For YOLOv5/v7 exports (which keep a separate objectness score), use
`YoloOutputLayout.withObjectness` instead — everything else is identical.

Already decoding boxes yourself, or using a detector that isn't YOLO at
all? Skip `YoloDecoder` entirely and build `Detection`s directly — the
tracker's public API only ever sees `Detection`, so it works with
anything.

## Tuning

| Parameter | Default | What it does |
|---|---|---|
| `highThresh` | `0.6` | Score cutoff for first-pass (high-confidence) matching |
| `lowThresh` | `0.1` | Score floor for second-pass recovery matching |
| `newTrackThresh` | `0.7` | Minimum score for an unmatched detection to spawn a new track |
| `matchThresh` | `0.3` | Minimum IoU for a first-pass match |
| `secondPassMatchThresh` | `0.5` | Minimum IoU for a second-pass (recovery) match |
| `maxAge` | `30` | Frames a track can go unmatched before being dropped |
| `minHits` | `3` | Consecutive matches before a new track is reported |

Start from the defaults and adjust `maxAge`/`minHits` for your frame rate —
at 30fps, `maxAge: 30` tolerates about a second of occlusion; scale that
down for lower frame rates or fast-moving cameras.

## What this package does *not* do

- Run inference — bring your own detector/runtime
- Re-identification via appearance embeddings (that's DeepSORT-style
  ReID; this is motion + IoU based, matching the original ByteTrack)
- Letterbox/image preprocessing — a few lines, but out of scope here to
  keep this dependency-free

## License

MIT
