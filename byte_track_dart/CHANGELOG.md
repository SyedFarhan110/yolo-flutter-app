## 0.1.0

- Initial release.
- `ByteTracker`: two-stage (high/low confidence) association tracker with
  Kalman-filter motion prediction and Hungarian-algorithm assignment.
- `YoloDecoder`: generic raw-tensor decoder supporting both
  `withObjectness` (YOLOv5/v7) and `classesOnly` (YOLOv8/v9/v10/v11)
  output layouts, with letterbox unscaling and per-class NMS.
- `Detection`, `BBox`, `Track`/`TrackState` public models.
- Zero third-party dependencies.
