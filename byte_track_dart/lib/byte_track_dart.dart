/// A pure-Dart, detector-agnostic multi-object tracker (ByteTrack/SORT
/// family) plus an optional generic YOLO output decoder.
///
/// Core tracking workflow:
/// ```dart
/// final tracker = ByteTracker();
/// final tracks = tracker.update(detections); // call once per frame
/// ```
///
/// If your detections come straight from a raw YOLO model tensor (any
/// version), decode them first:
/// ```dart
/// final detections = YoloDecoder.decode(
///   rawOutput: rows,
///   numClasses: 80,
///   layout: YoloOutputLayout.classesOnly, // v8/v9/v10/v11
/// );
/// final tracks = tracker.update(detections);
/// ```
library byte_track_dart;

export 'src/bbox.dart' show BBox, iouBBox;
export 'src/detection.dart' show Detection;
export 'src/track.dart' show Track, TrackState;
export 'src/byte_tracker.dart' show ByteTracker;
export 'src/yolo_decoder.dart' show YoloDecoder, YoloOutputLayout;
export 'src/hungarian_algorithm.dart' show HungarianAlgorithm;
