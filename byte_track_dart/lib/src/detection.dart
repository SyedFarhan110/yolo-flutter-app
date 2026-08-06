import 'bbox.dart';

/// A single detection produced by *any* object detector for one frame.
///
/// This is the entire contract between your model's output and the
/// tracker. It doesn't matter whether the box came from YOLOv5, YOLOv8,
/// YOLOv11, a custom-trained model, or something that isn't YOLO at all
/// (Faster R-CNN, SSD, ML Kit, ...) -- as long as you can produce a
/// [Detection], [ByteTracker] can track it.
class Detection {
  final BBox bbox;
  final double score;
  final int classId;

  const Detection({
    required this.bbox,
    required this.score,
    required this.classId,
  });

  /// Builds a [Detection] from a top-left/bottom-right box, the format
  /// most post-NMS detector outputs already use.
  factory Detection.xyxy({
    required double x1,
    required double y1,
    required double x2,
    required double y2,
    required double score,
    required int classId,
  }) {
    return Detection(
      bbox: BBox(x1, y1, x2, y2),
      score: score,
      classId: classId,
    );
  }

  /// Builds a [Detection] from a center-point box (`cx, cy, w, h`), the
  /// format raw YOLO model output tensors use before decoding.
  factory Detection.xywh({
    required double cx,
    required double cy,
    required double width,
    required double height,
    required double score,
    required int classId,
  }) {
    final halfW = width / 2;
    final halfH = height / 2;
    return Detection(
      bbox: BBox(cx - halfW, cy - halfH, cx + halfW, cy + halfH),
      score: score,
      classId: classId,
    );
  }

  @override
  String toString() =>
      'Detection(cls: $classId, score: ${score.toStringAsFixed(2)}, $bbox)';
}

/// Converts a [BBox] into the `[cx, cy, aspect, height]` measurement vector
/// the Kalman filter operates on internally.
List<double> measurementFromBBox(BBox b) {
  final h = b.height;
  final aspect = h > 0 ? b.width / h : 0.0;
  return [b.centerX, b.centerY, aspect, h];
}
