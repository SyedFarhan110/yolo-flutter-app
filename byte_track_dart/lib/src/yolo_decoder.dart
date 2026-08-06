import 'bbox.dart';
import 'detection.dart';

/// Which raw output layout a YOLO export uses. This is the only thing that
/// differs between YOLO generations from a post-processing point of view.
enum YoloOutputLayout {
  /// `[cx, cy, w, h, objectness, class0, class1, ...]`
  /// Used by YOLOv5 and YOLOv7 exports.
  withObjectness,

  /// `[cx, cy, w, h, class0, class1, ...]` (objectness folded into the
  /// class scores). Used by YOLOv8, YOLOv9, YOLOv10, and YOLOv11 exports.
  classesOnly,
}

/// Decodes a raw YOLO model output tensor into ready-to-track [Detection]s.
///
/// This is deliberately decoupled from any specific inference runtime
/// (tflite, ONNX Runtime, PyTorch Mobile, ncnn, a Jetson/TensorRT engine
/// over a socket, ...). Whatever runtime you use, get its output into a
/// `List<List<double>>` of shape `[numBoxes][4 + numClasses(+1)]` -- most
/// runtimes hand back a `[1, attrs, numBoxes]` tensor, so you'll typically
/// need to transpose it first (see the example in the package README).
///
/// Once you have that, this same [decode] call works whether the boxes
/// came from a YOLOv5n you trained yourself or a stock YOLOv11x export --
/// only [YoloOutputLayout] and [numClasses] need to change.
class YoloDecoder {
  /// Decodes and runs per-class non-max suppression on raw detections.
  ///
  /// [scaleX]/[scaleY]/[padX]/[padY] undo letterbox resizing so boxes come
  /// back in original-image pixel coordinates -- pass the same values you
  /// used to letterbox the frame before feeding it to the model. Leave the
  /// scale at `1.0` and pad at `0.0` if you resized without letterboxing
  /// (i.e. you're not preserving aspect ratio).
  static List<Detection> decode({
    required List<List<double>> rawOutput,
    required int numClasses,
    required YoloOutputLayout layout,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
    double scaleX = 1.0,
    double scaleY = 1.0,
    double padX = 0.0,
    double padY = 0.0,
  }) {
    final candidates = <Detection>[];

    for (final row in rawOutput) {
      final cx = row[0];
      final cy = row[1];
      final w = row[2];
      final h = row[3];

      var bestScore = 0.0;
      var bestClass = -1;

      if (layout == YoloOutputLayout.withObjectness) {
        final objectness = row[4];
        for (var c = 0; c < numClasses; c++) {
          final classScore = row[5 + c] * objectness;
          if (classScore > bestScore) {
            bestScore = classScore;
            bestClass = c;
          }
        }
      } else {
        for (var c = 0; c < numClasses; c++) {
          final classScore = row[4 + c];
          if (classScore > bestScore) {
            bestScore = classScore;
            bestClass = c;
          }
        }
      }

      if (bestClass == -1 || bestScore < confThreshold) continue;

      final origCx = (cx - padX) / scaleX;
      final origCy = (cy - padY) / scaleY;
      final origW = w / scaleX;
      final origH = h / scaleY;

      candidates.add(Detection.xywh(
        cx: origCx,
        cy: origCy,
        width: origW,
        height: origH,
        score: bestScore,
        classId: bestClass,
      ));
    }

    return _nonMaxSuppression(candidates, iouThreshold);
  }

  /// Per-class greedy NMS: within each class, keeps the highest-scoring
  /// box and discards any lower-scoring box that overlaps it above
  /// [iouThreshold], repeating until no boxes remain.
  static List<Detection> _nonMaxSuppression(
    List<Detection> detections,
    double iouThreshold,
  ) {
    final byClass = <int, List<Detection>>{};
    for (final d in detections) {
      byClass.putIfAbsent(d.classId, () => []).add(d);
    }

    final kept = <Detection>[];
    for (final group in byClass.values) {
      group.sort((a, b) => b.score.compareTo(a.score));
      final active = List<bool>.filled(group.length, true);

      for (var i = 0; i < group.length; i++) {
        if (!active[i]) continue;
        kept.add(group[i]);
        for (var j = i + 1; j < group.length; j++) {
          if (!active[j]) continue;
          if (iouBBox(group[i].bbox, group[j].bbox) > iouThreshold) {
            active[j] = false;
          }
        }
      }
    }

    return kept;
  }
}
