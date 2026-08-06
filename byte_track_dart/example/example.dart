import 'package:byte_track_dart/byte_track_dart.dart';

/// Minimal end-to-end example: two objects moving across three frames.
/// In a real app, replace `syntheticDetections(frame)` with the output of
/// your YOLO model for that frame (see README for the tflite/ONNX wiring).
void main() {
  final tracker = ByteTracker(
    highThresh: 0.6,
    lowThresh: 0.1,
    newTrackThresh: 0.7,
    matchThresh: 0.3,
    maxAge: 30,
    minHits: 1, // confirm immediately for this short demo
  );

  for (var frame = 0; frame < 5; frame++) {
    final detections = syntheticDetections(frame);
    final tracks = tracker.update(detections);

    print('--- frame $frame ---');
    for (final t in tracks) {
      print('  id=${t.id} class=${t.classId} score=${t.score.toStringAsFixed(2)} box=${t.bbox}');
    }
  }
}

/// Stand-in for real detector output: two boxes drifting to the right,
/// with the second one dropping to low confidence on frame 2 (simulating
/// partial occlusion) to show the second-pass recovery association.
List<Detection> syntheticDetections(int frame) {
  final personScore = frame == 2 ? 0.15 : 0.9; // dips into "low" band once
  return [
    Detection.xywh(
      cx: 100.0 + frame * 10,
      cy: 200.0,
      width: 50,
      height: 120,
      score: 0.95,
      classId: 0, // e.g. "person"
    ),
    Detection.xywh(
      cx: 400.0 + frame * 8,
      cy: 220.0,
      width: 60,
      height: 140,
      score: personScore,
      classId: 0,
    ),
  ];
}
