// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:byte_track_dart/byte_track_dart.dart';

/// Real-time YOLO camera inference with custom multi-object tracking via [ByteTracker].
/// Native detection overlays are hidden once the tracker is live; boxes and stable
/// track IDs are drawn on top using a [CustomPaint] layer instead.
class TrackingScreen extends StatefulWidget {
  const TrackingScreen({super.key});

  @override
  State<TrackingScreen> createState() => _TrackingScreenState();
}

class _TrackingScreenState extends State<TrackingScreen> {
  // One tracker instance for the whole session — it holds state across frames,
  // so it must NOT be recreated on every onResult callback.
  final ByteTracker _tracker = ByteTracker();

  final YOLOViewController _controller = YOLOViewController();

  List<Track> _tracks = [];

  // Native overlays can only be hidden once the platform view + method channel
  // are live (after the first onResult fires), so we guard this to call it once.
  bool _overlaysHidden = false;

  void _onResult(List<YOLOResult> results) {
    if (!_overlaysHidden) {
      _controller.setShowOverlays(false);
      _overlaysHidden = true;
    }

    // YOLOResult is already decoded/NMS'd by the plugin, so map straight to Detection.
    final detections = results
        .map(
          (r) => Detection.xyxy(
            x1: r.boundingBox.left,
            y1: r.boundingBox.top,
            x2: r.boundingBox.right,
            y2: r.boundingBox.bottom,
            score: r.confidence,
            classId: r.classIndex,
          ),
        )
        .toList();

    final tracks = _tracker.update(detections);

    if (!mounted) return;
    setState(() => _tracks = tracks);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          YOLOView(
            modelPath: 'yolo26n',
            task: YOLOTask.detect,
            controller: _controller,
            onResult: _onResult,
          ),
          Positioned.fill(child: CustomPaint(painter: _TrackPainter(_tracks))),
        ],
      ),
    );
  }
}

class _TrackPainter extends CustomPainter {
  final List<Track> tracks;
  _TrackPainter(this.tracks);

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.greenAccent;

    final labelStyle = const TextStyle(
      color: Colors.greenAccent,
      fontSize: 14,
      fontWeight: FontWeight.w600,
    );

    for (final t in tracks) {
      final rect = Rect.fromLTRB(t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2);
      canvas.drawRect(rect, boxPaint);

      final tp = TextPainter(
        text: TextSpan(text: 'ID ${t.id}', style: labelStyle),
        textDirection: TextDirection.ltr,
      )..layout();

      // Draw the label just above the box; clamp so it doesn't go off-screen
      // for detections near the top edge.
      final labelOffset = Offset(
        rect.left,
        (rect.top - tp.height - 2).clamp(0, size.height).toDouble(),
      );
      tp.paint(canvas, labelOffset);
    }
  }

  @override
  bool shouldRepaint(covariant _TrackPainter oldDelegate) => true;
}
