// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:byte_track_dart/byte_track_dart.dart';

/// Real-time YOLO camera inference with multi-object tracking via [ByteTracker].
///
/// NOTE: This replaces the original [YOLOShowcase]-based screen with a bare [YOLOView]
/// because [YOLOShowcase] doesn't expose `onResult`/`controller`, which the tracker needs
/// to see per-frame detections. As a result this screen is Detect-only — the 6-task
/// switcher (Seg/Sem/Cls/Pose/OBB) and model-size picker that [YOLOShowcase] provided
/// are not reproduced here. If you need those back, they'd need to be rebuilt manually
/// or this screen should stay separate from a `YOLOShowcase`-based one.
class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  String? _versionLabel;

  // One tracker instance for the whole session — holds state across frames.
  final ByteTracker _tracker = ByteTracker();
  final YOLOViewController _controller = YOLOViewController();
  List<Track> _tracks = [];
  bool _overlaysHidden = false;

  @override
  void initState() {
    super.initState();
    _loadVersion();
  }

  Future<void> _loadVersion() async {
    final info = await PackageInfo.fromPlatform();
    if (!mounted) return;
    // Match yolo-ios-app's `labelVersion`: "v<version> (<build>)".
    setState(() => _versionLabel = 'v${info.version} (${info.buildNumber})');
  }

  void _onResult(List<YOLOResult> results) {
    if (!_overlaysHidden) {
      _controller.setShowOverlays(false);
      _overlaysHidden = true;
    }

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

  Future<void> _onCapture(Uint8List bytes) async {
    // Capture the share-sheet anchor BEFORE any async gap (no BuildContext use after await). iOS 26 gives the activity
    // controller a popoverPresentationController even on iPhone; without a valid source rect the popover anchors at
    // (0,0) and can present/dismiss incorrectly (and it is required on iPad). Use the screen's render box.
    final box = context.findRenderObject() as RenderBox?;
    final origin = box != null && box.hasSize
        ? box.localToGlobal(Offset.zero) & box.size
        : null;

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/yolo_capture.jpg')..writeAsBytesSync(bytes);

    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(file.path)],
        text: 'Ultralytics YOLO',
        sharePositionOrigin: origin,
      ),
    );
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
          if (_versionLabel != null)
            Positioned(
              left: 12,
              bottom: 12,
              child: Text(
                _versionLabel!,
                style: const TextStyle(color: Colors.white70, fontSize: 12),
              ),
            ),
          Positioned(
            right: 12,
            bottom: 12,
            child: FloatingActionButton(
              mini: true,
              onPressed: () async {
                // If you want the actual camera frame here instead of a placeholder,
                // wire this up through streamingConfig's includeOriginalImage and
                // capture the bytes from onStreamingData instead.
                final placeholder = Uint8List(0);
                await _onCapture(placeholder);
              },
              child: const Icon(Icons.ios_share),
            ),
          ),
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
