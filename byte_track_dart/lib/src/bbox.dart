/// A simple axis-aligned bounding box in `[x1, y1, x2, y2]` (top-left,
/// bottom-right) pixel coordinates.
///
/// Used both for incoming detections and for the boxes predicted by each
/// track's Kalman filter, so IoU can be computed uniformly between the two.
class BBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  const BBox(this.x1, this.y1, this.x2, this.y2);

  double get width => (x2 - x1).clamp(0.0, double.infinity);
  double get height => (y2 - y1).clamp(0.0, double.infinity);
  double get area => width * height;
  double get centerX => (x1 + x2) / 2;
  double get centerY => (y1 + y2) / 2;

  @override
  String toString() =>
      'BBox(${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)}, '
      '${x2.toStringAsFixed(1)}, ${y2.toStringAsFixed(1)})';
}

/// Intersection-over-union between two boxes, in `[0, 1]`.
double iouBBox(BBox a, BBox b) {
  final interX1 = a.x1 > b.x1 ? a.x1 : b.x1;
  final interY1 = a.y1 > b.y1 ? a.y1 : b.y1;
  final interX2 = a.x2 < b.x2 ? a.x2 : b.x2;
  final interY2 = a.y2 < b.y2 ? a.y2 : b.y2;

  final interW = (interX2 - interX1).clamp(0.0, double.infinity);
  final interH = (interY2 - interY1).clamp(0.0, double.infinity);
  final interArea = interW * interH;

  final unionArea = a.area + b.area - interArea;
  if (unionArea <= 0) return 0.0;
  return interArea / unionArea;
}
