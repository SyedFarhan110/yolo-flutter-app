import 'bbox.dart';
import 'matrix.dart';

/// A constant-velocity Kalman filter for a single tracked bounding box.
///
/// State vector (8-dim): `[cx, cy, aspect, height, vcx, vcy, vaspect, vheight]`
/// Measurement vector (4-dim): `[cx, cy, aspect, height]`
///
/// This is the standard SORT/DeepSORT/ByteTrack state parameterization:
/// tracking center point, aspect ratio and height (rather than width/height
/// directly) keeps the aspect ratio roughly constant under the linear
/// motion model, which behaves better for walking/driving objects than
/// tracking width and height as independent quantities.
class KalmanFilter {
  static const int stateDim = 8;
  static const int measDim = 4;

  late Matrix _F; // state transition matrix
  late Matrix _H; // measurement matrix
  late Matrix _Q; // process noise covariance
  late Matrix _R; // measurement noise covariance
  late Matrix _P; // state covariance
  late Matrix _x; // state vector (8 x 1)

  KalmanFilter({required List<double> initialMeasurement}) {
    _F = Matrix.identity(stateDim);
    for (var i = 0; i < 4; i++) {
      _F.data[i][i + 4] = 1.0; // position += velocity each step
    }

    _H = Matrix.zeros(measDim, stateDim);
    for (var i = 0; i < 4; i++) {
      _H.data[i][i] = 1.0;
    }

    _P = Matrix.identity(stateDim);
    for (var i = 4; i < stateDim; i++) {
      _P.data[i][i] = 1000.0; // velocities start highly uncertain
    }

    _Q = Matrix.identity(stateDim);
    for (var i = 4; i < stateDim; i++) {
      _Q.data[i][i] = 0.01; // low trust in sudden velocity changes
    }

    _R = Matrix.identity(measDim).scale(1.0);

    _x = Matrix.column([
      initialMeasurement[0],
      initialMeasurement[1],
      initialMeasurement[2],
      initialMeasurement[3],
      0.0,
      0.0,
      0.0,
      0.0,
    ]);
  }

  /// Advances the state one frame using the constant-velocity model.
  /// Call this once per frame for every track, even ones that don't get
  /// matched to a detection -- that's what lets a track survive brief
  /// occlusion by coasting on its last known velocity.
  void predict() {
    _x = _F * _x;
    _P = (_F * _P * _F.transpose()) + _Q;
  }

  /// Corrects the predicted state using a new measurement
  /// `[cx, cy, aspect, height]`.
  void update(List<double> measurement) {
    final z = Matrix.column(measurement);
    final y = z - (_H * _x); // innovation
    final s = (_H * _P * _H.transpose()) + _R; // innovation covariance
    final k = _P * _H.transpose() * s.inverse(); // Kalman gain
    _x = _x + (k * y);
    final identity = Matrix.identity(stateDim);
    _P = (identity - (k * _H)) * _P;
  }

  /// The filter's current best estimate of the box, as a [BBox].
  BBox get bbox {
    final cx = _x.data[0][0];
    final cy = _x.data[1][0];
    final aspect = _x.data[2][0];
    final h = _x.data[3][0];
    final w = aspect * h;
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
  }
}
