import 'bbox.dart';
import 'detection.dart';
import 'kalman_filter.dart';

/// Lifecycle state of a [Track].
enum TrackState {
  /// Just created, hasn't accumulated enough consecutive matches yet to be
  /// reported to the caller. Prevents single noisy detections from
  /// flashing a new ID into existence for one frame.
  tentative,

  /// Matched consistently -- this is what [ByteTracker.update] returns.
  confirmed,

  /// Was confirmed but didn't match a detection this frame. Kept alive
  /// (coasting on its Kalman prediction) for up to `maxAge` frames in case
  /// the object reappears, e.g. after brief occlusion.
  lost,
}

/// A single tracked object, persisted across frames with a stable [id].
class Track {
  final int id;
  final int classId;

  TrackState state = TrackState.tentative;
  double score;
  int hits = 0;
  int age = 0;
  int timeSinceUpdate = 0;

  final KalmanFilter _kf;

  Track({required this.id, required Detection detection})
      : classId = detection.classId,
        score = detection.score,
        _kf = KalmanFilter(initialMeasurement: measurementFromBBox(detection.bbox)) {
    hits = 1;
  }

  /// The track's current predicted/corrected box.
  BBox get bbox => _kf.bbox;

  /// Advances this track's Kalman filter by one frame. Called for every
  /// live track at the start of each [ByteTracker.update], before
  /// association.
  void predict() {
    _kf.predict();
    age++;
    timeSinceUpdate++;
  }

  /// Corrects this track with a matched detection from the current frame.
  void update(Detection detection) {
    _kf.update(measurementFromBBox(detection.bbox));
    score = detection.score;
    hits++;
    timeSinceUpdate = 0;
  }

  @override
  String toString() =>
      'Track(id: $id, cls: $classId, state: $state, $bbox)';
}
