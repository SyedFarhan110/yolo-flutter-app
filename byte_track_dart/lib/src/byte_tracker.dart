import 'bbox.dart';
import 'detection.dart';
import 'hungarian_algorithm.dart';
import 'track.dart';

class _AssociationResult {
  final List<List<int>> matches; // [trackIndex, detectionIndex] pairs
  final List<int> unmatchedTracks;
  final List<int> unmatchedDetections;

  _AssociationResult({
    required this.matches,
    required this.unmatchedTracks,
    required this.unmatchedDetections,
  });
}

/// A detector-agnostic multi-object tracker implementing the ByteTrack
/// association strategy: match high-confidence detections to tracks first,
/// then make a second pass matching low-confidence "leftover" detections
/// against tracks that are still unmatched, before giving up on them.
///
/// This is the key idea from the ByteTrack paper -- most trackers throw
/// away low-score boxes entirely, but a low-score box is often a real,
/// partially-occluded object rather than a false positive. Recovering it
/// via a second association pass measurably reduces broken/switched IDs.
///
/// Feed it [Detection]s from any detector, once per frame, in order:
///
/// ```dart
/// final tracker = ByteTracker();
/// for (final frameDetections in videoFrames) {
///   final tracks = tracker.update(frameDetections);
///   for (final t in tracks) {
///     print('id=${t.id} class=${t.classId} box=${t.bbox}');
///   }
/// }
/// ```
class ByteTracker {
  /// Detections scoring at or above this are attempted first, against all
  /// live tracks.
  final double highThresh;

  /// Detections scoring at or above this (but below [highThresh]) are only
  /// used in the second-pass recovery association, against tracks that
  /// went unmatched in the first pass.
  final double lowThresh;

  /// A high-confidence detection that still doesn't match any existing
  /// track must score at least this high to spawn a brand-new track.
  /// Keeps stray low-confidence noise from constantly minting new IDs.
  final double newTrackThresh;

  /// Minimum IoU required for a detection/track pair to count as a match
  /// in the first association pass.
  final double matchThresh;

  /// Looser minimum IoU used in the second-pass (low-score) recovery
  /// association, since coasting/occluded boxes tend to have drifted
  /// further from their last known position.
  final double secondPassMatchThresh;

  /// How many consecutive frames a track may go unmatched before it's
  /// dropped entirely.
  final int maxAge;

  /// How many consecutive matched frames a new track needs before it's
  /// reported as [TrackState.confirmed] instead of [TrackState.tentative].
  final int minHits;

  int _nextId = 1;
  final List<Track> _tracks = [];

  ByteTracker({
    this.highThresh = 0.6,
    this.lowThresh = 0.1,
    this.newTrackThresh = 0.7,
    this.matchThresh = 0.3,
    this.secondPassMatchThresh = 0.5,
    this.maxAge = 30,
    this.minHits = 3,
  });

  /// All tracks currently being followed, including ones not yet
  /// [TrackState.confirmed] and ones temporarily [TrackState.lost].
  /// Most callers want [update]'s return value instead.
  List<Track> get allTracks => List.unmodifiable(_tracks);

  /// Advances the tracker by one frame given this frame's raw detections
  /// (any confidence level -- the tracker does its own thresholding).
  /// Returns the currently [TrackState.confirmed] tracks.
  List<Track> update(List<Detection> detections) {
    for (final t in _tracks) {
      t.predict();
    }

    final highDets = <Detection>[];
    final lowDets = <Detection>[];
    for (final d in detections) {
      if (d.score >= highThresh) {
        highDets.add(d);
      } else if (d.score >= lowThresh) {
        lowDets.add(d);
      }
    }

    // --- First pass: high-confidence detections vs. every live track ---
    final firstPass = _associate(_tracks, highDets, matchThresh);
    for (final pair in firstPass.matches) {
      _tracks[pair[0]].update(highDets[pair[1]]);
    }

    // --- Second pass: low-confidence detections vs. tracks still unmatched ---
    final remainingTracks = firstPass.unmatchedTracks.map((i) => _tracks[i]).toList();
    final secondPass = _associate(remainingTracks, lowDets, secondPassMatchThresh);
    for (final pair in secondPass.matches) {
      remainingTracks[pair[0]].update(lowDets[pair[1]]);
    }

    // --- Spawn new tracks from high-confidence detections nothing matched ---
    for (final di in firstPass.unmatchedDetections) {
      final det = highDets[di];
      if (det.score >= newTrackThresh) {
        _tracks.add(Track(id: _nextId++, detection: det));
      }
    }

    // --- Update lifecycle state & drop stale tracks ---
    _tracks.removeWhere((t) => t.timeSinceUpdate > maxAge);
    for (final t in _tracks) {
      if (t.timeSinceUpdate == 0) {
        if (t.state == TrackState.tentative && t.hits < minHits) {
          // stays tentative until it earns confirmation
        } else {
          t.state = TrackState.confirmed;
        }
      } else {
        if (t.state == TrackState.tentative) {
          // a tentative track that missed a frame is unreliable -- drop it
          // next cleanup pass by aging it out quickly via timeSinceUpdate.
        } else {
          t.state = TrackState.lost;
        }
      }
    }
    _tracks.removeWhere(
      (t) => t.state == TrackState.tentative && t.timeSinceUpdate > 1,
    );

    return _tracks.where((t) => t.state == TrackState.confirmed).toList();
  }

  /// Resets the tracker, dropping all tracks and restarting ID assignment.
  void reset() {
    _tracks.clear();
    _nextId = 1;
  }

  _AssociationResult _associate(
    List<Track> tracks,
    List<Detection> dets,
    double iouThreshold,
  ) {
    if (tracks.isEmpty || dets.isEmpty) {
      return _AssociationResult(
        matches: const [],
        unmatchedTracks: List.generate(tracks.length, (i) => i),
        unmatchedDetections: List.generate(dets.length, (i) => i),
      );
    }

    final costMatrix = List.generate(
      tracks.length,
      (i) => List.generate(
        dets.length,
        (j) => 1.0 - iouBBox(tracks[i].bbox, dets[j].bbox),
      ),
    );

    final assignment = HungarianAlgorithm.solve(costMatrix);

    final matches = <List<int>>[];
    final matchedTracks = <int>{};
    final matchedDets = <int>{};

    for (final pair in assignment) {
      final ti = pair[0];
      final di = pair[1];
      if (costMatrix[ti][di] <= (1.0 - iouThreshold)) {
        matches.add([ti, di]);
        matchedTracks.add(ti);
        matchedDets.add(di);
      }
    }

    final unmatchedTracks = [
      for (var i = 0; i < tracks.length; i++)
        if (!matchedTracks.contains(i)) i,
    ];
    final unmatchedDetections = [
      for (var j = 0; j < dets.length; j++)
        if (!matchedDets.contains(j)) j,
    ];

    return _AssociationResult(
      matches: matches,
      unmatchedTracks: unmatchedTracks,
      unmatchedDetections: unmatchedDetections,
    );
  }
}
