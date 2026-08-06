import 'package:byte_track_dart/byte_track_dart.dart';
import 'package:test/test.dart';

Detection box(double cx, double cy, {double score = 0.9, int classId = 0}) {
  return Detection.xywh(
    cx: cx,
    cy: cy,
    width: 50,
    height: 100,
    score: score,
    classId: classId,
  );
}

void main() {
  group('ByteTracker', () {
    test('assigns a stable id to a single object moving smoothly', () {
      final tracker = ByteTracker(minHits: 1);

      int? id;
      for (var i = 0; i < 10; i++) {
        final tracks = tracker.update([box(100.0 + i * 5, 200.0)]);
        expect(tracks.length, 1);
        id ??= tracks.first.id;
        expect(tracks.first.id, id, reason: 'id should not change frame-to-frame');
      }
    });

    test('assigns different ids to two well-separated objects', () {
      final tracker = ByteTracker(minHits: 1);

      final tracks = tracker.update([box(100, 200), box(500, 200)]);
      expect(tracks.length, 2);
      expect(tracks[0].id, isNot(equals(tracks[1].id)));
    });

    test('recovers a low-confidence detection via second-pass association', () {
      final tracker = ByteTracker(minHits: 1, lowThresh: 0.1, highThresh: 0.6);

      final firstId = tracker.update([box(100, 200, score: 0.9)]).first.id;

      // Same object, same position, but only a low-confidence detection
      // this frame (simulated partial occlusion). It should NOT get
      // dropped or reassigned a new id -- second-pass recovery should
      // keep matching it to the existing track.
      final recovered = tracker.update([box(102, 200, score: 0.15)]);
      expect(recovered.length, 1, reason: 'low-score box should still count as a match');
      expect(recovered.first.id, firstId);
    });

    test('drops a track after maxAge frames with no matching detection', () {
      final tracker = ByteTracker(minHits: 1, maxAge: 3);

      tracker.update([box(100, 200)]);
      expect(tracker.allTracks.length, 1);

      for (var i = 0; i < 5; i++) {
        tracker.update([]); // nothing detected this frame
      }

      expect(tracker.allTracks, isEmpty);
    });

    test('a tentative track needs minHits consecutive matches to be confirmed', () {
      final tracker = ByteTracker(minHits: 3);

      expect(tracker.update([box(100, 200)]), isEmpty); // hit 1
      expect(tracker.update([box(102, 200)]), isEmpty); // hit 2
      final confirmed = tracker.update([box(104, 200)]); // hit 3
      expect(confirmed.length, 1);
    });

    test('reset clears all tracks and restarts id numbering', () {
      final tracker = ByteTracker(minHits: 1);
      final firstId = tracker.update([box(100, 200)]).first.id;

      tracker.reset();
      final afterReset = tracker.update([box(100, 200)]).first.id;

      expect(afterReset, firstId, reason: 'ids should restart from the same seed after reset');
    });
  });

  group('HungarianAlgorithm', () {
    test('finds the minimum-cost assignment on a square matrix', () {
      final cost = [
        [4.0, 1.0, 3.0],
        [2.0, 0.0, 5.0],
        [3.0, 2.0, 2.0],
      ];
      final assignment = HungarianAlgorithm.solve(cost);

      double total = 0;
      for (final pair in assignment) {
        total += cost[pair[0]][pair[1]];
      }
      // Known optimal assignment for this matrix costs 5 (0->1, 1->0, 2->2
      // or an equivalent permutation summing to the same minimum).
      expect(total, closeTo(5.0, 1e-9));
      expect(assignment.length, 3);
    });

    test('handles rectangular matrices by leaving extras unmatched', () {
      final cost = [
        [1.0, 10.0],
        [10.0, 1.0],
        [5.0, 5.0],
      ];
      final assignment = HungarianAlgorithm.solve(cost);
      // Only 2 columns exist, so at most 2 pairs can be produced.
      expect(assignment.length, lessThanOrEqualTo(2));
    });
  });

  group('iouBBox', () {
    test('is 1.0 for identical boxes', () {
      final a = BBox(0, 0, 10, 10);
      expect(iouBBox(a, a), closeTo(1.0, 1e-9));
    });

    test('is 0.0 for non-overlapping boxes', () {
      final a = BBox(0, 0, 10, 10);
      final b = BBox(20, 20, 30, 30);
      expect(iouBBox(a, b), 0.0);
    });
  });
}
