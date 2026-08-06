/// Solves the linear assignment problem: given an `n x m` cost matrix,
/// finds the pairing of rows to columns that minimizes total cost, using
/// the classic O(n^3) Kuhn-Munkres (Hungarian) algorithm.
///
/// In this package, rows are existing tracks and columns are new-frame
/// detections; the cost matrix holds `1 - IoU` between each pair, so the
/// "cheapest" assignment is the one that maximizes total overlap.
class HungarianAlgorithm {
  /// Returns a list of `[rowIndex, colIndex]` pairs for the optimal
  /// assignment. If the matrix is rectangular, extra rows or columns are
  /// simply left unmatched -- callers should compute the leftover
  /// indices themselves (see `ByteTracker._associate`).
  static List<List<int>> solve(List<List<double>> costMatrix) {
    if (costMatrix.isEmpty || costMatrix[0].isEmpty) return [];

    final nRows = costMatrix.length;
    final nCols = costMatrix[0].length;
    final n = nRows > nCols ? nRows : nCols;

    // Pad to a square matrix with a very high cost so padded cells are
    // never chosen unless the algorithm has no other option.
    const bigCost = 1e9;
    final cost = List.generate(
      n,
      (i) => List.generate(
        n,
        (j) => (i < nRows && j < nCols) ? costMatrix[i][j] : bigCost,
      ),
    );

    // 1-indexed working arrays, per the standard competitive-programming
    // formulation of the algorithm (keeps the potential-update logic
    // simple to verify against reference implementations).
    final u = List<double>.filled(n + 1, 0.0);
    final v = List<double>.filled(n + 1, 0.0);
    final p = List<int>.filled(n + 1, 0); // p[j] = row currently assigned to column j
    final way = List<int>.filled(n + 1, 0);

    for (var i = 1; i <= n; i++) {
      p[0] = i;
      var j0 = 0;
      final minV = List<double>.filled(n + 1, double.infinity);
      final used = List<bool>.filled(n + 1, false);

      do {
        used[j0] = true;
        final i0 = p[j0];
        var delta = double.infinity;
        var j1 = -1;
        for (var j = 1; j <= n; j++) {
          if (used[j]) continue;
          final cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minV[j]) {
            minV[j] = cur;
            way[j] = j0;
          }
          if (minV[j] < delta) {
            delta = minV[j];
            j1 = j;
          }
        }
        for (var j = 0; j <= n; j++) {
          if (used[j]) {
            u[p[j]] += delta;
            v[j] -= delta;
          } else {
            minV[j] -= delta;
          }
        }
        j0 = j1;
      } while (p[j0] != 0);

      do {
        final j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0 != 0);
    }

    final result = <List<int>>[];
    for (var j = 1; j <= n; j++) {
      final row = p[j] - 1;
      final col = j - 1;
      if (row < nRows && col < nCols) {
        result.add([row, col]);
      }
    }
    return result;
  }
}
