/// A minimal dense-matrix implementation used internally by [KalmanFilter].
///
/// This is intentionally small and unoptimized (no SIMD, no external linear
/// algebra dependency) since the matrices involved are fixed-size and tiny
/// (at most 8x8) -- one per tracked object, per frame. If you're tracking
/// hundreds of objects per frame and profiling shows this as a bottleneck,
/// swap this file out for `package:ml_linalg` without touching the rest of
/// the tracker.
class Matrix {
  final List<List<double>> data;
  final int rows;
  final int cols;

  Matrix(this.data)
      : rows = data.length,
        cols = data.isEmpty ? 0 : data[0].length;

  factory Matrix.zeros(int rows, int cols) =>
      Matrix(List.generate(rows, (_) => List<double>.filled(cols, 0.0)));

  factory Matrix.identity(int n) {
    final m = Matrix.zeros(n, n);
    for (var i = 0; i < n; i++) {
      m.data[i][i] = 1.0;
    }
    return m;
  }

  /// Builds a column matrix (n x 1) from a flat list of values.
  factory Matrix.column(List<double> values) =>
      Matrix(values.map((v) => [v]).toList());

  Matrix operator +(Matrix other) => Matrix(List.generate(
        rows,
        (i) => List.generate(cols, (j) => data[i][j] + other.data[i][j]),
      ));

  Matrix operator -(Matrix other) => Matrix(List.generate(
        rows,
        (i) => List.generate(cols, (j) => data[i][j] - other.data[i][j]),
      ));

  Matrix operator *(Matrix other) {
    assert(
      cols == other.rows,
      'Matrix dimension mismatch: ($rows x $cols) * (${other.rows} x ${other.cols})',
    );
    final result = Matrix.zeros(rows, other.cols);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < other.cols; j++) {
        var sum = 0.0;
        for (var k = 0; k < cols; k++) {
          sum += data[i][k] * other.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  Matrix transpose() => Matrix(List.generate(
        cols,
        (i) => List.generate(rows, (j) => data[j][i]),
      ));

  Matrix scale(double s) =>
      Matrix(data.map((row) => row.map((v) => v * s).toList()).toList());

  /// Inverts a square matrix via Gauss-Jordan elimination with partial
  /// pivoting. Falls back to a small diagonal epsilon if a matrix is
  /// (near-)singular rather than throwing, since the tracker must keep
  /// running frame-to-frame even if a covariance matrix becomes degenerate.
  Matrix inverse() {
    assert(rows == cols, 'Only square matrices can be inverted');
    final n = rows;
    final aug = List.generate(
      n,
      (i) => [...data[i], ...List.generate(n, (j) => i == j ? 1.0 : 0.0)],
    );

    for (var col = 0; col < n; col++) {
      var pivotRow = col;
      var maxVal = aug[col][col].abs();
      for (var r = col + 1; r < n; r++) {
        if (aug[r][col].abs() > maxVal) {
          maxVal = aug[r][col].abs();
          pivotRow = r;
        }
      }
      if (maxVal < 1e-12) {
        aug[col][col] += 1e-6;
        maxVal = aug[col][col].abs();
      }
      final tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;

      final pivot = aug[col][col];
      for (var j = 0; j < 2 * n; j++) {
        aug[col][j] /= pivot;
      }
      for (var r = 0; r < n; r++) {
        if (r == col) continue;
        final factor = aug[r][col];
        if (factor == 0) continue;
        for (var j = 0; j < 2 * n; j++) {
          aug[r][j] -= factor * aug[col][j];
        }
      }
    }

    return Matrix(List.generate(n, (i) => aug[i].sublist(n, 2 * n)));
  }

  List<double> get column0 => List.generate(rows, (i) => data[i][0]);
}
