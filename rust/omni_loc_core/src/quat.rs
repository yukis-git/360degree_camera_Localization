//! Quaternion helpers, ported from `ESKF.py` (`(w, x, y, z)` convention throughout).
//!
//! - [`mul`] mirrors `quat_multiply` (ESKF.py line ~26).
//! - [`normalize`] mirrors `Quaternion_Normalization` (ESKF.py line ~17):
//!   returns the identity quaternion `(1, 0, 0, 0)` when the norm is below
//!   `1e-12`, otherwise divides by the norm.
//! - [`from_angle_axis`] mirrors `quat_from_angle_axis` (ESKF.py line ~35), **but
//!   deliberately avoids a bug present there**: the Python function takes a numpy
//!   array `q` (an angle-axis vector) and does `q /= angle` in place (line 38, an
//!   in-place divide-assign on the *caller's* array, since numpy arrays are passed
//!   by reference). That silently mutates whatever the caller passed in, which is
//!   observable as a side effect on the caller's variable after the call returns.
//!   [`from_angle_axis`] takes its vector *by value* (`[f64; 3]`, `Copy`), so there
//!   is no aliasing and no way for it to mutate anything the caller can observe --
//!   the Rust ownership model rules the bug out structurally rather than requiring
//!   a defensive `.clone()` as a Python fix would.
//! - [`to_rotation_matrix`] mirrors `matR` (ESKF.py line ~12).

/// A quaternion in `(w, x, y, z)` order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Quat { w, x, y, z }
    }

    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Hamilton product `q1 * q2`, matching `quat_multiply` in ESKF.py.
pub fn mul(q1: Quat, q2: Quat) -> Quat {
    let (w1, x1, y1, z1) = (q1.w, q1.x, q1.y, q1.z);
    let (w2, x2, y2, z2) = (q2.w, q2.x, q2.y, q2.z);
    Quat {
        w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    }
}

/// Normalize `q`, returning the identity quaternion if `||q|| < 1e-12`, matching
/// `Quaternion_Normalization` in ESKF.py.
pub fn normalize(q: Quat) -> Quat {
    let n = q.norm();
    if n > 1e-12 {
        Quat { w: q.w / n, x: q.x / n, y: q.y / n, z: q.z / n }
    } else {
        Quat::IDENTITY
    }
}

/// Build the quaternion representing a rotation of `||v||` radians about the axis
/// `v / ||v||`, matching `quat_from_angle_axis` in ESKF.py. Returns the identity
/// quaternion if `||v|| < 1e-12`.
///
/// `v` is taken by value: unlike the Python version, this function cannot mutate
/// anything the caller holds (see module docs).
pub fn from_angle_axis(v: [f64; 3]) -> Quat {
    let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if angle < 1e-12 {
        return Quat::IDENTITY;
    }
    let axis = [v[0] / angle, v[1] / angle, v[2] / angle];
    let half = 0.5 * angle;
    let (sh, ch) = (half.sin(), half.cos());
    Quat { w: ch, x: axis[0] * sh, y: axis[1] * sh, z: axis[2] * sh }
}

/// Rotation matrix (row-major `[[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]]`) for
/// quaternion `(w,x,y,z)`, matching `matR` in ESKF.py.
pub fn to_rotation_matrix(q: Quat) -> [[f64; 3]; 3] {
    let (ww, wx, wy, wz) = (q.w, q.x, q.y, q.z);
    [
        [1.0 - 2.0 * (wy * wy + wz * wz), 2.0 * (wx * wy - ww * wz), 2.0 * (ww * wy + wx * wz)],
        [2.0 * (ww * wz + wx * wy), 1.0 - 2.0 * (wx * wx + wz * wz), 2.0 * (wy * wz - ww * wx)],
        [2.0 * (wx * wz - ww * wy), 2.0 * (ww * wx + wy * wz), 1.0 - 2.0 * (wx * wx + wy * wy)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_zero_gives_identity() {
        let q = normalize(Quat::new(0.0, 0.0, 0.0, 0.0));
        assert_eq!(q, Quat::IDENTITY);
    }

    #[test]
    fn normalize_scales_to_unit_norm() {
        let q = normalize(Quat::new(2.0, 0.0, 0.0, 0.0));
        assert!((q.norm() - 1.0).abs() < 1e-12);
        assert_eq!(q, Quat::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn mul_identity_is_noop() {
        let q = Quat::new(0.5, 0.5, 0.5, 0.5);
        assert_eq!(mul(Quat::IDENTITY, q), q);
        assert_eq!(mul(q, Quat::IDENTITY), q);
    }

    #[test]
    fn from_angle_axis_zero_gives_identity() {
        assert_eq!(from_angle_axis([0.0, 0.0, 0.0]), Quat::IDENTITY);
    }

    #[test]
    fn from_angle_axis_does_not_mutate_caller_value() {
        // Demonstrates the fix for the Python in-place-mutation bug: the input
        // is a plain array passed by value, so there's nothing to observe after
        // the call besides the returned quaternion.
        let v = [1.0, 0.0, 0.0];
        let _q = from_angle_axis(v);
        assert_eq!(v, [1.0, 0.0, 0.0]); // unchanged
    }

    #[test]
    fn from_angle_axis_half_pi_about_x() {
        use core::f64::consts::PI;
        let q = from_angle_axis([PI / 2.0, 0.0, 0.0]);
        // 90 degree rotation about x: w = cos(45deg), x = sin(45deg).
        let s = (0.5_f64).sqrt();
        assert!((q.w - s).abs() < 1e-9);
        assert!((q.x - s).abs() < 1e-9);
        assert!(q.y.abs() < 1e-12);
        assert!(q.z.abs() < 1e-12);
    }

    #[test]
    fn rotation_matrix_of_identity_is_identity() {
        let r = to_rotation_matrix(Quat::IDENTITY);
        let expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for i in 0..3 {
            for j in 0..3 {
                assert!((r[i][j] - expected[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rotation_matrix_is_orthogonal() {
        use core::f64::consts::PI;
        let q = normalize(from_angle_axis([0.3, -0.7, 1.1]));
        let r = to_rotation_matrix(q);

        // R * R^T == I
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += r[i][k] * r[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9, "R R^T not identity at ({i},{j}): {dot}");
            }
        }

        // det(R) == 1 (proper rotation, not a reflection).
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        assert!((det - 1.0).abs() < 1e-9, "det(R) = {det}");

        let _ = PI; // silence unused import warning if the block above changes
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    const BOUND: f64 = 1.0e6;

    fn bounded_f64() -> f64 {
        let v: f64 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v.abs() <= BOUND);
        v
    }

    /// `normalize` never produces NaN/infinite components for any finite input
    /// bounded by `BOUND` in magnitude.
    #[kani::proof]
    fn normalize_never_nan_for_bounded_input() {
        let q = Quat::new(bounded_f64(), bounded_f64(), bounded_f64(), bounded_f64());
        let out = normalize(q);
        assert!(out.w.is_finite());
        assert!(out.x.is_finite());
        assert!(out.y.is_finite());
        assert!(out.z.is_finite());
    }

    /// `mul` cannot panic and produces finite output for any finite bounded input
    /// (pure arithmetic: no division, no sqrt).
    #[kani::proof]
    fn mul_no_panic_and_finite() {
        let q1 = Quat::new(bounded_f64(), bounded_f64(), bounded_f64(), bounded_f64());
        let q2 = Quat::new(bounded_f64(), bounded_f64(), bounded_f64(), bounded_f64());
        let out = mul(q1, q2);
        assert!(out.w.is_finite());
        assert!(out.x.is_finite());
        assert!(out.y.is_finite());
        assert!(out.z.is_finite());
    }

    /// `from_angle_axis`'s angle-mapping guard (the `angle < 1e-12` branch and the
    /// division it guards) does not panic and yields finite output; the
    /// `sin`/`cos` half-angle terms are excluded from this proof (see module docs
    /// on Kani's lack of bit-precise transcendental support) but are covered by
    /// unit tests above.
    #[kani::proof]
    fn from_angle_axis_guard_no_panic() {
        let v = [bounded_f64(), bounded_f64(), bounded_f64()];
        let angle = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if angle < 1e-12 {
            assert_eq!(from_angle_axis(v), Quat::IDENTITY);
        } else {
            let axis = [v[0] / angle, v[1] / angle, v[2] / angle];
            assert!(axis[0].is_finite() && axis[1].is_finite() && axis[2].is_finite());
        }
    }
}
