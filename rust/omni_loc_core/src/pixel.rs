//! Equirectangular pixel -> (azimuth, elevation) -> unit bearing vector.
//!
//! Ported from the Python prototype:
//!   - `SLAM_on_cpu.py`, `Observation._trans_func_vectorized` and `Observation.data`
//!     (image height `H`, width `2H`, branch at `x = 1.5*H` that keeps the azimuth
//!     wrapped into `[-pi, pi]` instead of letting it run past `pi`).
//!   - `ExKalmanFilter.py`, `Obsevation_AR.calc_obs` (fixed `W = 1280`), which uses
//!     the *same* linear coefficient `2*pi/W == pi/H` (with `W = 2*H`) but skips the
//!     `x >= 1.5*H` branch entirely. The Python comment there literally says
//!     "this mapping might need refinement" -- for `x` in roughly `(1.5H, 2H]` the
//!     un-branched formula produces an azimuth below `-pi` (out of range), which the
//!     `SLAM_on_cpu.py` version avoids via the second branch. This module follows the
//!     `SLAM_on_cpu.py` (correct / wrapped) behavior.
//!
//! Deriving the closed form (H = picv, elevation column reuses the *same*
//! coefficient in both branches, azimuth uses the `+pi` shift in the second branch):
//!   - elevation:  theta = pi/2 - pi*y/H                                  , y in [0, H]
//!   - azimuth:    phi   = pi/2 - pi*x/H                for 0   <= x < 1.5H
//!                 phi   = 5*pi/2 - pi*x/H              for 1.5H <= x <= 2H
//! The two branches meet at the panorama seam (x = 0 and x = 2H both map to
//! phi = pi/2), and the union of ranges is exactly `(-pi, pi]`.
//!
//! Kani note: transcendental functions (`sin`/`cos`) are not modeled bit-precisely by
//! CBMC/Kani (calls are either unsupported or treated as an uninterpreted/havoc'd
//! function depending on the backend), so this module is split in two:
//!   - `pixel_to_angles_*`: pure comparisons/arithmetic, no trig -- proven with Kani.
//!   - `angles_to_bearing_*`: `sin`/`cos` calls -- covered by unit tests only.

/// Convert an equirectangular pixel coordinate to (azimuth, elevation) in radians,
/// following `SLAM_on_cpu.py`'s `Observation._trans_func_vectorized`.
///
/// `h` is the image height (`picv` in the Python code); the image width is `2*h`.
///
/// Returns `None` when the input is invalid:
///   - `h` is not finite or `h <= 0`
///   - `x` or `y` is not finite
///   - `x` is outside `[0, 2*h]`
///   - `y` is outside `[0, h]`
///
/// This differs from the Python code, which silently produces `(0.0, 0.0)` (via
/// `np.zeros_like`) for pixels outside the two branch conditions instead of
/// signalling an error -- see the module docs for the "simplified model" bug in
/// `ExKalmanFilter.py`'s `calc_obs` for a related, unbounded-output edge case.
pub fn pixel_to_angles_f64(x: f64, y: f64, h: f64) -> Option<(f64, f64)> {
    if !h.is_finite() || h <= 0.0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    if x < 0.0 || x > 2.0 * h || y < 0.0 || y > h {
        return None;
    }

    let pi = core::f64::consts::PI;
    let pi_div_h = pi / h;

    let elevation = pi / 2.0 - pi_div_h * y;

    let azimuth = if x < 1.5 * h {
        pi / 2.0 - pi_div_h * x
    } else {
        2.5 * pi - pi_div_h * x
    };

    Some((azimuth, elevation))
}

/// `f32` counterpart of [`pixel_to_angles_f64`].
pub fn pixel_to_angles_f32(x: f32, y: f32, h: f32) -> Option<(f32, f32)> {
    if !h.is_finite() || h <= 0.0 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    if x < 0.0 || x > 2.0 * h || y < 0.0 || y > h {
        return None;
    }

    let pi = core::f32::consts::PI;
    let pi_div_h = pi / h;

    let elevation = pi / 2.0 - pi_div_h * y;

    let azimuth = if x < 1.5 * h {
        pi / 2.0 - pi_div_h * x
    } else {
        2.5 * pi - pi_div_h * x
    };

    Some((azimuth, elevation))
}

/// Convert (azimuth, elevation) to a unit bearing vector `(x, y, z)`, following
/// `SLAM_on_cpu.py`'s `Observation.data`:
///   x = cos(elevation) * cos(azimuth)
///   y = cos(elevation) * sin(azimuth)
///   z = sin(elevation)
pub fn angles_to_bearing_f64(azimuth: f64, elevation: f64) -> (f64, f64, f64) {
    let (sa, ca) = (azimuth.sin(), azimuth.cos());
    let (se, ce) = (elevation.sin(), elevation.cos());
    (ce * ca, ce * sa, se)
}

/// `f32` counterpart of [`angles_to_bearing_f64`].
pub fn angles_to_bearing_f32(azimuth: f32, elevation: f32) -> (f32, f32, f32) {
    let (sa, ca) = (azimuth.sin(), azimuth.cos());
    let (se, ce) = (elevation.sin(), elevation.cos());
    (ce * ca, ce * sa, se)
}

/// Full pixel -> unit bearing vector pipeline (`f64`). `None` on invalid input.
pub fn pixel_to_bearing_f64(x: f64, y: f64, h: f64) -> Option<(f64, f64, f64)> {
    let (az, el) = pixel_to_angles_f64(x, y, h)?;
    Some(angles_to_bearing_f64(az, el))
}

/// Full pixel -> unit bearing vector pipeline (`f32`). `None` on invalid input.
pub fn pixel_to_bearing_f32(x: f32, y: f32, h: f32) -> Option<(f32, f32, f32)> {
    let (az, el) = pixel_to_angles_f32(x, y, h)?;
    Some(angles_to_bearing_f32(az, el))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn seam_matches_on_both_sides() {
        // x = 0 and x = 2H must map to the same azimuth (panorama seam).
        let h = 240.0_f64;
        let (az0, el0) = pixel_to_angles_f64(0.0, h / 2.0, h).unwrap();
        let (az1, el1) = pixel_to_angles_f64(2.0 * h, h / 2.0, h).unwrap();
        assert!((az0 - az1).abs() < 1e-9);
        assert!((el0 - el1).abs() < 1e-9);
        assert!((az0 - PI / 2.0).abs() < 1e-9);
    }

    #[test]
    fn branch_boundary_wraps_to_minus_pi_and_pi() {
        let h = 240.0_f64;
        // Just below 1.5H (branch 1): azimuth approaches -pi.
        let (az_lo, _) = pixel_to_angles_f64(1.5 * h - 1e-6, 0.0, h).unwrap();
        assert!(az_lo < -PI + 1e-3);
        // Exactly at 1.5H (branch 2): azimuth is exactly pi.
        let (az_hi, _) = pixel_to_angles_f64(1.5 * h, 0.0, h).unwrap();
        assert!((az_hi - PI).abs() < 1e-9);
    }

    #[test]
    fn elevation_extremes() {
        let h = 240.0_f64;
        let (_, el_top) = pixel_to_angles_f64(0.0, 0.0, h).unwrap();
        let (_, el_bottom) = pixel_to_angles_f64(0.0, h, h).unwrap();
        assert!((el_top - PI / 2.0).abs() < 1e-9);
        assert!((el_bottom + PI / 2.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_inputs_return_none() {
        let h = 240.0_f64;
        assert!(pixel_to_angles_f64(-1.0, 0.0, h).is_none());
        assert!(pixel_to_angles_f64(2.0 * h + 1.0, 0.0, h).is_none());
        assert!(pixel_to_angles_f64(0.0, -1.0, h).is_none());
        assert!(pixel_to_angles_f64(0.0, h + 1.0, h).is_none());
        assert!(pixel_to_angles_f64(0.0, 0.0, 0.0).is_none());
        assert!(pixel_to_angles_f64(0.0, 0.0, -1.0).is_none());
        assert!(pixel_to_angles_f64(f64::NAN, 0.0, h).is_none());
        assert!(pixel_to_angles_f64(0.0, f64::INFINITY, h).is_none());
        assert!(pixel_to_angles_f64(0.0, 0.0, f64::NAN).is_none());
    }

    #[test]
    fn bearing_is_unit_length() {
        let h = 240.0_f64;
        let (x, y, z) = pixel_to_bearing_f64(100.0, 50.0, h).unwrap();
        let n = (x * x + y * y + z * z).sqrt();
        assert!((n - 1.0).abs() < 1e-9);
    }

    #[test]
    fn f32_matches_f64_closely() {
        let h64 = 240.0_f64;
        let h32 = 240.0_f32;
        let (az64, el64) = pixel_to_angles_f64(100.3, 50.7, h64).unwrap();
        let (az32, el32) = pixel_to_angles_f32(100.3, 50.7, h32).unwrap();
        assert!((az64 as f32 - az32).abs() < 1e-4);
        assert!((el64 as f32 - el32).abs() < 1e-4);
    }

    // A small direct check that the un-branched "simplified model" seen in
    // ExKalmanFilter.py's calc_obs (W = 2H, no wrap branch) can go out of the
    // documented [-pi, pi] azimuth range, unlike this module's wrapped formula.
    #[test]
    fn python_calc_obs_style_formula_can_exceed_minus_pi() {
        let h = 240.0_f64;
        let w = 2.0 * h;
        let x = 1.9 * h; // inside this module's branch-2 region
        let phi_simplified = -2.0 * PI / w * x + PI / 2.0; // ExKalmanFilter.py calc_obs, no branch
        assert!(phi_simplified < -PI, "simplified formula escapes [-pi, pi]: {phi_simplified}");

        let (phi_wrapped, _) = pixel_to_angles_f64(x, 0.0, h).unwrap();
        assert!((-PI..=PI).contains(&phi_wrapped));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    const BOUND: f64 = 1.0e6;
    const BOUND32: f32 = 1.0e6;

    fn bounded_f64() -> f64 {
        let v: f64 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v.abs() <= BOUND);
        v
    }

    fn bounded_f32() -> f32 {
        let v: f32 = kani::any();
        kani::assume(v.is_finite());
        kani::assume(v.abs() <= BOUND32);
        v
    }

    /// No panic for any finite bounded input, and when valid the angles land in
    /// the documented ranges: azimuth in [-pi, pi], elevation in [-pi/2, pi/2].
    #[kani::proof]
    fn pixel_to_angles_f64_in_range_or_none() {
        let x = bounded_f64();
        let y = bounded_f64();
        let h = bounded_f64();

        let result = pixel_to_angles_f64(x, y, h);

        if let Some((az, el)) = result {
            let pi = core::f64::consts::PI;
            // Small epsilon for floating point rounding at the exact boundary.
            let eps = 1e-9 * (1.0 + BOUND);
            assert!(az >= -pi - eps && az <= pi + eps);
            assert!(el >= -pi / 2.0 - eps && el <= pi / 2.0 + eps);
        }
    }

    #[kani::proof]
    fn pixel_to_angles_f32_in_range_or_none() {
        let x = bounded_f32();
        let y = bounded_f32();
        let h = bounded_f32();

        let result = pixel_to_angles_f32(x, y, h);

        if let Some((az, el)) = result {
            let pi = core::f32::consts::PI;
            let eps = 1e-3 * (1.0 + BOUND32);
            assert!(az >= -pi - eps && az <= pi + eps);
            assert!(el >= -pi / 2.0 - eps && el <= pi / 2.0 + eps);
        }
    }

    /// Invalid inputs (out-of-range pixel coordinates, non-positive height) must
    /// return `None`, never a silently-wrong angle pair.
    #[kani::proof]
    fn invalid_inputs_yield_none() {
        let x = bounded_f64();
        let y = bounded_f64();
        let h = bounded_f64();

        // Force at least one invalid condition.
        kani::assume(x < 0.0 || x > 2.0 * h || y < 0.0 || y > h || h <= 0.0);

        assert!(pixel_to_angles_f64(x, y, h).is_none());
    }

    /// Non-finite input must return `None`.
    #[kani::proof]
    fn non_finite_yields_none() {
        let h: f64 = kani::any();
        kani::assume(h.is_finite() && h > 0.0 && h <= BOUND);
        assert!(pixel_to_angles_f64(f64::NAN, 0.0, h).is_none());
        assert!(pixel_to_angles_f64(0.0, f64::NAN, h).is_none());
        assert!(pixel_to_angles_f64(0.0, 0.0, f64::NAN).is_none());
        assert!(pixel_to_angles_f64(f64::INFINITY, 0.0, h).is_none());
    }
}
