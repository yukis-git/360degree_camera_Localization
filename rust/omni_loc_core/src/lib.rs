//! Dependency-free, pure-function core intended for direct reuse by the future
//! Rust port of the Python 360-degree-camera EKF-SLAM localization prototype.
//!
//! Each module documents the Python source it is ported from and carries both
//! ordinary unit tests and (where feasible) Kani proof harnesses under
//! `#[cfg(kani)]`.

pub mod pixel;
pub mod quat;
pub mod state_layout;
