"""Regression test for a real customer rear three-quarter photo pattern."""

from app.services.segmentation import AngleFeatures, classify_angle


def build_features(**overrides: float) -> AngleFeatures:
    base = dict(
        bbox_aspect_ratio=2.05,
        bbox_fill_ratio=0.47,
        centroid_x_ratio=0.51,
        centroid_y_ratio=0.62,
        global_symmetry=0.54,
        upper_half_symmetry=0.48,
        lower_half_symmetry=0.60,
        upper_half_mass=0.44,
        lower_half_mass=0.56,
        left_half_mass=0.50,
        right_half_mass=0.43,
        solidity=0.88,
        extent=0.32,
        lower_left_circle_score=1.0,
        lower_right_circle_score=0.6,
        wheel_symmetry=0.6,
        horizontal_edge_ratio=0.44,
        vertical_edge_ratio=0.51,
        row_width_mean_ratio=0.74,
        row_width_std_ratio=0.16,
        wheel_band_ratio=0.20,
        lower_red_ratio=0.0047,
        upper_brightness=0.61,
        lower_brightness=0.44,
        brightness_gradient=0.17,
        dark_center_ratio=0.19,
    )
    base.update(overrides)
    return AngleFeatures(**base)


def test_three_quarter_rear_pattern() -> None:
    """
    Real-world pattern:
    - one wheel dominant, the other partially visible
    - taillights present but small
    - low symmetry from a rear three-quarter angle
    """
    angle, confidence, reason = classify_angle(build_features())

    assert angle in {"rear", "three_quarter"}, f"Expected rear/three_quarter, got {angle}. Reason: {reason}"
    assert confidence > 0.5, f"Confidence too low: {confidence}"
