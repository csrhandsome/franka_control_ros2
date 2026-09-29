"""Shared EE and joint data definitions used by collection and inference."""

import numpy as np
import pytest

from control.robot_state import EEPose, JointPose, quat_angle_xyzw


def test_ee_and_joint_actions_have_yaml_selected_dimensions():
    ee = EEPose.from_position_quat(
        np.array([0.4, -0.1, 0.5]), np.array([0.0, 0.0, 0.0, 1.0])
    )
    joint = JointPose(np.arange(7) * 0.1)
    assert ee.vector.shape == (6,)
    assert ee.action(1.0).shape == (7,)
    assert joint.vector.shape == (7,)
    assert joint.action(1.0).shape == (8,)
    assert ee.action(1.0)[-1] == joint.action(1.0)[-1] == 1.0
    with pytest.raises(ValueError):
        ee.action(1.2)


def test_ee_rpy_roundtrip_preserves_orientation():
    source = EEPose.from_vector(np.array([0.4, -0.1, 0.5, 0.2, -0.4, 3.0]))
    restored = EEPose.from_position_quat(source.xyz_m, source.quaternion_xyzw())
    np.testing.assert_allclose(restored.xyz_m, source.xyz_m, atol=1e-6)
    assert quat_angle_xyzw(restored.quaternion_xyzw(), source.quaternion_xyzw()) < 1e-5
