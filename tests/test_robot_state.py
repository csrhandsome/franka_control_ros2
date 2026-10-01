"""Shared EE and joint data definitions used by collection and inference."""

import numpy as np
import pytest

from control.robot_state import EEPose, JointPose, quat_angle_xyzw
from control.util.pose import (
    matrix_to_quat_xyzw,
    position_quat_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_matrix,
    quat_xyzw_to_wxyz,
)


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


@pytest.mark.parametrize(
    "quat",
    [
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0, 0.0]),
        np.array([0.2, -0.4, 0.7, 0.5]),
    ],
)
def test_shared_quaternion_conversions_roundtrip(quat):
    quat = quat / np.linalg.norm(quat)
    np.testing.assert_allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(quat)), quat)
    rotation = quat_xyzw_to_matrix(quat)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert quat_angle_xyzw(matrix_to_quat_xyzw(rotation), quat) < 1e-7
    transform = position_quat_to_matrix(np.array([0.3, -0.1, 0.5]), quat)
    np.testing.assert_allclose(transform[:3, :3], rotation)
    np.testing.assert_allclose(transform[:3, 3], [0.3, -0.1, 0.5])
    assert quat_angle_xyzw(quat, -quat) == pytest.approx(0.0)


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError, match="nonzero"):
        quat_xyzw_to_matrix(np.zeros(4))
    with pytest.raises(ValueError, match="nonzero"):
        quat_angle_xyzw(np.zeros(4), np.array([0.0, 0.0, 0.0, 1.0]))
