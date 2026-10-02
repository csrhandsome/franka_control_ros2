"""Configuration inheritance tests; no robot connections or motion."""

from pathlib import Path

import pytest

from control.motion_config import load_motion_config, workflow_motion_config
from control.robot_config import flatten_config, load_mapping


def test_relative_inheritance_and_overrides(tmp_path, monkeypatch):
    (tmp_path / 'base.yaml').write_text('robot:\n  robot_type: panda\n  joints: [1, 2]\ncamera:\n  topic: /image\n  image_hw: 224\n')
    child = tmp_path / 'nested' / 'child.yaml'
    child.parent.mkdir()
    child.write_text('extends: ../base.yaml\nrobot:\n  joints: [3]\ncamera:\n  image_hw: 128\n')
    monkeypatch.chdir('/')
    result = load_mapping(child)
    assert result == {'robot': {'robot_type': 'panda', 'joints': [3]}, 'camera': {'topic': '/image', 'image_hw': 128}}
    assert flatten_config(child).robot_type == 'panda'
    result['robot']['joints'].append(4)
    assert load_mapping(child)['robot']['joints'] == [3]


def test_multiple_parents_and_legacy_file(tmp_path):
    (tmp_path / 'a.yaml').write_text('robot:\n  robot_type: panda\n  use_fake_hardware: false\n')
    (tmp_path / 'b.yaml').write_text('robot:\n  use_fake_hardware: true\n')
    child = tmp_path / 'child.yaml'
    child.write_text('extends: [a.yaml, b.yaml]\nrobot:\n  robot_type: panda\n')
    assert load_mapping(child)['robot'] == {'robot_type': 'panda', 'use_fake_hardware': True}
    assert load_mapping(tmp_path / 'b.yaml') == {'robot': {'use_fake_hardware': True}}


def test_cycles_invalid_parents_and_missing_file(tmp_path):
    a, b = tmp_path / 'a.yaml', tmp_path / 'b.yaml'
    a.write_text('extends: b.yaml\n')
    b.write_text('extends: a.yaml\n')
    with pytest.raises(ValueError, match='Cyclic'):
        load_mapping(a)
    for invalid in ('42', '[null]', '{}'):
        a.write_text(f'extends: {invalid}\n')
        with pytest.raises(TypeError, match='extends'):
            load_mapping(a)
    a.write_text('extends: missing.yaml\n')
    with pytest.raises(FileNotFoundError):
        load_mapping(a)
    a.write_text('- not-a-mapping\n')
    with pytest.raises(TypeError, match='mapping'):
        load_mapping(a)


def test_checked_in_workflows_and_motion_profiles():
    root = Path(__file__).resolve().parents[1] / 'config'
    configs = {stage: load_mapping(root / stage / 'franka.yaml') for stage in ('collect', 'inference', 'hitl')}
    assert all(cfg['robot']['robot_type'] == 'panda' for cfg in configs.values())
    assert configs['collect']['robot']['use_fake_hardware'] is False
    assert configs['inference']['robot']['use_fake_hardware'] is True
    assert configs['hitl']['camera']['camera_backend'] == 'none'
    assert configs['collect']['camera']['image_hw'] == 224
    assert configs['hitl']['camera']['image_hw'] == 128
    assert configs['collect']['camera']['camera_timeout_ms'] == 1000
    assert configs['hitl']['control']['reset_time_s'] == 5.0
    assert configs['collect']['camera']['external_image_topic'] == '/external/color/image_raw'
    panda = load_motion_config()
    assert panda['robot']['robot_type'] == 'panda'
    assert panda['joint']['streaming']['enabled'] is True
    assert panda['ee']['enabled'] is True
    assert panda['gripper']['enabled'] is True
    for cfg in configs.values():
        for mode in ('ee', 'joint'):
            motion = workflow_motion_config(cfg, control_mode=mode)
            assert motion['robot']['use_fake_hardware'] == cfg['robot']['use_fake_hardware']
            assert motion['gripper']['enabled'] is True
            assert motion['limits']['max_joint_excursion_rad'] == 3.2
    fake = load_mapping(root / 'collect' / 'franka_fake.yaml')
    assert fake['robot']['use_fake_hardware'] is True
    assert fake['camera']['camera_backend'] == 'none'
    assert not fake['dataset']['enable_logging']
    assert not workflow_motion_config(fake, control_mode='ee')['gripper']['enabled']
    with pytest.raises(ValueError, match='Only Panda'):
        load_motion_config(robot_type='fr3')


def test_motion_overrides_stay_nested_and_are_validated(tmp_path):
    child = tmp_path / 'collect.yaml'
    child.write_text('robot:\n  robot_type: panda\n  use_fake_hardware: false\nmotion:\n  robot:\n    use_fake_hardware: true\n  ee:\n    streaming:\n      publish_rate_hz: 30.0\n')
    flat = flatten_config(child)
    assert flat.use_fake_hardware is False
    assert flat.motion['ee']['streaming']['publish_rate_hz'] == 30.0
    motion = workflow_motion_config(load_mapping(child), control_mode='ee')
    assert motion['robot']['use_fake_hardware'] is False
    assert motion['ee']['streaming']['publish_rate_hz'] == 30.0
    assert load_motion_config()['ee']['streaming']['publish_rate_hz'] == 100.0
    with pytest.raises(ValueError, match='Unknown'):
        load_motion_config(overrides={'ee': {'enabeld': True}})
    with pytest.raises(ValueError, match='conflicts'):
        load_motion_config(overrides={'robot': {'robot_type': 'fr3'}})


@pytest.mark.parametrize('motion, mode, message', [
    ({'ee': {'enabled': False}}, 'ee', 'requires enabled ee streaming'),
    ({'joint': {'streaming': {'enabled': False}}}, 'joint', 'requires enabled joint streaming'),
    ({'gripper': {'enabled': False}}, 'ee', 'gripper_type=franka'),
    ({'ee': {'streaming': {'limits': {'max_translation_step_m': 0}}}}, 'ee', 'positive'),
    (None, 'ee', 'mapping'),
])
def test_workflow_rejects_unusable_motion_settings(motion, mode, message):
    with pytest.raises(ValueError, match=message):
        workflow_motion_config({'motion': motion}, control_mode=mode)
