"""Read a single episode's summary and complete feature catalog."""

from pathlib import Path

from ._common import _episode_summary, _feature_blocks, _get_episode, _load_metadata


def read_episode(root: Path, episode_index: int) -> dict:
    """Return episode metadata; unknown episodes raise KeyError."""
    info, episodes = _load_metadata(Path(root))
    episode = _get_episode(episodes, episode_index)
    return {**_episode_summary(info, episode), "blocks": _feature_blocks(info)}
