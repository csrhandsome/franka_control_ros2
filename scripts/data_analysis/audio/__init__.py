"""音频相关的数据集处理：VAD、指令音频窗口、ASR 转写数据集生成。

流水线顺序：

    audio/vad.py        在 audio/episode_*.sync.json 里写入 vad_segments
      └─ audio/window.py  据此推导 instruction_audio_window
    audio/asr_convert.py  按 vad_segments 逐段转写，产出 *_asr 数据集
"""
