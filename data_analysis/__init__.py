"""LeRobot 数据集分析与整理工具。

按功能分组：

- `lerobot_meta`  —— 共享的磁盘格式读写（被下面所有模块使用）
- `quality_check` —— 数据集质量检查与报告
- `episode_edit`  —— 按 episode_index 删除并重编号
- `dataset_merge` —— 合并多个数据集
- `route_labels`  —— 直行/绕行路线标注
- `audio/`        —— VAD 预处理、指令窗口推导、ASR 转写数据集
- `cot_pipeline`  —— 基于大模型的视频 CoT 分析（与本包其余部分无关）

所有命令都从仓库根用模块形式调用，例如：

    uv run -m data_analysis.quality_check --help
"""
