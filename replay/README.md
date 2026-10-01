# Replay 数据回放

独立的本地数据工作台：React + Vite + TypeScript + Tailwind CSS 前端，FastAPI 后端，使用 `pnpm` / `uv`，不依赖 ROS、机器人连接或 Docker。现有采集、训练环境不需要改动。

从仓库根目录运行：

```bash
./replay/dev.sh
```

脚本安装锁定依赖，首次生成两套演示数据，并同时启动前后端；打开 **http://127.0.0.1:5173**，API 文档在 **http://127.0.0.1:8000/docs**。`Ctrl+C` 关闭两个进程组。需要本机有 `uv`、`pnpm`、Node.js 20.19+ / 22.12+ 和 Linux `setsid`。

也可以分开启动，便于调试：

```bash
uv sync --project replay
env -u PYTHONPATH uv run --project replay python -m replay.scripts.generate_demo
pnpm --dir replay/frontend install --frozen-lockfile

# 终端 1，工作目录为仓库根目录
env -u PYTHONPATH uv run --project replay uvicorn replay.backend.main:app --reload --host 127.0.0.1 --port 8000

# 终端 2，工作目录为仓库根目录
pnpm --dir replay/frontend dev --host 127.0.0.1
```

## 页面操作

左侧选择数据集、episode，然后查看字段的类型、shape 和名称。将视频或 EE 数据块拖到右侧参考栏，也可用添加按钮完成同样操作。添加多个视图后，用底部统一时间轴播放、暂停或跳转，同时查看相机画面、EE 空间轨迹和 XYZ 随时间的变化。视图可以单独移除或全部清空；切换 dataset / episode 会重置工作台。

目前只启用 MP4 视频和 EE 可视化，其他字段仍完整列在数据栏中。嵌入 Parquet 的 `image`、关节、动作、力等字段暂未增加独立可视化。EE 空间图为带坐标轴的投影，时间曲线使用 episode 内的秒数，位置单位为米。

## 演示数据和解析脚本

默认数据在 `replay/demo_data/`，由脚本生成，未纳入 Git。两套数据均含三个 episode、两路实际 H.264 MP4 视频以及与本项目一致的 `ee_pose`、`joint_position`、`gripper_position`、`actions` 等字段。每个 episode 为 30 Hz、180 帧，演示场景为合成工作台，不包含真实机器人记录。

- `demo_v21`：`meta/episodes.jsonl`，每个 episode 一个 Parquet / MP4。
- `demo_v30`：`meta/episodes/...parquet`，多个 episode 共用 Parquet / MP4；episode 1、2 有非零视频偏移，用于验证正确读取和播放。

目录与元数据依据 [LeRobot 官方格式说明](https://huggingface.co/docs/lerobot/main/porting_datasets_v3)。解析器直接读取 JSON、JSONL 和 Parquet，无需安装 LeRobot / PyTorch；这是本地回放读取支持，未声称覆盖整个 LeRobot 训练 SDK。

```bash
# 生成 / 重新生成专用演示数据
env -u PYTHONPATH uv run --project replay python -m replay.scripts.generate_demo

# 查看数据集结构和字段
env -u PYTHONPATH uv run --project replay python -m replay.scripts.inspect_dataset replay/demo_data/demo_v21
env -u PYTHONPATH uv run --project replay python -m replay.scripts.inspect_dataset replay/demo_data/demo_v30 --episode 1
```

配置自己的本地数据：

```bash
# 可以指向一个数据集，或包含多个数据集的父目录
REPLAY_DATA_ROOT=/absolute/path/to/datasets ./replay/dev.sh
```

后端只允许读取配置目录内的数据，HTTP 参数不接受任意磁盘路径；不写入真实数据。`meta/info.json` 中声明的模板、v3 episode 的 chunk/file 索引和视频时间边界用于解析文件，而不是假设所有 episode 都从同一个文件的零秒开始。v2.0 / v2.1 和 v3.0 自动识别；未知版本和损坏数据会返回明确错误。

## 代码分层

```text
replay/
├── frontend/src/
│   ├── pages/         # 页面组合和工作台状态
│   ├── routes/        # 页面路由
│   ├── components/    # 数据栏、数据块、参考栏、视图卡片、视频、EE、时间轴
│   ├── hooks/         # 数据加载 / 播放逻辑
│   ├── lib/           # API 调用和纯工具
│   └── types/         # API / 界面类型
├── backend/
│   ├── api/           # HTTP 路由和依赖
│   ├── services/      # 一个公共业务函数一个文件
│   └── main.py        # FastAPI app / app factory
├── scripts/           # 双格式解析、演示生成、独立查看 CLI
├── tests/             # 先解析、后 API 验证
├── pyproject.toml     # 独立 Python 项目
└── dev.sh             # 本地启动
```

业务调用关系：路由 → 对应 service 函数 → `read_dataset` / `read_episode` / `read_ee` / `resolve_video`。视频元数据与视频文件响应分开，文件接口支持 Range 请求，供浏览器 seek；v3 播放器依据片段起止时间限制在当前 episode 内。前后端约定可见 [CONTRACT.md](CONTRACT.md)，完整接口可在 `/docs` 查看。

## 验证

一键按顺序验证：`./replay/check.sh`。也可以先验证两种假数据的读取，再验证后端接口，最后构建前端：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project replay pytest replay/tests/test_readers.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project replay pytest replay/tests/test_api.py -q
pnpm --dir replay/frontend build
env -u PYTHONPATH uv run --project replay ruff check replay
```

测试使用临时合成数据，不修改已有真实数据。命令清除宿主机的 `PYTHONPATH`，并禁用 pytest 插件自动加载，避免 ROS shell 环境影响独立 uv 项目。重点检查 v3 共享文件中的 episode 隔离、视频偏移、EE 降采样首尾点、缺失字段 / 文件、路径约束，以及视频 HTTP Range 响应。

浏览器测试会自动启动本地 API 和 Vite，验证拖拽、真实视频解码、EE 曲线/空间轨迹、共享视频片段边界、窄屏操作及错误重试：

```bash
# 默认使用本机 Google Chrome
pnpm --dir replay/frontend test:e2e

# 没有 Google Chrome 时，可安装 Playwright 的 Chromium
pnpm --dir replay/frontend exec playwright install chromium
REPLAY_BROWSER_CHANNEL=chromium pnpm --dir replay/frontend test:e2e
```
