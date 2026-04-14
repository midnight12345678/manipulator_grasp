# VLS 到 MuJoCo 抓取环境迁移说明

本仓库新增了 `vls_bridge`，用于把 VLS（Vision-Language Steering）的推理时引导思想接入当前 UR5 MuJoCo 仿真环境。

## 新增能力

- 统一环境适配接口（`reset / step / get_obs / get_camera_image`）
- 兼容扩散策略与 Pi 系列策略的策略适配层
- 三种推理时引导机制：
  - 梯度引导（gradient refinement）
  - RBF 多样性项（diversity bonus）
  - Feynman–Kac 重采样（particle resampling）
- 任务入口脚本 `vls_mujoco_runner.py`，支持语言指令与配置文件驱动

## 目录

- `vls_bridge/config.py`：运行与引导配置
- `vls_bridge/env_adapter.py`：MuJoCo 环境适配
- `vls_bridge/policy_adapter.py`：扩散/Pi 策略适配
- `vls_bridge/steering.py`：引导算法核心
- `vls_bridge/task_runner.py`：端到端 rollout
- `vls_mujoco_runner.py`：CLI 入口
- `vls_config.example.json`：示例配置

## 使用方式

1. 将你的预训练策略封装为可调用对象，返回形状为 `[batch_size, horizon, action_dim]` 的动作序列。
2. 在 `vls_mujoco_runner.py` 的 `build_policy_adapter` 中替换 `RandomPolicy` 为你的模型加载与推理逻辑。
3. 运行：

```bash
python /home/runner/work/manipulator_grasp/manipulator_grasp/vls_mujoco_runner.py \
  --config /home/runner/work/manipulator_grasp/manipulator_grasp/vls_config.example.json
```

## 接口约定

- 观测字段：
  - `proprio`：关节位置与速度拼接
  - `rgb` / `depth`：相机观测
  - `action`：当前控制量
- 动作字段：
  - 默认按 MuJoCo actuator 顺序写入 `mj_data.ctrl`
  - 自动按 `actuator_ctrlrange` 进行裁剪

## 个性化任务

- 通过 `runtime.instruction` 传入自然语言任务。
- 默认 `SimpleGuidanceProvider` 为可替换占位实现；可在 `task_runner.py` 中接入你的 VLM 与关键点检测/跟踪系统。

## 重要说明

- 当前实现已完成 VLS 核心“接口迁移 + 数据匹配 + 引导机制”骨架接入。
- 若要达到开源项目同等级效果，请进一步接入：
  - 真实 VLM（OpenAI/Anthropic 等）
  - 关键点检测与时序跟踪模块
  - 你的预训练扩散模型或 Pi 模型权重与前后处理

