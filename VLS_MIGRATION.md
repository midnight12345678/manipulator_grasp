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
- `vls_bridge/action_mapping.py`：策略动作到 MuJoCo 控制量映射
- `vls_bridge/steering.py`：引导算法核心
- `vls_bridge/task_runner.py`：端到端 rollout
- `vls_mujoco_runner.py`：CLI 入口
- `vls_config.example.json`：示例配置

## 使用方式

1. 在配置中选择策略后端：
   - `backend: "random"`：随机策略（用于联调流程）
   - `backend: "torchscript"`：加载 TorchScript 文件
   - `backend: "factory"`：通过 `module.sub:build_fn` 注入你自己的模型加载逻辑
2. 配置动作映射：
   - `action_mode`: `joint_delta` 或 `joint_absolute`
   - `input_normalized`: 是否将策略输出按 `[-1, 1]` 解释
   - `gripper_mode`: `absolute` 或 `delta`
3. 运行：

```bash
python vls_mujoco_runner.py --config vls_config.example.json
```

## 接口约定

- 观测字段：
  - `proprio`：关节位置与速度拼接
  - `rgb` / `depth`：相机观测
  - `action`：当前控制量
- 动作字段：
  - 默认按 MuJoCo actuator 顺序写入 `mj_data.ctrl`
  - 自动按 `actuator_ctrlrange` 进行裁剪
  - 支持从策略动作到控制量的映射（关节增量/绝对值，夹爪绝对/增量）

## 个性化任务

- 通过 `runtime.instruction` 传入自然语言任务。
- 默认 `SimpleGuidanceProvider` 为可替换占位实现；可在 `task_runner.py` 中接入你的 VLM 与关键点检测/跟踪系统。

## 重要说明

- 当前实现已完成 VLS 核心“接口迁移 + 数据匹配 + 引导机制”骨架接入。
- 当前实现已打通“策略加载 + 动作映射 + 引导采样 + 环境执行”的完整链路，可直接运行。
- 若要达到开源项目同等级效果，请进一步接入：
  - 真实 VLM（OpenAI/Anthropic 等）
  - 关键点检测与时序跟踪模块
  - 你的预训练扩散模型或 Pi 模型权重与训练时一致的前后处理
