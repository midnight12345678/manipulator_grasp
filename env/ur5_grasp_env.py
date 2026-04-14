import os.path
import sys

sys.path.append('../../manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj


class UR5GraspEnv:

    def __init__(self, sim_hz: int = 500, show_gui: bool = True, camera_id: int = 0):
        self.sim_hz = sim_hz
        self.show_gui = show_gui
        self.camera_id = camera_id

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096
        self.ctrl_low = None
        self.ctrl_high = None

    def reset(self):
        filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.robot = UR5e()
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t)
        self.robot_q = np.array([0.0, 0.0, np.pi / 2 * 0, 0.0, -np.pi / 2 * 0, 0.0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                            "wrist_2_joint", "wrist_3_joint"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", self.robot.fkine(self.robot_q))
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(-np.pi / 2, -np.pi / 2, 0.0)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()

        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, self.camera_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.camera_id)
        self.mj_depth_renderer.enable_depth_rendering()
        if self.show_gui:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        else:
            self.mj_viewer = None

        if self.mj_model.actuator_ctrlrange is not None and len(self.mj_model.actuator_ctrlrange) > 0:
            self.ctrl_low = self.mj_model.actuator_ctrlrange[:, 0].copy()
            self.ctrl_high = self.mj_model.actuator_ctrlrange[:, 1].copy()
        else:
            self.ctrl_low = None
            self.ctrl_high = None

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # observation = self._get_obs()
        observation = None
        return observation

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

    def step(self, action=None):
        if action is not None:
            action = np.asarray(action, dtype=float)
            if self.ctrl_low is not None and self.ctrl_high is not None:
                action = np.clip(action, self.ctrl_low, self.ctrl_high)
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None:
            self.mj_viewer.sync()

    def render(self):
        self.mj_renderer.update_scene(self.mj_data, self.camera_id)
        self.mj_depth_renderer.update_scene(self.mj_data, self.camera_id)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }

    def get_obs(self):
        frames = self.render()
        return {
            "joint_q": self.mj_data.qpos[:6].copy(),
            "joint_dq": self.mj_data.qvel[:6].copy(),
            "action": self.mj_data.ctrl.copy(),
            "rgb": frames["img"],
            "depth": frames["depth"],
        }

    def get_camera_image(self):
        frames = self.render()
        return frames["img"], frames["depth"]


if __name__ == '__main__':
    env = UR5GraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
