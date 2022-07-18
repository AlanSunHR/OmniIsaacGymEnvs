# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb

class Module(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Module",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = "./usds/urdf_closed.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

class TestModule(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_episode_length = 1000

        self._num_observations = 10

        self._num_dof = 3
        self._num_actions = self._num_dof
        self._dof_positions = torch.tensor([0, 0.7, -0.7], dtype=torch.float32)

        self._dof_names = ["a__dof1", "a__dof2", "a__dof3"]
        self._dof_indices = []
        self._dof_limits = []

        self._robot = Module
        self._robot_name = "TestModule"
        self._articulation_prim = "/module"

        RLTask.__init__(self, name, env)

        self._world_position = torch.tensor([0,0,0.5], device=self._device, dtype=torch.float32)
        self._world_rot = torch.tensor([1,0,0,0], device=self._device, dtype=torch.float32)   
        self._dof_positions.to(self._device)     
        
        return

    def set_up_scene(self, scene) -> None:
        self.get_robot()
        super().set_up_scene(scene)

        env_prims = scene.stage.GetPrimAtPath(self.default_zero_env_path)
        children_refs = env_prims.GetAllChildren()
        print("Env prims: ", children_refs)

        prim_ref = scene.stage.GetPrimAtPath(self.default_zero_env_path + "/" + self._robot_name)
        children_refs = prim_ref.GetChildren()
        print("Child prims: ", children_refs)

        arti_root_name = "/World/envs/.*/" + self._robot_name + self._articulation_prim
        self._robot_arti = ArticulationView(prim_paths_expr=arti_root_name, name="robot_view")
        scene.add(self._robot_arti)

        print("Articulation root name: ", arti_root_name)
        print("Articulation valid? ", self._robot_arti.is_valid())

        print("Finished setting up scenes! ")
        return

    def get_robot(self):
        _robot = self._robot(prim_path=self.default_zero_env_path + "/" + self._robot_name, name=self._robot_name, translation=self._world_position)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(self._robot_name, get_prim_at_path(_robot.prim_path), self._sim_config.parse_actor_config(self._robot_name))


    def get_observations(self) -> dict:
        dof_positions = self._robot_arti.get_joint_positions(clone=False)
        dof_velocities = self._robot_arti.get_joint_velocities(clone=False)

        self.obs_buf = torch.cat((dof_positions, dof_velocities), dim=-1)

        observations = {
            self._robot_arti.name: {
                "obs_buf": self.obs_buf
            }
        }

        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)


        if self.progress_buf[0] % 200 == 0:
            if self.progress_buf[0] >= 200:
                # Wait for a while to check the default position
                if self._dof_names:
                    self._robot_arti.set_joint_position_targets(actions, joint_indices=self._dof_indices)
                else:
                    self._robot_arti.set_joint_position_targets(actions)
            print("Progress: ", self.progress_buf[0])
        
        #long_dof_indices = torch.tensor(self._dof_indices, dtype=torch.long, device=self._device)
        #print("Joint errors: ", self._robot_arti.get_joint_positions()[:,long_dof_indices] - actions)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        reset_joint_velocities = torch.zeros(num_resets, self._num_actions)
        # Reset joint positions/velocities to default
        if self._dof_names: 
            #self._robot_arti.set_joint_position_targets(self._dof_positions, indices=env_ids, joint_indices=self._dof_indices)
            self._robot_arti.set_joint_positions(self._dof_positions, indices=env_ids, joint_indices=self._dof_indices)
            self._robot_arti.set_joint_velocities(reset_joint_velocities, indices=env_ids, joint_indices=self._dof_indices)
        else:
            self._robot_arti.set_joint_positions(self._dof_positions, indices=env_ids)
            self._robot_arti.set_joint_velocities(reset_joint_velocities, indices=env_ids)

        # Set to initial pose
        world_position = self._world_position.repeat(num_resets, 1)
        world_rot = self._world_rot.repeat(num_resets, 1)
        self._robot_arti.set_world_poses(world_position, world_rot, env_ids)
        # Set velocities to zeros
        zero_velocities = torch.zeros(num_resets, 6, device=self._device, dtype=torch.float32)
        self._robot_arti.set_velocities(zero_velocities, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self) -> None:
        # Read the dof indices
        if self._dof_names:
            # If dof_names has been defined
            for dof_name in self._dof_names:
                self._dof_indices.append(self._robot_arti.get_dof_index(dof_name))
            print("Finished reading drivable DoF indices: ")
            print("DoF names: ", self._robot_arti.dof_names)
            print("DoF indices: ", self._dof_indices)
            # Convert dof indices to torch tensor
            self._dof_indices = torch.tensor(self._dof_indices, device=self._device, dtype=torch.int32)

        # Reset all Envs at start
        indices = torch.arange(self._robot_arti.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def is_done(self):
        self.reset_buf[:] = is_done(
            self.reset_buf, 
            self.progress_buf,
            self._max_episode_length
        )

    def calculate_metrics(self) -> None:
        pass

@torch.jit.script
def is_done(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: float
) -> torch.Tensor:
    '''
        Only reset after max episode length reached
        for testing the robot
    '''
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    return reset
