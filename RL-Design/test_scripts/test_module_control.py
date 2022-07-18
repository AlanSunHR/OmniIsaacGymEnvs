from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
import numpy as np


from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView

import carb


module_usd_path = "~/SHR/RL-Design/module_urdf/overconstrained_module/overconstrained_module.usd"

class DofModule(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "DofModule",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = module_usd_path

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

#launch Isaac Sim before any other imports
#default first two lines in any standalone application
world = World()
world.scene.add_default_ground_plane()

module = DofModule("/World/Module")
module_view = ArticulationView(prim_paths_expr="/World/Module",
                               name="module_view")
world.scene.add(module_view)

# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()
for i in range(500):
    dof_pos = module_view.get_joint_positions(clone=False)
    print(dof_pos)
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim