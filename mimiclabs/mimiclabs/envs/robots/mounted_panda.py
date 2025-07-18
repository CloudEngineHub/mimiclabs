import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

from mimiclabs.mimiclabs.utils import get_robosuite_version


class MountedPanda(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.
    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]  # for robosuite v1.5

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
        )

    @property
    def default_mount(self):
        # for backward compatibility with robosuite v1.4
        return "RethinkMount"

    @property
    def default_base(self):
        # for robosuite v1.5
        return "RethinkMount"

    @property
    def default_gripper(self):
        if get_robosuite_version() < "1.5":
            # for backward compatibility with robosuite v1.4
            return "PandaGripper"
        return {"right": "PandaGripper"}

    @property
    def default_controller_config(self):
        if get_robosuite_version() < "1.5":
            # for backward compatibility with robosuite v1.4
            return "default_panda"
        return {"right": "default_panda"}

    @property
    def init_qpos(self):
        return np.array(
            [0, -1.61037389e-01, 0.00, -2.44459747e00, 0.00, 2.22675220e00, np.pi / 4]
        )

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
