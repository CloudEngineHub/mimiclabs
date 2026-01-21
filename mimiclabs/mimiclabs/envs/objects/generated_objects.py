import numpy as np
import xml.etree.ElementTree as ET

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.utils.transform_utils as T

from libero.libero.envs.base_object import register_object


@register_object
class Bin(CompositeObject):
    """
    Generates a four-walled bin container with an open top.
    Args:
        name (str): Name of this Bin object
        bin_size (3-array): (x,y,z) full size of bin
        wall_thickness (float): How thick to make walls of bin
        transparent_walls (bool): If True, walls will be semi-translucent
        friction (3-array or None): If specified, sets friction values for this bin. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name="bin",
        bin_size=(0.05, 0.05, 0.12),
        wall_thickness=0.01,
        transparent_walls=False,
        friction=None,
        density=1000.0,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.bin_size = np.array(bin_size)
        self.wall_thickness = wall_thickness
        self.transparent_walls = transparent_walls
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.bin_mat_name = "light_wood_mat"

        # Element references
        self._base_geom = "base"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Add contain_region site to object
        self._add_contain_region_site()

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        bin_mat = CustomMaterial(
            texture="WoodLight",
            tex_name="light_wood",
            mat_name=self.bin_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(bin_mat)

        # Add category name and object properties for framework integration
        self.category_name = "bin"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}

    def _add_contain_region_site(self):
        """
        Adds a contain_region site to the bin object.
        This site represents the usable volume inside the bin walls.
        """
        _obj = self.get_obj()

        # Calculate contain region size (inside the walls)
        contain_size_x = (self.bin_size[0] - 2 * self.wall_thickness) / 2
        contain_size_y = (self.bin_size[1] - 2 * self.wall_thickness) / 2
        contain_size_z = (self.bin_size[2] - self.wall_thickness) / 2

        # Position is centered horizontally and vertically above the base
        contain_pos_z = self.wall_thickness / 2

        site_str = f"<site \
            name='{self.name}_contain_region' \
            type='box' \
            pos='0 0 {contain_pos_z}' \
            quat='1 0 0 0' \
            size='{contain_size_x} {contain_size_y} {contain_size_z}' \
            rgba='0.8 0 0 0' \
            group='0' \
            />"
        site_obj = ET.fromstring(site_str)
        # Append site to root object
        _obj.append(site_obj)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": self.bin_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Base
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, -(self.bin_size[2] - self.wall_thickness) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=(
                np.array((self.bin_size[0], self.bin_size[1], self.wall_thickness))
                - np.array((self.wall_thickness, self.wall_thickness, 0))
            )
            / 2,
            geom_names=self._base_geom,
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )

        # Walls
        x_vals = np.array(
            [
                0,
                -(self.bin_size[0] - self.wall_thickness) / 2,
                0,
                (self.bin_size[0] - self.wall_thickness) / 2,
            ]
        )
        y_vals = np.array(
            [
                -(self.bin_size[1] - self.wall_thickness) / 2,
                0,
                (self.bin_size[1] - self.wall_thickness) / 2,
                0,
            ]
        )
        w_vals = np.array(
            [self.bin_size[0], self.bin_size[1], self.bin_size[0], self.bin_size[1]]
        )
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        if self.transparent_walls:
            wall_rgba = (1.0, 1.0, 1.0, 0.3)
            wall_mat = None
        else:
            wall_rgba = None if self.use_texture else self.rgba
            wall_mat = self.bin_mat_name if self.use_texture else None
        for i, (x, y, w, r) in enumerate(zip(x_vals, y_vals, w_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(x, y, 0),
                geom_quats=T.convert_quat(
                    T.axisangle2quat(np.array([0, 0, r])), to="wxyz"
                ),
                geom_sizes=(self.wall_thickness / 2, w / 2, self.bin_size[2] / 2),
                geom_names=f"wall{i}",
                geom_rgbas=wall_rgba,
                geom_materials=wall_mat,
                geom_frictions=self.friction,
            )

        # Add back in base args and site args
        obj_args.update(base_args)
        # obj_args.update({"joints": None})
        # Return this dict
        return obj_args

    @property
    def base_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to bin base
        """
        return [self.correct_naming(self._base_geom)]


@register_object
class MugTree(CompositeObject):
    """
    Generates a four-walled bin container with an open top.
    Args:
        name (str): Name of this object
        base_size (3-array): (x,y,z) full size of base
        branch_height (float): Height of the branch
        branch_size (3-array): (x,y,z) size of the branch
        tree_size (3-array): (x,y,z) size of the tree
        friction (3-array or None): If specified, sets friction values for this object. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        base_size=(0.16, 0.16, 0.03),
        branch_height=0.12,
        branch_size=(0.08, 0.005, 0.015),
        tree_size=(0.03, 0.03, 0.16),
        friction=None,
        density=5000.0,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.base_size = np.array(base_size)
        self.branch_height = branch_height
        self.branch_size = np.array(branch_size)
        self.tree_size = np.array(tree_size)
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.bin_mat_name = "light_wood_mat"

        # Element references
        self._base_geom = "base"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Add hang_region site to object
        self._add_hang_region_site()

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        bin_mat = CustomMaterial(
            texture="WoodLight",
            tex_name="light_wood",
            mat_name=self.bin_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(bin_mat)

        # Add category name and object properties for framework integration
        self.category_name = "mug_tree"
        self.rotation = None
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": (self.base_size + np.array([0, 0, self.tree_size[2]])) / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Base
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                0,
                0,
                self.base_size[2] / 2 - (self.tree_size[2] + self.base_size[2]) / 2,
            ),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.base_size / 2,
            geom_names=self._base_geom,
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )
        # Tree
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                0,
                0,
                self.base_size[2]
                + self.tree_size[2] / 2
                - (self.tree_size[2] + self.base_size[2]) / 2,
            ),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.tree_size / 2,
            geom_names="tree",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )
        # Branches
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                self.tree_size[0] / 2 + self.branch_size[0] / 2,
                0,
                self.base_size[2]
                + self.branch_height
                - (self.tree_size[2] + self.base_size[2]) / 2,
            ),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.branch_size / 2,
            geom_names="branches",
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.bin_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
        )
        # Add back in base args and site args
        obj_args.update(base_args)
        # obj_args.update({"joints": None})
        # Return this dict
        return obj_args

    def _add_hang_region_site(self):
        """
        Adds a hang_region site to the mug tree object.
        This site represents the volume where mugs can hang from the branch,
        extending down to 50% of the tree height.
        """
        _obj = self.get_obj()

        # Calculate hang region size
        # Width and depth based on branch size
        hang_size_x = self.branch_size[0] / 2
        hang_size_y = self.branch_size[0] / 2
        # Height is 50% of tree height
        hang_size_z = self.tree_size[2] / 2 * 0.5

        # Position: centered at branch location, below it
        # Branch x position
        hang_pos_x = self.tree_size[0] / 2 + self.branch_size[0] / 2
        hang_pos_y = 0
        # Position z: below the branch, at half the tree height
        hang_pos_z = (
            self.base_size[2]
            + self.branch_height
            - (self.tree_size[2] + self.base_size[2]) / 2
            - hang_size_z
        )

        site_str = f"<site \
            name='{self.name}_hang_region' \
            type='box' \
            pos='{hang_pos_x} {hang_pos_y} {hang_pos_z}' \
            quat='1 0 0 0' \
            size='{hang_size_x} {hang_size_y} {hang_size_z}' \
            rgba='0 0.8 0 0.3' \
            group='0' \
            />"
        site_obj = ET.fromstring(site_str)
        # Append site to root object
        _obj.append(site_obj)

    @property
    def base_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to bin base
        """
        return [self.correct_naming(self._base_geom)]
