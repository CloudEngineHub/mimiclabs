import os
import re

from robosuite.models.objects import MujocoXMLObject
import xml.etree.ElementTree as ET

from libero.libero.envs.base_object import (
    register_object,
)

from ...utils import disable_module_import


with disable_module_import("robocasa"):
    from robocasa.models import assets_root as robocasa_assets_root
BASE_ASSET_PATH = os.path.join(robocasa_assets_root, "objects")
BASE_FIXTURE_PATH = os.path.join(robocasa_assets_root, "fixtures")


def parse_position(pos_str):
    """Convert position string to a list of floats."""
    return [float(val) for val in pos_str.split()]


def calculate_midpoint(pos1, pos2):
    """Calculate the midpoint between two positions."""
    return [(a + b) / 2 for a, b in zip(pos1, pos2)]


class RobocasaObject(MujocoXMLObject):
    def __init__(
        self, relative_path, name, joints=[dict(type="free", damping="0.0005")]
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, relative_path),
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=True,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = None
        self.rotation_axis = "x"

        articulation_object_properties = {
            "default_open_ranges": [],
            "default_close_ranges": [],
        }
        self.object_properties = {
            "articulation": articulation_object_properties,
            "vis_site_names": {},
        }


@register_object
class RobocasaAvocado0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_avocado_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/avocado/avocado_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaAvocado1(RobocasaObject):
    def __init__(
        self,
        name="robocasa_avocado_1",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/avocado/avocado_1/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaKiwi0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_kiwi_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/kiwi/kiwi_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot1(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_1",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_1/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot2(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_2",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_2/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_3",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_3/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot4(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_4",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_4/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot6(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_6",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_6/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCarrot7(RobocasaObject):
    def __init__(
        self,
        name="robocasa_carrot_7",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/carrot/carrot_7/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCucumber0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_cucumber_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/cucumber/cucumber_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaEgg0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_egg_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/egg/egg_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBagel4(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bagel_4",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bagel/bagel_4/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaShaker0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_shaker_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/shaker/shaker_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBroccoli0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_broccoli_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/broccoli/broccoli_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom1(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_1",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_1/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom2(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_2",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_2/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_3",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_3/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom4(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_4",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_4/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom5(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_5",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_5/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom6(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_6",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_6/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMushroom7(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mushroom_7",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mushroom/mushroom_7/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCorn0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_corn_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/corn/corn_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug1(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_1",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_1/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug9(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_9",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_9/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_3",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_3/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug8(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_8",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_8/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug5(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_5",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_5/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug2(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_2",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_16/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug4(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_4",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_12/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaMug6(RobocasaObject):
    def __init__(
        self,
        name="robocasa_mug_6",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/mug/mug_11/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaTeapot0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_teapot_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/teapot/teapot_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaTeapot3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_teapot_3",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/teapot/teapot_3/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaTeapot5(RobocasaObject):
    def __init__(
        self,
        name="robocasa_teapot_5",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/teapot/teapot_5/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaTeapot6(RobocasaObject):
    def __init__(
        self,
        name="robocasa_teapot_6",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/teapot/teapot_6/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaTeapot7(RobocasaObject):
    def __init__(
        self,
        name="robocasa_teapot_7",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/teapot/teapot_7/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan2(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_2",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_2/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_3",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_3/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan4(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_4",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_4/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan6(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_6",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_6/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan8(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_8",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_8/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaCan9(RobocasaObject):
    def __init__(
        self,
        name="robocasa_can_9",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/can/can_9/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaWaterBottle0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_water_bottle_0",
    ):
        super().__init__(
            os.path.join(
                BASE_ASSET_PATH, "objaverse/water_bottle/water_bottle_0/model.xml"
            ),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaWaterBottle1(RobocasaObject):
    def __init__(
        self,
        name="robocasa_water_bottle_1",
    ):
        super().__init__(
            os.path.join(
                BASE_ASSET_PATH, "objaverse/water_bottle/water_bottle_1/model.xml"
            ),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaWaterBottle2(RobocasaObject):
    def __init__(
        self,
        name="robocasa_water_bottle_2",
    ):
        super().__init__(
            os.path.join(
                BASE_ASSET_PATH, "objaverse/water_bottle/water_bottle_2/model.xml"
            ),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaWaterBottle3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_water_bottle_3",
    ):
        super().__init__(
            os.path.join(
                BASE_ASSET_PATH, "objaverse/water_bottle/water_bottle_3/model.xml"
            ),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl0(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_0",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_0/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl1(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_1",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_1/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl2(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_2",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_2/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl3(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_3",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_3/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl4(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_4",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_4/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl6(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_6",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_6/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl7(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_7",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_7/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaBowl8(RobocasaObject):
    def __init__(
        self,
        name="robocasa_bowl_8",
    ):
        super().__init__(
            os.path.join(BASE_ASSET_PATH, "objaverse/bowl/bowl_8/model.xml"),
            name=name,
            joints=[dict(type="free")],
        )


@register_object
class RobocasaToaster(RobocasaObject):
    def __init__(
        self,
        name="robocasa_toaster",
    ):
        xml_path = os.path.join(BASE_FIXTURE_PATH, "toasters/basic_popup/model.xml")
        tmp_xml = xml_path.replace("model.xml", "tmp.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        scale = 0.2

        # Find the mesh element and modify its scale
        for mesh in root.findall(".//mesh"):
            mesh.set("scale", str(scale) + " " + str(scale) + " " + str(scale))

        # Find all sites and scale their positions by 0.1
        for site in root.findall(".//site"):
            pos = site.get("pos")
            pos = " ".join([str(float(p) * scale) for p in pos.split()])
            site.set("pos", pos)

        # Find all the geoms of class collision and scale their size by 0.1
        for geom in root.findall(".//geom"):
            if geom.get("class") == "collision":
                size = geom.get("size")
                size = " ".join([str(float(s) * scale) for s in size.split()])
                geom.set("size", size)

        # Get position of first 4 sites in .//worldbody/body/
        for body in root.findall(".//worldbody/body"):
            site_positions = []
            for site in body.findall(".//site"):
                site_positions.append(parse_position(site.get("pos")))
            break

        # min and max y values from site positions
        min_y = min([pos[1] for pos in site_positions])
        max_y = max([pos[1] for pos in site_positions])
        min_x = min([pos[0] for pos in site_positions])
        max_x = max([pos[0] for pos in site_positions])
        min_z = min([pos[2] for pos in site_positions])
        max_z = max([pos[2] for pos in site_positions])

        # Calculate the positions for the new sites
        bottom_site_pos = [0, 0, min_z]
        top_site_pos = [0, 0, max_z]
        horizontal_radius_site_pos = [0, 0, 0]
        horizontal_radius_site_size = max(max_x - min_x, max_z - min_z) / 2.0

        # Find the body element within worldbody to add new sites
        for body in root.findall(".//worldbody/body"):
            # Define the new site elements to be added
            new_sites = [
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, bottom_site_pos)),
                    "name": "bottom_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, top_site_pos)),
                    "name": "top_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, horizontal_radius_site_pos)),
                    "name": "horizontal_radius_site",
                },
            ]

            # Add the new sites to the body element
            for site_info in new_sites:
                site = ET.SubElement(body, "site")
                for key, value in site_info.items():
                    site.set(key, value)

        # Create a temporary file to save the modified XML content
        tree.write(tmp_xml, encoding="utf-8", xml_declaration=True)

        super().__init__(tmp_xml, name=name, joints=[dict(type="free")])

        # delete the temporary file
        os.remove(tmp_xml)


@register_object
class RobocasaDrawer(RobocasaObject):
    def __init__(
        self,
        name="robocasa_drawer",
    ):
        xml_path = os.path.join(BASE_FIXTURE_PATH, "cabinets/drawer.xml")
        tmp_xml = xml_path.replace("drawer.xml", "tmp.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        scale = 0.1

        # Find the mesh element and modify its scale
        for mesh in root.findall(".//mesh"):
            mesh.set("scale", str(scale) + " " + str(scale) + " " + str(scale))

        # Find all sites and scale their positions by 0.1
        for site in root.findall(".//site"):
            pos = site.get("pos")
            pos = " ".join([str(float(p) * scale) for p in pos.split()])
            site.set("pos", pos)

        # Find all the geoms of class collision and scale their size by 0.1
        for geom in root.findall(".//geom"):
            if geom.get("class") == "collision":
                size = geom.get("size")
                size = " ".join([str(float(s) * scale) for s in size.split()])
                geom.set("size", size)

        # Get position of first 4 sites in .//worldbody/body/
        for body in root.findall(".//worldbody/body"):
            site_positions = []
            for site in body.findall(".//site"):
                site_positions.append(parse_position(site.get("pos")))
            break

        # create site in worldbody/body

        # min and max y values from site positions
        min_y = min([pos[1] for pos in site_positions])
        max_y = max([pos[1] for pos in site_positions])
        min_x = min([pos[0] for pos in site_positions])
        max_x = max([pos[0] for pos in site_positions])
        min_z = min([pos[2] for pos in site_positions])
        max_z = max([pos[2] for pos in site_positions])

        # Calculate the positions for the new sites
        bottom_site_pos = [0, 0, min_z]
        top_site_pos = [0, 0, max_z]
        horizontal_radius_site_pos = [0, 0, 0]
        horizontal_radius_site_size = max(max_x - min_x, max_z - min_z) / 2.0

        # Find the body element within worldbody to add new sites
        for body in root.findall(".//worldbody/body"):
            # Define the new site elements to be added
            new_sites = [
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, bottom_site_pos)),
                    "name": "bottom_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, top_site_pos)),
                    "name": "top_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, horizontal_radius_site_pos)),
                    "name": "horizontal_radius_site",
                },
            ]

            # Add the new sites to the body element
            for site_info in new_sites:
                site = ET.SubElement(body, "site")
                for key, value in site_info.items():
                    site.set(key, value)

        # Create a temporary file to save the modified XML content
        tree.write(tmp_xml, encoding="utf-8", xml_declaration=True)

        super().__init__(tmp_xml, name=name, joints=[dict(type="free")])

        # delete the temporary file
        # os.remove(tmp_xml)


@register_object
class RobocasaMicrowave(RobocasaObject):
    def __init__(
        self, name="robocasa_microwave", joints=[dict(type="free", damping="0.0005")]
    ):
        xml_path = os.path.join(BASE_FIXTURE_PATH, "microwaves/standard/model.xml")
        tmp_xml = xml_path.replace("model.xml", "tmp.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        scale = 0.3

        # Find the mesh element and modify its scale
        for mesh in root.findall(".//mesh"):
            mesh.set("scale", str(scale) + " " + str(scale) + " " + str(scale))

        # Find all sites and scale their positions by 0.1
        for site in root.findall(".//site"):
            pos = site.get("pos")
            pos = " ".join([str(float(p) * scale) for p in pos.split()])
            site.set("pos", pos)
            # set rgba to 0 0 0 0
            site.set("rgba", "0 0 0 0")

        for default in root.findall(".//default"):
            if default.get("class") == "collision":
                for geom in default.findall(".//geom"):
                    geom.set("rgba", "0 0 0 0")

        # Find all the geoms of class collision and scale their size by 0.1
        for geom in root.findall(".//geom"):
            if geom.get("class") == "collision" or (
                geom.get("class") == "visual" and geom.get("type") != "mesh"
            ):
                size = geom.get("size")
                size = " ".join([str(float(s) * scale) for s in size.split()])
                geom.set("size", size)
                pos = geom.get("pos")
                pos = " ".join([str(float(p) * scale) for p in pos.split()])
                geom.set("pos", pos)

        for joint in root.findall(".//joint"):
            if joint.get("name") == "microjoint":
                pos = joint.get("pos")
                pos = " ".join([str(float(p) * scale) for p in pos.split()])
                joint.set("pos", pos)
                joint.set("type", "hinge")

        # Get position of first 4 sites in .//worldbody/body/
        for body in root.findall(".//worldbody/body"):
            site_positions = []
            for site in body.findall(".//site"):
                site_positions.append(parse_position(site.get("pos")))
            break

        # min and max y values from site positions
        min_y = min([pos[1] for pos in site_positions])
        max_y = max([pos[1] for pos in site_positions])
        min_x = min([pos[0] for pos in site_positions])
        max_x = max([pos[0] for pos in site_positions])
        min_z = min([pos[2] for pos in site_positions])
        max_z = max([pos[2] for pos in site_positions])

        # Calculate the positions for the new sites
        bottom_site_pos = [0, 0, min_z]
        top_site_pos = [0, 0, max_z]
        horizontal_radius_site_pos = [0, 0, 0]
        horizontal_radius_site_size = max(max_x - min_x, max_z - min_z) / 2.0

        heating_region_pos = [0 * scale, 0.045 * scale, -0.204 * scale]
        heating_region_size = [0.5 * scale, 0.315 * scale, 0.047 * scale]

        for body in root.findall(".//worldbody/body/body"):
            if body.get("name") == "object":
                heating_region = {
                    "rgba": "0 0 0 0",
                    "quat": "1 0 0 0",
                    "type": "box",
                    "size": " ".join(map(str, heating_region_size)),
                    "pos": " ".join(map(str, heating_region_pos)),
                    "name": "heating_region",
                }
                site = ET.SubElement(body, "site")
                for key, value in heating_region.items():
                    site.set(key, value)

        # Find the body element within worldbody to add new sites
        for body in root.findall(".//worldbody/body"):
            # Define the new site elements to be added
            new_sites = [
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, bottom_site_pos)),
                    "name": "bottom_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, top_site_pos)),
                    "name": "top_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, horizontal_radius_site_pos)),
                    "name": "horizontal_radius_site",
                },
            ]

            # Add the new sites to the body element
            for site_info in new_sites:
                site = ET.SubElement(body, "site")
                for key, value in site_info.items():
                    site.set(key, value)

        # Create a temporary file to save the modified XML content
        tree.write(tmp_xml, encoding="utf-8", xml_declaration=True)

        # Optional: Clean up the temporary file later if desired
        # os.remove(temp_file_path)
        super().__init__(tmp_xml, name, joints)

        # delete the temporary file
        # os.remove(tmp_xml)

        self.object_properties["articulation"]["default_open_ranges"] = [-1.57, -1.3]
        self.object_properties["articulation"]["default_close_ranges"] = [-0.005, 0.0]

        self.object_state_joints = [super().joints[0]]
        # from IPython import embed; embed()

        os.remove(tmp_xml)

    @property
    def joints(self):
        return [super().joints[1]]  # return free joint

    def is_open(self, qpos):
        if qpos < max(self.object_properties["articulation"]["default_open_ranges"]):
            return True
        else:
            return False

    def is_close(self, qpos):
        if qpos > min(self.object_properties["articulation"]["default_close_ranges"]):
            return True
        else:
            return False


@register_object
class RobocasaSink(RobocasaObject):
    def __init__(
        self,
        name="robocasa_sink",
    ):
        xml_path = os.path.join(
            BASE_FIXTURE_PATH, "sinks/1_bin_storage_right_dark/model.xml"
        )
        tmp_xml = xml_path.replace("model.xml", "tmp.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        scale = 1.0

        # Find the mesh element and modify its scale
        for mesh in root.findall(".//mesh"):
            mesh.set("scale", str(scale) + " " + str(scale) + " " + str(scale))

        # Find all sites and scale their positions by 0.1
        for site in root.findall(".//site"):
            pos = site.get("pos")
            pos = " ".join([str(float(p) * scale) for p in pos.split()])
            site.set("pos", pos)

        # Find all the geoms of class collision and scale their size by 0.1
        for geom in root.findall(".//geom"):
            if geom.get("class") == "collision":
                size = geom.get("size")
                size = " ".join([str(float(s) * scale) for s in size.split()])
                geom.set("size", size)

        # Get position of first 4 sites in .//worldbody/body/
        for body in root.findall(".//worldbody/body"):
            site_positions = []
            for site in body.findall(".//site"):
                site_positions.append(parse_position(site.get("pos")))
            break

        # min and max y values from site positions
        min_y = min([pos[1] for pos in site_positions])
        max_y = max([pos[1] for pos in site_positions])
        min_x = min([pos[0] for pos in site_positions])
        max_x = max([pos[0] for pos in site_positions])
        min_z = min([pos[2] for pos in site_positions])
        max_z = max([pos[2] for pos in site_positions])

        # Calculate the positions for the new sites
        bottom_site_pos = [0, 0, min_z]
        top_site_pos = [0, 0, max_z]
        horizontal_radius_site_pos = [0, 0, 0]
        horizontal_radius_site_size = max(max_x - min_x, max_z - min_z) / 2.0

        # Find the body element within worldbody to add new sites
        for body in root.findall(".//worldbody/body"):
            # Define the new site elements to be added
            new_sites = [
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, bottom_site_pos)),
                    "name": "bottom_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, top_site_pos)),
                    "name": "top_site",
                },
                {
                    "rgba": "0 0 0 0",
                    "size": "0.005",
                    "pos": " ".join(map(str, horizontal_radius_site_pos)),
                    "name": "horizontal_radius_site",
                },
            ]

            # Add the new sites to the body element
            for site_info in new_sites:
                site = ET.SubElement(body, "site")
                for key, value in site_info.items():
                    site.set(key, value)

        # Create a temporary file to save the modified XML content
        tree.write(tmp_xml, encoding="utf-8", xml_declaration=True)

        super().__init__(tmp_xml, name=name, joints=[dict(type="free")])

        # delete the temporary file
        # os.remove(tmp_xml)
