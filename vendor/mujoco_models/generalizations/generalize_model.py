
import numpy as np
from os import path, mkdir

folder = path.dirname(__file__)
build_folder = path.join(path.dirname(folder), "build_folder")

RGB = {
    "red" : [1, 0, 0],
    "green" : [0, 1, 0],
    "yellow" : [1, 1, 0],
    "black" : [0, 0, 1]
}

def comment_before(is_3d):
    return "" if is_3d else "<!--"
def comment_after(is_3d):
    return "" if is_3d else "-->"

def arm_joints(is_3d, links):
    vals = dict(before_3d=comment_before(is_3d), after_3d=comment_after(is_3d),
                before_4_link=comment_before(links >= 4), after_4_link=comment_after(links >= 4),
                before_5_link=comment_before(links >= 5), after_5_link=comment_after(links >= 5))
    with open(path.join(folder, "generalized_arm.xml")) as f:
        arm = f.read().format(**vals)
    with open(path.join(folder, "generalized_joints.xml")) as f:
        joints = f.read().format(**vals)
    return arm, joints

def block(color, movable, is_solid, location):
    with open(path.join(folder, "generalized_block.xml")) as f:
        return f.read().format(color_name=color,
                               position=" ".join(str(x) for x in location),
                               before_slides=comment_before(movable), after_slides=comment_after(movable),
                               is_solid=int(bool(is_solid)),
                               rgb=" ".join(str(x) for x in RGB[color]))

def values(model_joints, objects):
    model, joints = model_joints
    with open(path.join(folder, "color_blocks_generalized.xml")) as f:
        return f.read().format(model=model, objects="\n".join(objects), joints=joints)

def mujoco_xml(links, is_push, is_3d):
    return "{build_folder}/{links}link_colors_{push_or_reach}{is_3d}.xml".format(
                build_folder=build_folder,
                links=links,
                push_or_reach="push" if is_push else "reach",
                is_3d="_3d" if is_3d else "")

if __name__ == "__main__":
    try:
        mkdir(build_folder)
    except FileExistsError:
        pass
    for is_3d in True, False:
        for links in 3, 4, 5:
            for is_push in True, False:
                xml = values(arm_joints(is_3d=is_3d, links=links), [block(color, movable=is_push, is_solid=is_push, location=[0, 0, 0])
                                for color in ("red", "green", "yellow", "black")])
                location = mujoco_xml(links, is_push, is_3d)
                with open(location, "w") as f:
                    f.write(xml)
