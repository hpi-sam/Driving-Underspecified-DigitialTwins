import json
import os
from glob import glob

import numpy as np
from PIL import Image

DATA_FOLDER_NAME = 'xshkpjvsld'
folder_path = os.path.join('.', 'data')

narrow_paths = []
narrow_sem_paths = []
wide_paths = []
wide_sem_paths = []

num_trajectories = 10

# Iteriere über alle Elemente im Hauptordner
for idx, trajectory in enumerate(os.listdir(folder_path)):
    trajectory_path = os.path.join(folder_path, trajectory)
    element_path = os.path.join(trajectory_path, 'rgbs')

    with open(os.path.join(trajectory_path, 'data.json')) as json_file:
        json_content = json.load(json_file)

    # Überprüfe, ob es sich um einen Ordner handelt
    is_dir = os.path.isdir(element_path)

    if is_dir:
        print(element_path)

        narrow_paths = [path for path in glob(element_path + "/narr_0_*.jpg")]

        potential_image_ids = [os.path.basename(path).split('.')[0].split("_", maxsplit=1)[-1] for path in
                               glob(element_path + "/narr_0_*.jpg")]

        image_files = {}
        for image_id in potential_image_ids:
            potential_images = ("narr_" + image_id + ".jpg",
                                "narr_sem_" + image_id + ".png",
                                "wide_" + image_id + ".jpg",
                                "wide_sem_" + image_id + ".png")

            images_exist = [True if os.path.isfile(os.path.join(element_path, image_versions)) else False for
                            image_versions
                            in potential_images]
            if all(images_exist):
                image_files[int(image_id.split('_')[-1])] = {x.split(image_id)[0] + "rgb": os.path.join(element_path, x)
                                                             for x in potential_images}

# for element in json_content:
#     if element == 'len': continue
#     locs = np.asarray(Image.open(os.path.join(trajectory_path, json_content[element]['wide_rgb_0'])))
#     # wide_sem, narr_rgb, lbls, locs, rots, speds, cmd
#     print(wide_rgb)
#     # , wide_sem, narr_rgb, lbls, locs, rots, spds, int(cmd)

for key, value in image_files.items():
    # @TODO verify if respective images are in other path lists filename = os.path.basename(image)

    narr_rgb = np.asarray(Image.open(value['narr_rgb']))
    # narrow_sem_rgb = np.asarray(Image.open(value['narr_sem_rgb']))
    wide_rgb = np.asarray(Image.open(value['wide_rgb']))
    wide_sem = np.asarray(Image.open(value['wide_sem_rgb']))

    json_element = json_content[str(key)]
    lbl_path = os.path.join(trajectory_path, *json_element['lbl_00'].split('/'))
    lbls = np.asarray(Image.open(lbl_path))
    locs = json_element['loc']
    rots = json_element['rot']
    spds = json_element['spd']
    cmd = json_element['cmd'][0]

    print(wide_rgb, wide_sem, narr_rgb, lbls, locs, rots, spds, int(cmd))
    break

print("Done")
