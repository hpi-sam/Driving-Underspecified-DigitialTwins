import os
from glob import glob

folderpath = os.path.join("hpi", "fs00", "share", "fg-giese", "masterproject_SoSe2023", "RL.4.Autonomous.Vehicles",
                          "data", "data_collect_town01_results")

episode_length = 10
episode_index = []
image_index = []
episode_paths = {}

paths = sorted(list(os.listdir(folderpath)))
for idx, element in enumerate(paths):
    elementpath = os.path.join(folderpath, element)
    elementpath = os.path.join(elementpath, 'rgb')

    # Überprüfe, ob es sich um einen Ordner handelt
    if os.path.isdir(elementpath):
        print(elementpath)

        current_paths = list(glob(os.path.join(elementpath, "*.png")))

        effective_length = max(len(current_paths) - episode_length, 0)

        episode_index = episode_index + [idx for _ in range(effective_length)]

        image_index = image_index + list(range(effective_length))

        episode_paths[idx] = current_paths
