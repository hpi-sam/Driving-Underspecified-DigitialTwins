import os
from glob import glob

folderpath = '/hpi/fs00/share/fg-friedrich/carla/main_trajs6_converted2'

narrow_paths = []
narrow_sem_paths = []
wide_paths = []
wide_sem_paths = []

num_trajectories = 10

# Iteriere über alle Elemente im Hauptordner
for idx, element in enumerate(os.listdir(folderpath)):
    elementpath = os.path.join(folderpath, element)
    elementpath = os.path.join(elementpath, 'rgbs')
    # Überprüfe, ob es sich um einen Ordner handelt
    if os.path.isdir(elementpath):
        print(elementpath)

        narrow_paths = narrow_paths + [path for path in glob(elementpath + "/narr_0_*.jpg")]
        narrow_sem_paths = narrow_sem_paths + [path for path in glob(elementpath + "/narr_sem_0_*.png")]
        wide_paths = wide_paths + [path for path in glob(elementpath + "/wide_0_*.jpg")]
        wide_sem_paths = wide_sem_paths + [path for path in glob(elementpath + "/wide_sem_0_*.png")]

    if idx == num_trajectories:
        break

narrow_paths = sorted(narrow_paths)
narrow_sem_paths = sorted(narrow_sem_paths)
wide_paths = sorted(wide_paths)
wide_sem_paths = sorted(wide_sem_paths)
