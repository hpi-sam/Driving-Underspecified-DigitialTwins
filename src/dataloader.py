import glob
import numpy as np
import os
import torch


def main():
    route_folder_path = os.path.join('.', 'data', 'data_collect_town01_results', 'routes_town01_06_01_16_07_30')
    action_folder_path = os.path.join(route_folder_path, 'supervision')
    embeddings_folder_path = os.path.join(route_folder_path, 'embeddings')

    theme_file_names = sorted(list(glob.glob(os.path.join(embeddings_folder_path, 'theme_*'))))
    spatial_file_names = sorted(list(glob.glob(os.path.join(embeddings_folder_path, 'spatial_*'))))
    action_file_names = sorted(list(glob.glob(os.path.join(action_folder_path, '*'))))

    mock_tensor = torch.tensor([0])

    for index, action_file_name in enumerate(action_file_names):

        if index + 1 >= len(spatial_file_names) or index + 1 >= len(theme_file_names):
            break

        theme_file_name = theme_file_names[index]
        spatial_file_name = theme_file_names[index]
        next_theme_file_name = theme_file_names[index + 1]
        next_spatial_file_name = spatial_file_names[index + 1]
        torch.save(mock_tensor, theme_file_name)
        torch.save(mock_tensor, spatial_file_name)
        torch.save(mock_tensor, next_theme_file_name)
        torch.save(mock_tensor, next_spatial_file_name)

        theme_tensor = torch.load(theme_file_name)
        spatial_tensor = torch.load(spatial_file_name)

        next_theme_tensor = torch.load(next_theme_file_name)
        next_spatial_tensor = torch.load(next_spatial_file_name)

        action = np.load(action_file_name, allow_pickle=True).item()['action']

        print(theme_tensor, spatial_tensor, action, next_theme_tensor, next_spatial_tensor)

    pass


if __name__ == '__main__':
    main()
