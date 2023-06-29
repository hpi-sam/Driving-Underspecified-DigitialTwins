import time
import torch

def load_state(filename):
    data = torch.load(filename, map_location='cpu')

    #print('spatial_mu', data['spatial_mu'].shape)
    #print('theme_mu', data['theme_mu'].shape)

    spatial_mu = data['spatial_mu'].reshape(data['spatial_mu'].shape[0], -1)

    #print('reshape spatial_mu', spatial_mu.shape)
    theme_mu = data['theme_mu']
    #print(spatial_mu.shape, theme_mu.shape)

    state = torch.cat([spatial_mu, theme_mu], dim=1).reshape(-1)

    return state

starttime = time.time()
for i in range(32*128):
    #data = load_state('/hpi/fs00/share/fg-giese/masterproject_SoSe2023/data/data_collect_town01_results/routes_town01_06_01_16_07_30/rgb/0000.pkl')
    data = torch.load('/hpi/fs00/share/fg-giese/masterproject_SoSe2023/DriveGAN_code/state.st')
t =  time.time() - starttime
print(t)