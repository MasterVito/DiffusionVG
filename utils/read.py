import os
import numpy
import h5py
import pdb
import torch
import torch.nn as nn

# rootDir = '/home/liangxiao/Datasets/tsgvFeatures/'
# testingFile = [os.path.join(rootDir,x) for x in os.listdir(rootDir) if 'activitynet' in x and 'text' not in x]


# for file in testingFile:
#     cur = h5py.File(file)
#     try:
#         print(file.split('/')[-1], cur[list(cur.keys())[0]].shape)
#     except:
#         print(cur[list(cur.keys())[0]])
        
# file = rootDir = '/home/liangxiao/Datasets/tsgvFeatures/activitynet_c3d_features.hdf5'
# cur = h5py.File(file)
# key_0 = list(cur.keys())[0]
# pdb.set_trace()
    
'''
('charades_clip_256_features.hdf5', (256, 512))
('charades_i3d_features.hdf5', (91, 1024))
('charades_c3d.hdf5', (30, 4096))
('charades_slowfast_features.hdf5', (56, 2304))
('charades_i3d_finetuned.hdf5', (47, 1024))
'''

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# video_encoder = nn.GRU(1024, 128, num_layers=1, bidirectional=True, batch_first=True).to(device)
# video = torch.randn(size=(32,64,1024)).to(device)
# res = video_encoder(video)
# print(res[0].shape)