import scipy.io as scio
import numpy as np
import os
import random
files=os.listdir("/home/huyt/yzy-vqa-code/SimpleVQA-main/youtube_ugc/youtube_ugc_image")
# print(files)
data = list(range(1098))
random.shuffle(data)
index=[]
video_names = []
score = []
filename_path = 'data/youtube_ugc_data.mat'
dataInfo = scio.loadmat(filename_path)
index_all = dataInfo['index'][0]
for i in range(1142):
    if dataInfo['video_names'][i][0][0][:-4] in files:
        video_names.append(dataInfo['video_names'][i][0][0])
        score.append(dataInfo['scores'][0][i])
video_names = np.array(video_names, dtype=object)
scio.savemat('data/youtube_ugc_data_1098_1.mat', mdict={'index': data,'video_names': video_names, 'scores': score, })