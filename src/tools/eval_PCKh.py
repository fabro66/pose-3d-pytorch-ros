#from https://github.com/bearpaw/pytorch-pose/blob/master/evaluation/eval_PCKh.py
import sys
from scipy.io import loadmat
from numpy import transpose
import skimage.io as sio
import numpy as np
import os

threshold = 0.5
SC_BIAS = 0.6
pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15]

this_dir = os.path.dirname(__file__)
dict = loadmat('{}/../../data/mpii/detections_our_format.mat'.format(this_dir))
dataset_joints = dict['dataset_joints']
jnt_missing = dict['jnt_missing']
pos_pred_src = dict['pos_pred_src']
pos_gt_src = dict['pos_gt_src']
headboxes_src = dict['headboxes_src']



#predictions
model_name = 'hg'
predfile = sys.argv[1]
preds = loadmat(predfile)['preds']

pos_pred_src = transpose(preds, [1, 2, 0])

head = np.where(dataset_joints == 'head')[1][0]
lsho = np.where(dataset_joints == 'lsho')[1][0]
lelb = np.where(dataset_joints == 'lelb')[1][0]
lwri = np.where(dataset_joints == 'lwri')[1][0]
lhip = np.where(dataset_joints == 'lhip')[1][0]
lkne = np.where(dataset_joints == 'lkne')[1][0]
lank = np.where(dataset_joints == 'lank')[1][0]

rsho = np.where(dataset_joints == 'rsho')[1][0]
relb = np.where(dataset_joints == 'relb')[1][0]
rwri = np.where(dataset_joints == 'rwri')[1][0]
rkne = np.where(dataset_joints == 'rkne')[1][0]
rank = np.where(dataset_joints == 'rank')[1][0]
rhip = np.where(dataset_joints == 'rhip')[1][0]

jnt_visible = 1 - jnt_missing
uv_error = pos_pred_src - pos_gt_src
uv_err = np.linalg.norm(uv_error, axis=1)
headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
headsizes = np.linalg.norm(headsizes, axis=0)
headsizes *= SC_BIAS
scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
scaled_uv_err = np.divide(uv_err, scale)
scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
jnt_count = np.sum(jnt_visible, axis=1)
less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)


# save
rng = np.arange(0, 0.5, 0.01)
pckAll = np.zeros((len(rng), 16))

for r in range(len(rng)):
    threshold = rng[r]
    less_than_threshold = np.multiply(scaled_uv_err < threshold, jnt_visible)
    pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

name = predfile.split(os.sep)[-1]
PCKh = np.ma.array(PCKh, mask=False)
PCKh.mask[6:8] = True
print("Model,  Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean")
print('{:5s}   {:.2f}   {:.2f}     {:.2f}   {:.2f}    {:.2f}    {:.2f}    {:.2f}   {:.2f}'.format(model_name, PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho])\
        , 0.5 * (PCKh[lelb] + PCKh[relb]),0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]), 0.5 * (PCKh[lkne] + PCKh[rkne]) \
        , 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)))
