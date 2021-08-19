import numpy as np
np.set_printoptions(threshold=np.inf)

Model_name = 'level_1012'
outputs_ori = f'SSIM/results/{Model_name}_outputs_ori.npy'
outputs_scal = f'SSIM/results/{Model_name}_outputs_scal.npy'

outputs_ori_array = np.load(outputs_ori)
outputs_scal_array = np.load(outputs_scal)

print(outputs_ori_array.shape)

outputs_ori_array_reshape = np.array_split(outputs_ori_array, outputs_ori_array.shape[0]//3)

print(len(outputs_ori_array_reshape))