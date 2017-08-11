import scipy.io as sio
import numpy as np

file_directory ='/Users/JH/Documents/GitHub/Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems/data/data_read.mat'
mat_contents = sio.loadmat(file_directory)

selectData = {}

selectData['data_p'] = mat_contents['data_p']
selectData['data_q'] = mat_contents['data_q']
selectData['data_w'] = mat_contents['data_w']
selectData['date_idx'] = mat_contents['week']

print selectData
