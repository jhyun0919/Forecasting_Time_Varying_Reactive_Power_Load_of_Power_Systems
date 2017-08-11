import scipy.io as sio


def save_mat_data(file_dir, dict_data):
    sio.savemat(file_name=file_dir, mdict=dict_data)
