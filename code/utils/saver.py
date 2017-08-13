import scipy.io as sio


def save_mat_data(file_dir, dict_data):
    """
    save given data in mat format
    :param file_dir: save file directory
    :param dict_data: save file data
    :return: n.a.
    """
    sio.savemat(file_name=file_dir, mdict=dict_data)
