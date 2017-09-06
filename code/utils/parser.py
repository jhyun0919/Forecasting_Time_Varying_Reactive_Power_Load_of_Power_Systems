import scipy.io as sio


def read_raw_data(file_dir):
    mat_contents = sio.loadmat(file_name=file_dir)

    select_data = dict()

    select_data['data_p'] = mat_contents['data_p']
    select_data['data_q'] = mat_contents['data_q']
    select_data['data_w'] = mat_contents['data_w']
    select_data['date_idx'] = mat_contents['week']

    return select_data


def read_select_raw_data(file_dir):
    """
    read raw mat format data and return as a dictionary dtype dataset
    :return: dictionary dataset of raw data
    """
    mat_contents = sio.loadmat(file_name=file_dir)

    select_data = dict()

    select_data['data_p'] = mat_contents['data_p']
    select_data['data_q'] = mat_contents['data_q']
    select_data['data_w'] = mat_contents['data_w']
    select_data['date_idx'] = mat_contents['date_idx']

    return select_data


def read_preprocess_data(file_dir):
    """
    read pre-processed data and return as a dictionary dtype dataset
    :param file_dir: pre-processed data directory
    :return: dictionary dataset of pre-processed data
    """
    return sio.loadmat(file_name=file_dir)


def read_predict_p_data(file_dir):
    """
    read predicted active power data and return as a dictionary dtype dataset
    :param file_dir: predicted active data directory
    :return: dictionary dataset of predicted active data
    """
    mat_contents = sio.loadmat(file_name=file_dir)

    select_data = dict()

    select_data['predict_p'] = mat_contents['predict_p']
    return select_data


if __name__ == "__main__":
    pass
