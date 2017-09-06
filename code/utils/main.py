import parser
import preprocess
import saver
import os

# directory parameters
repo_dir = '/Users/JH/Documents/GitHub'
repo_name = 'Forecasting_Time_Varying_Reactive_Power_Load_of_Power_Systems'
data_dir = 'data'

# prediction length parameter
CONST_predict_len = 1

# version
CONST_predict_ver = 'elm_predict4p.mat'

def data_split_pq():
    # file directories
    read_dir = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_raw.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_raw4p.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_raw4q.mat')

    # split data
    org_data = parser.read_raw_data(read_dir)
    data4p, data4q = preprocess.split_data(org_data)

    # save data
    saver.save_mat_data(write_dir1, data4p)
    saver.save_mat_data(write_dir2, data4q)


def data_preprocess():
    # file directories
    read_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_raw4p.mat')
    read_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_raw4q.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'label_data4p.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'feature_data4p.mat')
    write_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'label_data4q.mat')
    write_dir4 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'feature_data4q.mat')

    dir_dict = dict()
    dir_dict['p'] = [read_dir1, write_dir1, write_dir2]
    dir_dict['q'] = [read_dir2, write_dir3, write_dir4]

    keyList = dir_dict.keys()

    for key in range(len(keyList)):
        # data pre-process for label data
        org_data = parser.read_select_raw_data(dir_dict[keyList[key]][0])
        sum_data = preprocess.sum_data(org_data)
        reshape_data = preprocess.reshape(sum_data)
        saver.save_mat_data(dir_dict[keyList[key]][1], reshape_data)

        # data pre-process for feature data
        scale_data = preprocess.scale(reshape_data)
        saver.save_mat_data(dir_dict[keyList[key]][2], scale_data)


def data4predict_p():
    # file directories
    read_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'label_data4p.mat')
    read_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'feature_data4p.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'train4p.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'val4p.mat')
    write_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'test4p.mat')

    # read dataset
    data4label = parser.read_preprocess_data(read_dir1)
    data4pfeature = parser.read_preprocess_data(read_dir2)

    # build dataset
    train_data, val_data, test_data = preprocess.build_predict_p_data(dict_data4label=data4label,
                                                                      dict_data4feature=data4pfeature,
                                                                      day_distance=CONST_predict_len)

    # save dataset
    saver.save_mat_data(write_dir1, train_data)
    saver.save_mat_data(write_dir2, val_data)
    saver.save_mat_data(write_dir3, test_data)


def data4predict_q():
    # file directories
    read_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'label_data4q.mat')
    read_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'feature_data4q.mat')
    read_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), CONST_predict_ver)
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'train4q.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'val4q.mat')
    write_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'test4q.mat')

    # read dataset
    data4label = parser.read_preprocess_data(read_dir1)
    data4feature = parser.read_preprocess_data(read_dir2)
    predict_p = parser.read_predict_p_data(read_dir3)

    # build dataset
    train_data, val_data, test_data = preprocess.build_predict_q_data(dict_data4label=data4label,
                                                                      dict_data4feature=data4feature,
                                                                      dict_predict_p=predict_p,
                                                                      day_distance=CONST_predict_len)

    # save dataset
    saver.save_mat_data(write_dir1, train_data)
    saver.save_mat_data(write_dir2, val_data)
    saver.save_mat_data(write_dir3, test_data)


def buildInputData(split_pq_data, preprocess, predict4p, predict4q):
    # data split
    if split_pq_data is True:
        data_split_pq()
    # pre-process
    if preprocess is True:
        data_preprocess()
    # build dataset for predicting avtive power
    if predict4p is True:
        data4predict_p()
    # build dataset for predicting reactive power
    if predict4q is True:
        data4predict_q()


if __name__ == "__main__":
    buildInputData(split_pq_data=False, preprocess=False, predict4p=False, predict4q=True)

