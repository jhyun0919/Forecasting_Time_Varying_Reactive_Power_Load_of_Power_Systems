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


def data_preprocess():
    # file directories
    read_dir = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_raw.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess4label.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess4feature.mat')

    # data pre-process for label data
    org_data = parser.read_raw_data(read_dir)
    sum_select_data = preprocess.sum_select_data(org_data)
    reshape_data = preprocess.reshape(sum_select_data)
    saver.save_mat_data(write_dir1, reshape_data)

    # data pre-process for feature data
    scale_data = preprocess.scale(reshape_data)
    saver.save_mat_data(write_dir2, scale_data)


def data4predict_p():
    # file directories
    read_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess4label.mat')
    read_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess4feature.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'train_p.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'val_p.mat')
    write_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'test_p.mat')

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
    read_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess4label.mat')
    read_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess4feature.mat')
    read_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'predict_p.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'train_q.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'val_q.mat')
    write_dir3 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'test_q.mat')

    # read dataset
    data4label = parser.read_preprocess_data(read_dir1)
    data4pfeature = parser.read_preprocess_data(read_dir2)
    predict_p = parser.read_predict_p_data(read_dir3)

    # build dataset
    train_data, val_data, test_data = preprocess.build_predict_q_data(dict_data4label=data4label,
                                                                      dict_data4feature=data4pfeature,
                                                                      predict_p=predict_p,
                                                                      day_distance=CONST_predict_len)

    # save dataset
    saver.save_mat_data(write_dir1, train_data)
    saver.save_mat_data(write_dir2, val_data)
    saver.save_mat_data(write_dir3, test_data)


def buildInputData(preprocess, predict4p, predict4q):
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
    buildInputData(preprocess=True, predict4p=True, predict4q=False)
