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

# predict type parameter
CONST_preprocess = True
CONST_predict4p = True
CONST_predict4q = True


def data_preprocess():
    read_dir = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_read.mat')
    write_dir = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess.mat')

    org_data = parser.read_raw_data(read_dir)
    sum_data = preprocess.sum_utility(org_data)
    reshape_data = preprocess.reshape(sum_data)
    scale_data = preprocess.scale(reshape_data)

    saver.save_mat_data(write_dir, scale_data)


def data4predict_p():
    read_dir = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess.mat')
    write_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'train_p.mat')
    write_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'test_p.mat')

    data = parser.read_preprocess_data(read_dir)

    train_data, test_data = preprocess.build_predict_p_data(dict_data=data, day_distance=CONST_predict_len)

    saver.save_mat_data(write_dir1, train_data)
    saver.save_mat_data(write_dir2, test_data)


def data4predict_q():
    read_dir1 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data_preprocess.mat')
    read_dir2 = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'predict_p.mat')
    write_dir = os.path.join(os.path.join(os.path.join(repo_dir, repo_name), data_dir), 'data4predict_q.mat')

    data = parser.read_preprocess_data(read_dir1)
    predict_p = parser.read_predict_p_data(read_dir2)

    input_data = preprocess.build_predict_q_data(dict_data=data, predict_p=predict_p, day_distance=CONST_predict_len)

    saver.save_mat_data(write_dir, input_data)


def buildInputData(preprocess, predict4p, predict4q):
    if preprocess is True:
        data_preprocess()
    if predict4p is True:
        data4predict_p()
    if predict4q is True:
        data4predict_q()


if __name__ == "__main__":
    buildInputData(preprocess=True, predict4p=True, predict4q=False)
