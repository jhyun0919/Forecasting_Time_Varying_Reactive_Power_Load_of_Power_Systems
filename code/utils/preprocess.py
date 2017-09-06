import numpy as np

CONST_p = 'data_p'
CONST_q = 'data_q'
CONST_w = 'data_w'
CONST_idx = 'date_idx'

CONST_t = 'data_t'
CONST_ws = 'data_ws'
CONST_h = 'data_h'
CONST_w1 = 'data_w1'
CONST_w2 = 'data_w2'

CONST_pred_p = 'predict_p'

# CONST_row = int(1048 / 2)
CONST_col = 24

CONST_train_ratio_p = 0.3
CONST_val_ratio_p = 0.1
CONST_test_ratio_p = 0.6

CONST_train_ratio_q = 0.5
CONST_val_ratio_q = 0.2
CONST_test_ratio_q = 0.3

CONST_p_ratio = 1 - CONST_test_ratio_p
CONST_q_ratio = CONST_test_ratio_p


def split_data(dict_data):
    dict4p = dict()
    dict4q = dict()
    keyList = dict_data.keys()
    for key in range(len(keyList)):
        data_len = int(len(dict_data[keyList[key]]) * CONST_p_ratio)
        dict4p[keyList[key]] = dict_data[keyList[key]]
        dict4q[keyList[key]] = dict_data[keyList[key]][data_len:, :]
    return dict4p, dict4q


def sum_data(dict_data):
    new_dict = dict()

    # power data
    new_dict[CONST_p] = np.sum(dict_data[CONST_p], axis=1)
    new_dict[CONST_q] = np.sum(dict_data[CONST_q], axis=1)

    # weather data
    new_dict[CONST_t] = dict_data[CONST_w][:, 0]
    # new_dict[CONST_ws] = dict_data[CONST_w][:, 1]
    # new_dict[CONST_h] = dict_data[CONST_w][:, 2]
    # new_dict[CONST_w1] = dict_data[CONST_w][:, 3]
    # new_dict[CONST_w2] = dict_data[CONST_w][:, 4]

    # day-type index data
    new_dict[CONST_idx] = dict_data[CONST_idx]

    return new_dict


def reshape(dict_data):
    keyList = dict_data.keys()
    data_len = int(len(dict_data[keyList[0]]) / 24)
    for key in range(len(keyList)):
        dict_data[keyList[key]] = np.reshape(dict_data[keyList[key]][:data_len * 24], [data_len, CONST_col])
    return dict_data


def scale(dict_data):
    """
    normalize the input attributes into the range [-1, 1]
    :param dict_data: dataset in dictionary dtype
    :return: normalized dataset in dictionary dtype
    """
    keyList = dict_data.keys()
    data_len = len(dict_data[keyList[0]])
    for key in range(len(keyList)):
        if keyList[key] != CONST_idx:
            for row_idx in range(data_len):
                max_value = np.amax(dict_data[keyList[key]][row_idx])
                min_value = np.amin(dict_data[keyList[key]][row_idx])
                for col_idx in range(len(dict_data[keyList[key]][row_idx])):
                    numerator = dict_data[keyList[key]][row_idx][col_idx] - min_value
                    denominator = max_value - min_value
                    dict_data[keyList[key]][row_idx][col_idx] = 2 * (numerator / denominator) - 1
    return dict_data


def train_val_test_divide(array_data, train_ratio, val_ratio):
    data_len = len(array_data)
    train_len = int(data_len * train_ratio)
    val_len = int(data_len * val_ratio)

    return array_data[0:train_len, :], array_data[train_len:train_len + val_len, :], array_data[train_len + val_len:, :]


def build_predict_p_data(dict_data4label, dict_data4feature, day_distance):
    label = []
    feature = []

    keyList = dict_data4label.keys()
    data_len = len(dict_data4label[keyList[0]])

    for row in range(data_len - day_distance):
        temp_label = []
        temp_feature = []

        temp_label.append(dict_data4label[CONST_p][row + day_distance].tolist())
        temp_feature.append(dict_data4feature[CONST_p][row].tolist() +
                            dict_data4feature[CONST_t][row].tolist() +
                            dict_data4feature[CONST_t][row + day_distance].tolist() +
                            [dict_data4feature[CONST_idx][row + day_distance][0]])

        label.append(temp_label)
        feature.append(temp_feature)

    label = np.reshape(np.array(label), [data_len - day_distance, CONST_col * 1])
    feature = np.reshape(np.array(feature), [data_len - day_distance, CONST_col * 3 + 1])

    input_data = np.hstack((label, feature))
    train_data, val_data, test_data = train_val_test_divide(input_data, CONST_train_ratio_p, CONST_val_ratio_p)

    return {'train_p': train_data}, {'val_p': val_data}, {'test_p': test_data}


def build_predict_q_data(dict_data4label, dict_data4feature, dict_predict_p, day_distance):
    label = []
    feature = []

    keyList = dict_data4label.keys()
    data_len = len(dict_data4label[keyList[0]])

    # scale active prediction result to input into feature data
    dict_predict_p = scale(dict_predict_p)

    for row in range(data_len - day_distance):
        temp_label = []
        temp_feature = []

        temp_label.append(dict_data4label[CONST_q][row + day_distance].tolist())
        temp_list = dict_predict_p[CONST_pred_p][row + day_distance].tolist() + \
                    dict_data4feature[CONST_p][row].tolist() + \
                    dict_data4feature[CONST_q][row].tolist() + \
                    dict_data4feature[CONST_t][row].tolist() + \
                    dict_data4feature[CONST_t][row + day_distance].tolist() + \
                    [dict_data4feature[CONST_idx][row + day_distance][0]]
        temp_feature.append(temp_list)

        label.append(temp_label)
        feature.append(temp_feature)

    label = np.reshape(np.array(label), [data_len - day_distance, CONST_col * 1])
    feature = np.reshape(np.array(feature), [data_len - day_distance, CONST_col * 5 + 1])

    input_data = np.hstack((label, feature))
    train_data, val_data, test_data = train_val_test_divide(input_data, CONST_train_ratio_q, CONST_val_ratio_q)

    return {'train_q': train_data}, {'val_q': val_data}, {'test_q': test_data}


if __name__ == "__main__":
    pass
