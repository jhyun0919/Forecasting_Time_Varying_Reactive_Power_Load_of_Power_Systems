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

CONST_row = 1048
CONST_col = 24

CONST_train_ratio = 0.7
CONST_val_ratio = 0.1
CONST_test_ratio = 0.2


def sum_select_data(dict_data):
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
    for key in range(len(keyList)):
        dict_data[keyList[key]] = np.reshape(dict_data[keyList[key]], [CONST_row, CONST_col])
    return dict_data


def scale(dict_data):
    """
    normalize the input attributes into the range [-1, 1]
    :param dict_data: dataset in dictionary dtype
    :return: normalized dataset in dictionary dtype
    """
    keyList = dict_data.keys()
    for key in range(len(keyList)):
        if keyList[key] != CONST_idx:
            for row_idx in range(CONST_row):
                max_value = np.amax(dict_data[keyList[key]][row_idx])
                min_value = np.amin(dict_data[keyList[key]][row_idx])
                for col_idx in range(len(dict_data[keyList[key]][row_idx])):
                    numerator = dict_data[keyList[key]][row_idx][col_idx] - min_value
                    denominator = max_value - min_value
                    dict_data[keyList[key]][row_idx][col_idx] = 2*(numerator/denominator) - 1
    return dict_data


def train_val_test_divide(array_data):
    data_len = len(array_data)
    train_len = int(data_len * CONST_train_ratio)
    val_len = int(data_len * CONST_val_ratio)

    return array_data[0:train_len, :], array_data[train_len:train_len+val_len, :], array_data[train_len+val_len:, :]


def build_predict_p_data(dict_data4label, dict_data4feature, day_distance):
    label = []
    feature = []

    for row in range(CONST_row - day_distance):
        temp_label = []
        temp_feature = []

        temp_label.append(dict_data4label[CONST_p][row + day_distance].tolist())
        temp_feature.append(dict_data4feature[CONST_p][row].tolist() +
                            dict_data4feature[CONST_t][row].tolist() +
                            dict_data4feature[CONST_t][row + day_distance].tolist() +
                            [dict_data4feature[CONST_idx][row + day_distance][0]])

        label.append(temp_label)
        feature.append(temp_feature)

    label = np.reshape(np.array(label), [CONST_row - day_distance, CONST_col * 1])
    feature = np.reshape(np.array(feature), [CONST_row - day_distance, CONST_col * 3 + 1])

    input_data = np.hstack((label, feature))
    train_data, val_data, test_data = train_val_test_divide(input_data)

    return {'train_p': train_data}, {'val_p': val_data}, {'test_p': test_data}


def build_predict_q_data(dict_data4label, dict_data4feature, predict_p, day_distance):
    label = []
    feature = []

    for row in range(CONST_row - day_distance):
        temp_label = []
        temp_feature = []

        temp_label.append(dict_data4label[CONST_q][row + day_distance].tolist())
        temp_feature.append(predict_p[row + day_distance].tolist() +
                            dict_data4feature[CONST_p][row].tolist() +
                            dict_data4feature[CONST_q][row].tolist() +
                            dict_data4feature[CONST_t][row].tolist() +
                            dict_data4feature[CONST_t][row + day_distance].tolist() +
                            [dict_data4feature[CONST_idx][row + day_distance][0]])

        label.append(temp_label)
        feature.append(temp_feature)

    label = np.reshape(np.array(label), [CONST_row - day_distance, CONST_col * 1])
    feature = np.reshape(np.array(feature), [CONST_row - day_distance, CONST_col * 5 + 1])

    input_data = np.hstack((label, feature))
    train_data, val_data, test_data = train_val_test_divide(input_data)

    return {'train_q': train_data}, {'val_q': val_data}, {'test_q': test_data}


if __name__ == "__main__":
    pass
