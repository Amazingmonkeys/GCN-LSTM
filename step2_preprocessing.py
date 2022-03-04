import numpy as np


class Preprocessing:
    train_rate = 0.8
    seq_len = 10
    pre_len = 12

    def train_test_split(self, data, train_portion=train_rate):
        time_len = data.shape[1]
        train_size = int(time_len * train_portion)
        train_data = np.array(data.iloc[:, :train_size])
        test_data = np.array(data.iloc[:, train_size:])
        return train_data, test_data

    def scale_data(self, train_data, test_data):
        max_speed = train_data.max()
        min_speed = train_data.min()
        train_scaled = (train_data - min_speed) / (max_speed - min_speed)
        test_scaled = (test_data - min_speed) / (max_speed - min_speed)
        return train_scaled, test_scaled

    def sequence_data_preparation(self, train_data, test_data, seq_len=seq_len, pre_len=pre_len):
        trainX, trainY, testX, testY = [], [], [], []

        for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
            a = train_data[:, i: i + seq_len + pre_len]
            trainX.append(a[:, :seq_len])
            trainY.append(a[:, -1])

        for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
            b = test_data[:, i: i + seq_len + pre_len]
            testX.append(b[:, :seq_len])
            testY.append(b[:, -1])

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)

        return trainX, trainY, testX, testY
