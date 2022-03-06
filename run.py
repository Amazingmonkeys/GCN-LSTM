from step1_read_data import ReadData
from step2_preprocessing import Preprocessing
from step3_gcn_lstm import GcnLstm
from step4_evaluation import Evaluation
import sys

if __name__ == '__main__':
    adj = ReadData().read_data(sys.argv[1])
    speed = ReadData().read_data(sys.argv[2])

    train_data, test_data = Preprocessing().train_test_split(speed)
    train_scaled, test_scaled = Preprocessing().scale_data(train_data, test_data)
    trainX, trainY, testX, testY = Preprocessing().sequence_data_preparation(train_scaled, test_scaled)

    gcn_lstm = GcnLstm().create_gcn_lstm_object(adj)
    x_input, x_output = gcn_lstm.in_out_tensors()
    y_predict = GcnLstm().build_model(x_input, x_output, trainX, trainY, testX, testY)

    test_rescref, test_rescpred, testnpredc = Evaluation().rescale_values(train_data, trainY, testX, testY, y_predict)
    Evaluation().performance_measure(testX, test_rescref, test_rescpred, testnpredc)
