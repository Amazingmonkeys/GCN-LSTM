import matplotlib.pyplot as plt
import stellargraph
from stellargraph.layer import GCN_LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping

from step2_preprocessing import Preprocessing


class GcnLstm:
    gc_layer_sizes = [16, 10]
    gc_activations = ["relu", "relu"]
    lstm_layer_sizes = [200, 200]
    lstm_activations = ["tanh", "tanh"]
    batch_size = 60

    def create_gcn_lstm_object(self, adj):
        return GCN_LSTM(
            seq_len=Preprocessing.seq_len,
            adj=adj,
            gc_layer_sizes=self.gc_layer_sizes,
            gc_activations=self.gc_activations,
            lstm_layer_sizes=self.lstm_layer_sizes,
            lstm_activations=self.lstm_activations,
        )

    def build_model(self, x_input, x_output, trainX, trainY, testX, testY):
        es = EarlyStopping(monitor="val_mse", patience=20)
        model = Model(inputs=x_input, outputs=x_output)
        model.compile(optimizer="adam", loss="mae", metrics=["mse"])
        history = model.fit(
            trainX,
            trainY,
            epochs=500,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(testX, testY), callbacks=[es])
        stellargraph.utils.plot_history(history)
        plt.savefig("test.png")
        y_predict = model.predict(testX)
        return y_predict
