import numpy as np
from matplotlib import pyplot as plt


class Evaluation:

    def rescale_values(self, train_data, trainY, testX, testY, y_predict):
        max_speed = train_data.max()
        min_speed = train_data.min()
        train_rescref = np.array(trainY * max_speed)
        test_rescref = np.array(testY * max_speed)
        test_rescpred = np.array(y_predict * max_speed)

        testnpred = np.array(testX)[:, :, -1]
        testnpredc = testnpred * max_speed
        return test_rescref, test_rescpred, testnpredc

    def performance_measure(self, testX, test_rescref, test_rescpred, testnpredc):
        seg_mael = []
        seg_masel = []
        seg_nmael = []

        for j in range(testX.shape[-1]):
            seg_mael.append(
                np.mean(np.abs(test_rescref.T[j] - test_rescpred.T[j]))
            )  # Mean Absolute Error for NN
            seg_nmael.append(
                np.mean(np.abs(test_rescref.T[j] - testnpredc.T[j]))
            )  # Mean Absolute Error for naive prediction
            if seg_nmael[-1] != 0:
                seg_masel.append(
                    seg_mael[-1] / seg_nmael[-1]
                )  # Ratio of the two: Mean Absolute Scaled Error
            else:
                seg_masel.append(np.NaN)

        print("Total (ave) MAE for NN: " + str(np.mean(np.array(seg_mael))))
        print("Total (ave) MAE for naive prediction: " + str(np.mean(np.array(seg_nmael))))
        print(
            "Total (ave) MASE for per-segment NN/naive MAE: "
            + str(np.nanmean(np.array(seg_masel)))
        )
        print(
            "...note that MASE<1 (for a given segment) means that the NN prediction is better than the naive prediction."
        )

        plt.figure(figsize=(15, 8))
        a_pred = test_rescpred[:, 100]
        a_true = test_rescref[:, 100]
        plt.plot(a_pred, "r-", label="prediction")
        plt.plot(a_true, "b-", label="true")
        plt.xlabel("time")
        plt.ylabel("speed")
        plt.legend(loc="best", fontsize=10)
        plt.savefig("plot.png")
