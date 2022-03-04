import pandas as pd


class ReadData:
    root_path = "dataset/"

    def read_data(self, filename):
        return pd.read_csv(self.root_path + filename, index_col=0)
