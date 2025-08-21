"""datalaoder for test step"""
import os
import pandas as pd

class TestdatatLoader():
    def __init__(self, hparams, data_indices) -> None:
          self.data_indices = data_indices
          self.man_data_path = hparams.man_data_path
          self.imp_data_path = hparams.imp_data_path
          self.dataset = self.get_data()

    def get_data(self):
        dataset = {}
        data_fold_list = [f"manometry_{v}.pickle" for v in self.data_indices]
        for ind, pfile in enumerate(data_fold_list):
            p =os.path.join(self.man_data_path, pfile)
            df = self.load_pickle(p)
            dataset[self.data_indices[ind]] = df
        return dataset

    def load_pickle(self, file):
            return pd.read_pickle(file)
