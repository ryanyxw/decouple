from torch.utils.data import Dataset
import pandas as pd
import torch

class PandasDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return torch.tensor(self.data.iloc[idx]["hidden_state"]), torch.tensor(self.data.iloc[idx]["label"])

