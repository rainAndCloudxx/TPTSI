import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MetaphorDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        label = torch.tensor(row['label'], dtype=torch.long)
        context = row['context']
        expression = row['expression']
        Positive_index = row['Positive_index']
        NegativeSentence = row['NegativeSentence']
        NegativeExpression = row['NegativeExpression']
        AnchorContext = row['AnchorContext']
        Negative_index = row['Negative_index']
        AnchorExpression = '[Mask]'

        return str(context), str(expression), label,Positive_index,str(NegativeSentence), str(NegativeExpression), str(AnchorContext),str(AnchorExpression),Negative_index
    
class Data_loader():
    def __init__(self, args):
        super(Data_loader, self).__init__()
        self.args = args
        self.dataset = getattr(self.args, self.args.Dataset)
        self.BatchSize = self.dataset.BatchSize
        self.data_dir = self.dataset.data_dir
    def get_train_data(self):
        dataPath = self.data_dir+'/train.csv'
        train_dataset = MetaphorDataset(dataPath)
        train_loader = DataLoader(train_dataset, batch_size=self.BatchSize, shuffle=True)
        return train_loader

    def get_val_data(self):
        dataPath = self.data_dir+'/val.csv'
        val_dataset = MetaphorDataset(dataPath)
        val_loader = DataLoader(val_dataset, batch_size=self.BatchSize, shuffle=False)
        return val_loader

    def get_test_data(self):
        dataPath = self.data_dir+'/test.csv'
        test_dataset = MetaphorDataset(dataPath)
        test_loader = DataLoader(test_dataset, batch_size=self.BatchSize, shuffle=False)
        return test_loader
    
if __name__ == '__main__':
    pass