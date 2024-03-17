import argparse
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("data_path")
parser.add_argument('model_file')
args = parser.parse_args()

CLASSES = ['Centering', 'MakingHole', 'Pressing', 'Raising', 'Smoothing', 'Sponge', 'Tightening']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class HandMotionDataset(Dataset):
    def __init__(self, path=None):
        self.data = []

        files = []

        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.txt'):
                    files.append(os.path.join(dirpath, filename))

        max_x = 0
        max_y = 0
        max_z = 0
        min_x = 0
        min_y = 0
        min_z = 0

        mean_x = 0
        mean_y = 0
        mean_z = 0

        lines = 0

        for file in files:
            frames = []
            with open(file) as f:
                for line in f:
                    data = line.strip().split(";")[:-1]
                    coords = []
                    for _ in range(1, 85, 3):
                        x = float(data[_])
                        y = float(data[_ + 1])
                        z = float(data[_ + 2])

                        lines += 1

                        mean_x += x
                        mean_y += y
                        mean_z += z

                        if x > max_x:
                            max_x = x
                        if y > max_y:
                            max_y = y
                        if z > max_z:
                            max_z = z

                        if x < min_x:
                            min_x = x
                        if y < min_y:
                            min_y = y
                        if z < min_z:
                            min_z = z

                        coords.append([x, y, z])
                    frames.append(coords)


                mean_x /= (lines * 28)
                mean_y /= (lines * 28)
                mean_z /= (lines * 28)

            self.data.append(frames)

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                for k in range(len(self.data[i][j])):
                    self.data[i][j][k][0] -= mean_x
                    self.data[i][j][k][1] -= mean_y
                    self.data[i][j][k][2] -= mean_z

                    L = max((max_x - min_x), (max_y - min_y), (max_z - min_z))
                    self.data[i][j][k][0] /= L
                    self.data[i][j][k][1] /= L
                    self.data[i][j][k][2] /= L
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], device=DEVICE)
        return sample
    

class BidirectionalLSTMActionRecognition(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectionalLSTMActionRecognition, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional LSTM

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional LSTM
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # selecting only the hidden state of the last time step
        return out
    

ds = HandMotionDataset(args.data_path)
dl = DataLoader(ds, batch_size=1, shuffle=False)

input_size = 3 * 14 * 2
hidden_size = 128
num_layers = 1
num_classes = 7

model = BidirectionalLSTMActionRecognition(input_size, hidden_size, num_layers, num_classes).to(DEVICE)
model.load_state_dict(torch.load(args.model_file))
model.eval()

with open('Results.txt', 'w') as f:
    for i, test_data in enumerate(dl):
        test_data = test_data.reshape(1, -1, input_size)
        test_out = model(test_data)
        _, test_pred = torch.max(test_out, 1)
        print(i + 1, CLASSES[test_pred.item()])
        print(i + 1, CLASSES[test_pred.item()], file=f)