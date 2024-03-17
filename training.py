import argparse
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('n_epochs')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class HandMotionDataset(Dataset):
    def __init__(self, path="Data Split/", train=True, transform=None):
        self.train = train
        self.path = os.path.join(path, ("Train-set" if train else "Test-set"))
        labels = os.listdir(self.path)

        self.labels = []
        self.data = []

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

        for i, label in enumerate(labels):
            for file in os.listdir(os.path.join(self.path, label)):
                frames = []
                with open(os.path.join(self.path, label, file)) as f:
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

                self.labels.append(i)
                self.data.append(frames)

        self.transform = transform

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
        return len(self.labels)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], device=DEVICE)
        if self.transform:
            sample = self.transform(sample)
        return sample, torch.tensor(self.labels[index], device=DEVICE)
    
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
    
training_data = HandMotionDataset(path=args.data_path, train=True)
test_data = HandMotionDataset(path=args.data_path, train=False)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

# Calculate weights for cross-entropy
class_counts = ((13, 4, 8, 8, 7, 6, 4))
num_classes = 7
total_samples = 50

class_weights = []
for count in class_counts:
    weight = 1 / (count / total_samples)
    class_weights.append(weight)
    

input_size = 3 * 14 * 2
hidden_size = 128
num_layers = 1
num_classes = 7

model = BidirectionalLSTMActionRecognition(input_size, hidden_size, num_layers, num_classes).to(DEVICE)

# Xavier init
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=DEVICE))

num_epochs = int(args.n_epochs)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create tqdm progress bar
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    
    for i, (inputs, labels) in enumerate(pbar):
        inputs = inputs.reshape(1, -1, input_size)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track total loss
        total_loss += loss.item()
        
        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update tqdm progress bar description
        pbar.set_postfix({'Tr. Loss': total_loss / (i + 1), 'Tr. Acc.': 100 * correct / total})


    # Test
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for test_batch_data, test_batch_labels in test_dataloader:
            test_batch_data = test_batch_data.reshape(1, -1, input_size)
            test_outputs = model(test_batch_data)
            _, test_predicted = torch.max(test_outputs, 1)
            test_correct += (test_predicted == test_batch_labels).sum().item()
            test_total += test_batch_labels.size(0)
        
        test_accuracy = 100 * test_correct / test_total
        print(f'Test Set Accuracy: {test_accuracy:.2f}%')

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/epoch_{epoch+1}.pt')
    
    # Set model back to training mode
    model.train()

# Print final training statistics
print(f'Training finished. Average Loss: {total_loss / len(train_dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')
