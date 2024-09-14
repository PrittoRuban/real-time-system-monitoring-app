from flask import Flask, render_template
from flask_socketio import SocketIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

app = Flask(__name__)
socketio = SocketIO(app)

class SimpleModel(nn.Module):
    def __init__(self):  
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)
loader = DataLoader(dataset, batch_size=10)

def train_model():
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_epochs = 5
    for epoch in range(total_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            socketio.emit('training_data', {
                'epoch': epoch + 1,
                'iteration': i + 1,
                'total_batches': len(loader),
                'loss': round(running_loss / (i + 1), 4),
                'accuracy': round(100 * correct / total, 2),
                'batch_loss': round(loss.item(), 4),
                'batch_accuracy': round(100 * (predicted == labels).sum().item() / labels.size(0), 2)
            })
            
            time.sleep(1)  

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def handle_start_training():
    train_model()

if __name__ == '__main__':
    socketio.run(app, debug=True)
