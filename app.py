from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
import psutil
import datetime as dt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import csv  

# Initialize Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

@app.route('/stats')
def stats():
    cpu = psutil.cpu_percent(interval=None)
    memory_stats = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    uptime = dt.datetime.now() - dt.datetime.fromtimestamp(psutil.boot_time())
    battery = psutil.sensors_battery()

    return jsonify({
        'cpu': cpu,
        'memory': memory_stats.percent,
        'gpu': 60,
        'disk': disk_usage.percent,
        'battery': battery.percent if battery else None,
        'power_plugged': battery.power_plugged if battery else None
    })

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Dummy data for training
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)
loader = DataLoader(dataset, batch_size=10)

# Training model and emit data to the frontend
# Training model and emit data to the frontend
def train_model():
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_epochs = 5

    # Prepare CSV file to store results
    csv_filename = 'training_results.csv'
    
    # Open the file in write mode and set up the CSV writer
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Iteration', 'Total Batches', 'Loss', 'Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header

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
                
                # Calculate metrics
                avg_loss = round(running_loss / (i + 1), 4)
                accuracy = round(100 * correct / total, 2)
                socketio.emit('training_data', {
                    'epoch': epoch + 1,
                    'iteration': i + 1,
                    'total_batches': len(loader),
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'batch_loss': round(loss.item(), 4),
                    'batch_accuracy': round(100 * (predicted == labels).sum().item() / labels.size(0), 2)
                })

                # Write to the CSV file
                writer.writerow({
                    'Epoch': epoch + 1,
                    'Iteration': i + 1,
                    'Total Batches': len(loader),
                    'Loss': avg_loss,
                    'Accuracy': accuracy
                })

                time.sleep(1)



@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_training')
def handle_start_training():
    train_model()

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
