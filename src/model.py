import torch


class Net(torch.nn.Module):
    def __init__(self, num_qubits, output_classes, hidden_layer_size=256):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_qubits, hidden_layer_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_layer_size, output_classes)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x
