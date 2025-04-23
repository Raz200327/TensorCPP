import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 50)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(50, 3)
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))
    
model = ExampleModel()

#torch.save(model.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    for key, val in model.state_dict().items():
        print(val.shape)