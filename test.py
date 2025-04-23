import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(512, 50)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(50, 3)
    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))
    
model = ExampleModel()

torch.save(model.state_dict(), 'model_weights.pth')


model_file = ""
os.makedirs("model_info", exist_ok=True)
os.makedirs("model_info/weights", exist_ok=True)


for key, val in model.state_dict().items():
    if len(val.shape) > 1:
        model_file += f"{val.shape[0]}, {val.shape[1]} {key}.csv\n"
    else:
        model_file += f"{val.shape[0]} {key}.csv\n"
    weights = val.cpu().numpy()
    np.savetxt(f"./model_info/weights/{key}.csv", weights, delimiter=",", fmt="%f", header="", comments="")

with open("./model_info/model_file.txt", "w") as file:
    file.write(model_file)


    
