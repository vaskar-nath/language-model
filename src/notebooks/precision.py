import torch
import torch.nn as nn

s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(f"{s=} {s.dtype=}")
s = torch.tensor(0,dtype=torch.float16) 
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(f"{s=} {s.dtype=}")
s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(f"{s=} {s.dtype=}")
s = torch.tensor(0,dtype=torch.float32) 
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(f"{s=} {s.dtype=}")


class ToyModel(nn.Module):
    
    def __init__(self, in_features: int, out_features: int): 
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False) 
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False) 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        print(f"input: {x=} {x.dtype=}")
        x = self.relu(self.fc1(x)) 
        print(f"relu: {x=} {x.dtype=}")
        x = self.ln(x)
        print(f"ln: {x=} {x.dtype=}")
        x = self.fc2(x)
        print(f"fc2: {x=} {x.dtype=}")
        return x

with torch.autocast(device_type="cuda",dtype=torch.float16): 
    model = ToyModel(10, 10)
    model.to("cuda")
    model.eval()
    x = torch.randn(1, 10).to("cuda")
    y = model(x)
    print(f"{y=} {y.dtype=}")