import torch

label_true = torch.FloatTensor([1.0])
label_false = torch.FloatTensor([0.0])

def generate_random_seed(size):
    random_data = torch.randn(size) # normal
    return random_data