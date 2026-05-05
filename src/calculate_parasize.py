import torch

model_path = 'C:/hienlt/multilingual_vd/src/moe/saved_models/final_best_model.pt'
checkpoint = torch.load(model_path, map_location='cpu')

def calculate_tottal_params():
    total_params = 0
    model_type = model_path.split('/')[-1].split('.')[1]
    if model_type == 'bin':
        for key in checkpoint:
            total_params += checkpoint[key].numel()
    elif model_type == 'pt':
        if isinstance(checkpoint, torch.nn.Module):
            total_params = sum(p.numel() for p in checkpoint.parameters())
        else:
            total_params = sum(tensor.numel() for tensor in checkpoint.values())
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    calculate_tottal_params()


# 250,235,400: Moe

# 124,647,170: codebert baseline c++
# 124,647,170: codebert baseline python
# 124,647,170: codebert baseline java