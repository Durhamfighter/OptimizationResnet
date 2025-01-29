import time
import torch 
from tqdm import tqdm

def test(args,model,test_loader,howmany):
    model.eval()
    s=time.time()
    with torch.no_grad():
        total=0
        correct=0
        for images,labels in tqdm(test_loader):
            images,labels= images.to(args.device),labels.to(args.device)
            output=model(images)
            _,predicted = torch.max(output,dim=1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
        
        accuracy=100*correct/total
        e=time.time()
        print(f'Accuracy: {accuracy}%, Forward Time: {e - s:.2f}s, pruned_channel: {howmany}')
        get_model_memory_usage(model)

def get_model_memory_usage(model):
    total_params = 0
    total_memory = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            total_memory += param.numel() * param.element_size()  # Bytes

    print(f"Total Parameters: {total_params}")
    print(f"Memory Usage for Parameters: {total_memory / 1e6:.2f} MB")  # Convert to MB