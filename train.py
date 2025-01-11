import tqdm
from optimizer import *
import torch.nn as nn
from utils import *
from torchvision import models
def train(args,num_epochs=10):

    model = models.resnet18(pretrained=True).to(args.device)
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(args.device)
    
    train_dataset ,test_dataset = get_train_data(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    optimizer = get_optimizer(args,model)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epochs in range(num_epochs):
        model.train()
        running_loss=0.0
        for images, labels in tqdm(train_loader):
            images,labels= images.to(args.device),labels.to(args.device)

            output=model(images)
            loss = criterion(output,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        print(f'Epochs: {epochs+1}/{num_epochs} Training loss: {running_loss/len(train_loader)}')
    return model