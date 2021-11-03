import torch
from numpy import np
from tqdm import tqdm


def train_wit_network(
    model, device, dataloader, optimizer, criterion, epochs, save=True, epoch_start=0
):
    target = torch.arange(dataloader.batch_size).to(device)
    for epoch in tqdm(np.arange(epoch_start, epoch_start + epochs, 1)):
        for _, batch in enumerate(dataloader):
            optimizer.zero_grad()
            sim = model.forward(batch)
            loss = criterion(sim, target)
            loss.backward()
            optimizer.step()
        print("loss", loss)
        if save:
            torch.save(model.state_dict(), f"./trained_model_{epoch}.pth")
    return model
