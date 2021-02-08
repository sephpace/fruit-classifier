#!/usr/bin/python
"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from data import Fruits360Dataset
from model import FruitClassifier


EPOCHS = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
SHUFFLE = True


def train():
    """
    Train the fruit classifier model.
    """
    # Set up dataset and data loader
    dataset = Fruits360Dataset(load_all=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    
    # Set up neural net and training parameters
    net = FruitClassifier(len(dataset)).train()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    # Train the neural net
    data_len = len(loader)
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        for step, (x, t) in enumerate(loader):
            # Training step
            y = net(x)
            loss = criterion(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display progress
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            status = f'Epoch: {epoch:{len(str(EPOCHS))}}/{EPOCHS}  ' \
                     f'Step: {step + 1:{len(str(data_len))}}/{data_len}  ' \
                     f'Loss: {avg_loss}'
            print(status, end='\r')
        print()

    # Save the trained model state
    torch.save(net.state_dict(), 'states/fruit_classifier.pt')


if __name__ == '__main__':
    train()
