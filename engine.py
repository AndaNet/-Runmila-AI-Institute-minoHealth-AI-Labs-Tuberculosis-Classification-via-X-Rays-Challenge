# engine.py
import torch
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

def train(data_loader, model, optimizer, device):
    model.train()
    # go over every batch of data in data loader
    for data in data_loader:
        inputs = data["ID"]
        targets = data["LABEL"]

        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

        loss.backward()
        optimizer.step()

def evaluate(data_loader, model, device):
        """
        This function does evaluation for one epoch
        :param data_loader: this is the pytorch dataloader
        :param model: pytorch model
        :param device: cuda/cpu
        """

    # put model in evaluation mode
        model.eval()
    # init lists to store targets and outputs
        final_targets = []
        final_outputs = []
    # we use no_grad context
        with torch.no_grad():
            for data in data_loader:
                inputs = data["ID"]
                targets = data["LABEL"]
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.float)
                output = model(inputs)
                targets = targets.detach().cpu().numpy().tolist()
                output = output.detach().cpu().numpy().tolist()
                # extend the original list
                final_targets.extend(targets)
                final_outputs.extend(output)
        return final_outputs, final_targets










