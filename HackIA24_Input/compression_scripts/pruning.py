# pruner retrain and log everything in wandb
# Imports
import wandb
import torch.nn.functional as F
import torch
import torch_pruning as tp
import os
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from functools import partial
assert torch.cuda.is_available()


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



### CONFIG ###  

# model
model = torch.load('models/classifier.pt').to(device)

# training parameters
batch_size = 512
epochs = 100
lr = 0.01
random_seed = 42

# pruning parameters 
method = "random" # choices "random", "l1", "lamp", "slim", "group_norm", "group_sl"
speed_up = 2 # speed up ratio based on MACs
compression_ratio = 2 # compression ratio based on model size
global_pruning = False # if True: global pruning else: local
iterative_steps = 400 # pruning steps 
max_sparsity = 1 # max sparsity 

num_classes = 3 # to avoid pruning last layer


# Fixer le seed pour la reproductibilité
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Here you can create your dataloader because after pruning, you should retrain the model: récupérer code depuis code entraînement
# dataloader = ...  

# Evaluation loop
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    verbose=True,
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    loss = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)
        # Calculate loss
        loss += F.cross_entropy(outputs, targets, reduction="sum")
        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)
        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    return (num_correct / num_samples * 100).item(), (loss / num_samples).item()

# training loop
def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    # for pruning
    weight_decay=5e-4,
    pruner=None,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay if pruner is None else 0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,100], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = dict()

  
    for epoch in range(epochs):
        model.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients (from the last iteration)
            optimizer.zero_grad()

            # Forward inference
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward propagation
            loss.backward()

            # Pruner regularize for sparsity learning
            if pruner is not None:
                pruner.regularize(model)

            # Update optimizer
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model, test_loader)
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_acc = acc
        # Update LR scheduler
        scheduler.step()
    model.load_state_dict(best_checkpoint['state_dict'])
    if save:
        # on veut sauvegarder le meilleur modèle
        path = os.path.join(os.getcwd(), "results", save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model.state_dict(), path)
        else:
            torch.save(model, path)     
    print(f'Best val acc: {best_acc:.2f}')



# Pruner
# définir le nbre de classses => évite de pruner la dernière couche
def get_pruner(model, example_input):
    sparsity_learning = False
    if method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=global_pruning)
    elif method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=global_pruning)
    elif method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=global_pruning)
    elif method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=1e-5, global_pruning=global_pruning)
    elif method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=global_pruning)
    elif method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=1e-5, global_pruning=global_pruning)
    else:
        raise NotImplementedError

    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == num_classes:
            ignored_layers.append(m)
    

    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_input,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=1,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=max_sparsity,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner

# pruning jusqu'à atteindre le speed up voulu
def progressive_pruning_speedup(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        # print(current_speed_up)
        if pruner.current_step == pruner.iterative_steps:
            break
    return current_speed_up


# pruning jusqu'à atteindre le ratio de compression voulu
def progressive_pruning_compression_ratio(pruner, model, compression_ratio, example_inputs):
    # compression ratio défini par taille initiale / taille finale
    model.eval()
    _, base_params = tp.utils.count_ops_and_params(
        model, example_inputs=example_inputs)
    current_compression_ratio = 1
    while current_compression_ratio < compression_ratio:
        pruner.step(interactive=False)
        _, pruned_params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs)
        current_compression_ratio = float(base_params) / pruned_params
        if pruner.current_step == pruner.iterative_steps:
            break
        # print(current_compression_ratio)
    return current_compression_ratio



def main():
    # Avant pruning
    example_input = torch.rand(1, 3, 224, 224).to(device)
    start_macs, start_params = tp.utils.count_ops_and_params(model, example_input)
    print('----- Avant pruning -----')
    print(f'Nombre de MACs = {start_macs/1e6:.3f} M')
    print(f'Nombre de paramètres = {start_params/1e6:.3f} M')
    print('')
    
    pruner = get_pruner(model, example_input)
    print(f'Pruning method = {method}')

    # pruning using compression ratio as objective
    # progressive_pruning_compression_ratio(pruner, model, compression_ratio, example_input)

    # pruning using speed up ratio as objective (Number of MACs)
    progressive_pruning_speedup(pruner, model, speed_up, example_input)
    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_input)
    print('----- Après pruning -----')
    print(f'Nombre de MACs = {pruned_macs/1e6:.3f} M')
    print(f'Nombre de paramètres = {pruned_params/1e6:.3f} M')
    print('')

    # Results
    print('----- Results before fine tuning -----')
    print(f'Params: {start_params/1e6:.2f} M => {pruned_params/1e6:.2f} M')
    print(f'MACs: {start_macs/1e6:.2f} M => {pruned_macs/1e6:.2f} M')
    print('')

    
if __name__ == "__main__":
    main()
