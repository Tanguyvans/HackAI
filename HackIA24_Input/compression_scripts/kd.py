import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import torch.nn.functional as F
import random
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
def train_kd(
    model_student: nn.Module,
    model_teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: int,
    weight_decay=5e-4,
    callbacks=None,
    save=None,
    save_only_state_dict=False,
) -> None:

    optimizer = torch.optim.SGD(model_student.parameters(
    ), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1
    best_checkpoint = dict()

  
    for epoch in range(epochs):
        model_student.train()
        model_teacher.train()
        for inputs, targets in tqdm(train_loader, leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset the gradients (from the last iteration)
            optimizer.zero_grad()

            # Forward inference
            out_student = model_student(inputs)
            out_teacher = model_teacher(inputs)


            # kd loss
            kd_T = 4
            predict_student = F.log_softmax(out_student / kd_T, dim=1)
            predict_teacher = F.softmax(out_teacher / kd_T, dim=1)
            alpha = 0.9
            loss = nn.KLDivLoss()(predict_student, predict_teacher) * (alpha * kd_T * kd_T) + criterion(out_student, targets) * (1-alpha)
            
            loss.backward()


            # Update optimizer
            optimizer.step()

            if callbacks is not None:
                for callback in callbacks:
                    callback()

        acc, val_loss = evaluate(model_student, test_loader)
        print(
            f'Epoch {epoch + 1}/{epochs} | Val acc: {acc:.2f} | Val loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
        if best_acc < acc:
            best_checkpoint['state_dict'] = copy.deepcopy(model_student.state_dict())
            best_acc = acc
        # Update LR scheduler
        scheduler.step()
    model_student.load_state_dict(best_checkpoint['state_dict'])
    if save:
        # on veut sauvegarder le meilleur modèle
        path = os.path.join(os.getcwd(), "results", save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if save_only_state_dict:
            torch.save(model_student.state_dict(), path)
        else:
            torch.save(model_student, path)     
    print(f'Best val acc: {best_acc:.2f}')

# Fixer le seed pour la reproductibilité
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# Here you should defined your dataloader
train_dataloader = ... 
test_dataloader = ... 

student_model = ... # torch.load(model_path)
teacher_model = ... # torch.load(model_path)


# train student model
print('train student model')
train_kd(student_model, teacher_model, train_dataloader, test_dataloader, epochs=100, lr=0.001, save="kdtest.pt")
