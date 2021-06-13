from tqdm.notebook import tqdm
import torch.nn.functional as F

def train(
    model, device, 
    train_loader, 
    optimizer, lr_scheduler, 
    L1=False, l1_lambda = 0.01
):
    model.train()
    pbar = tqdm(train_loader)
    
    train_losses = []
    train_acc = []
    
    correct = 0
    processed = 0
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
            
        # Calculate loss
        loss = F.nll_loss(y_pred, target)
        if L1:
            l1_loss = 0
            for p in model.parameters():
                l1_loss = l1_loss + p.abs().sum()
            loss = loss + l1_lambda * l1_loss
        else:
            loss = loss
        
        train_loss += loss.item()
        train_losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
        )
        train_acc.append(100*correct/processed)
    
    return train_losses, train_acc

