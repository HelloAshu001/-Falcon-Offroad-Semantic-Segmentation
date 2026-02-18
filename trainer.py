import torch
from tqdm import tqdm
from utils.metrics import calculate_iou

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        iou = calculate_iou(outputs, masks)
        total_loss += loss.item()
        total_iou += iou
        
        pbar.set_postfix(loss=loss.item(), IoU=iou)
        
    return total_loss / len(loader), total_iou / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            iou = calculate_iou(outputs, masks)
            
            total_loss += loss.item()
            total_iou += iou
            
    return total_loss / len(loader), total_iou / len(loader)
