import cv2
import numpy as np
import torch
import torchvision


### image manipulation functions ###
def resize_img(path, new_path, size):
    """
    Resize image with black borders and write to new folder.
    """
    desired_size = size
    img = cv2.imread(path)
    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    img = cv2.resize(img, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    
    cv2.imwrite(new_path, new_img)
    

def center_crop(path, new_path, size):
    """
    Center crop image with square area.
    """
    dim = (size, size)
    img = cv2.imread(path)
    width, height = img.shape[1], img.shape[0]
    
    crop_width = dim[0] if dim[0] < width else height
    crop_height = dim[1] if dim[1] < height else width
    
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    
    cv2.imwrite(new_path, crop_img)
    
    
def load_image(path):
    """
    Perform necessary reading in and transformations for pretrained image classification models.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img / 255.).float()
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    return img
  

### training loops ###
def one_pass_classification(model, dataloader, optimizer, lossFun, backwards=True, print_loss=False):
    """
    One training epoch for classification model.
    """
    if backwards == True:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    avg_accs = []
    for x, y in tqdm(dataloader):
        
        y_pred = model(x)
        loss = lossFun(y_pred, y)
        total_loss += loss.item()
        avg_accs.append(torch.sum(torch.argmax(y_pred, dim=1) == y).item()/len(y))
        
        if backwards == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    avg_loss = total_loss / len(dataloader)
    avg_acc = sum(avg_accs) / len(dataloader)
    
    if print_loss == True:
        print(avg_loss)
        print(avg_acc)
    
    return avg_loss, avg_acc


def one_pass_siamese(model, dataloader, optimizer, lossFun, backwards=True, print_loss=False):
    """
    One training epoch for Siamese network.
    """
    if backwards == True:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    avg_accs = []
    for x1, x2, y in tqdm(dataloader):
        
        y_soft_pred = model(x1, x2)
        loss = lossFun(y_soft_pred, y)
        total_loss += loss.item()
        y_hard_pred = torch.where(y_soft_pred > 0, 1, 0)
        avg_accs.append(torch.sum(y_hard_pred == y).item()/len(y))
        
        if backwards == True:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    avg_loss = total_loss / len(dataloader)
    avg_acc = sum(avg_accs) / len(dataloader)
    
    if print_loss == True:
        print(avg_loss)
        print(avg_acc)
    
    return avg_loss, avg_acc


def validation_check(model, dataloader):
    """
    Perform validation with a pair of images for 
    classification model.
    """
    model.eval ()
    
    trues = []
    preds = []
    for x1, x2, y in tqdm(dataloader):
        y_pred_1 = model(x1)
        y_pred_2 = model(x2)
        y_hard_preds = torch.where(torch.argmax(y_pred_1, dim=1) == torch.argmax(y_pred_2, dim=1), 1, 0)
        preds.extend(y_hard_preds.tolist())
        trues.extend(y.tolist())
        
    trues = np.array(trues)
    preds = np.array(preds)
    
    auc = roc_auc_score(trues, preds)
        
    print(confusion_matrix(trues, preds))
    print(auc)
    
    
def validation_check_siamese(model, dataloader):
    """"
    Perform validation with a pair of images for 
    Siamese network model.
    """
    model.eval()
    
    preds = []
    trues = []
    for x1, x2, y in tqdm(dataloader):
        y_pred = model(x1, x2)
        probs = torch.sigmoid(y_pred)
        preds.extend(probs.tolist())
        trues.extend(y.tolist())
        
    trues = np.array(trues)
    preds = np.array(preds)
    
    auc = roc_auc_score(trues, preds)
        
    print(confusion_matrix(trues, np.where(preds > 0.5, 1, 0)))
    print(auc)