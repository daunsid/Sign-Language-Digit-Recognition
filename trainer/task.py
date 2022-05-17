import data
import models
from data import transform

import torch

model, criterion = models.init_model_params()

def train_model(model, optimizer):
    
    since = time.time()
    best_acc = 0.0
    for epoch in range(data.get_epoch()):
        
        print(f'epoch {epoch}/{data.get_epoch()-1}')
        print('-' * 10)
            
        # Each epoch has a training and validation phase
        for phase in ['train_loader', 'valid_loader']:
            if phase == 'train_loader':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in data.loadData()[phase]:
                #inputs = inputs.to(device)
                #labels = labels.to(device)
                
            

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train_loader'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
            
                        # backward
                    if phase == 'train_loader':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            #if phase == 'train_loader':
             #   scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # deep copy the model
            if phase == 'valid_loader' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                

        print()
        
        
    time_eplapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def export_model(dl_model, export_dir, exported_model="model.pth"):
    if not os.path.isdir(export_dir):
        export_path = os.mkdir(export_dir)
    model_path = path.Path(export_dir)/exported_model
    
    timestamp = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    model_path = path.Path(str(model_path)+timestamp)
    torch.save(dl_model.state_dict(), model_path)

def train_export_model():
    model=train_model(model, criterion)
    export_model(model, "outputs")

    
def inference(img, model_path):
    #model = SignNet()
    model.load_state_dict(torch.load(model_path))
    #with model.eval():
    infer_img = transform(img)
    infer_img=infer_img.view(-1, infer_img.shape[0], infer_img.shape[1], infer_img.shape[2])
    outputs = models.softmax(model(infer_img))
    probability, classes = torch.max(outputs, 1)
    return classes.item(), probability.item()
    #return f'Prediction: {classes.item()}; Probability: {probability.item()}'

#if __name__ == __main__:
    