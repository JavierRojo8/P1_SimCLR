# Here we will compare the resultf from both checkpoints
import torch
from models.resnet_simclr import ResNetSimCLR
from models.resnet_linear_probing import ResNetLP
import numpy as np
from sklearn.metrics import confusion_matrix

def load_model(checkpoint_path, model_class, base_model, out_dim):
    model = model_class(base_model=base_model, out_dim=out_dim)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def get_model_outputs(model, data_loader):
    all_outputs = []
    with torch.no_grad():
        for images, _ in data_loader:
            outputs = model(images)
            all_outputs.append(outputs.cpu().numpy())
    return np.concatenate(all_outputs)

def compare_models(simclr_checkpoint, lp_checkpoint, base_model, out_dim, data_loader):
    simclr_model = load_model(simclr_checkpoint, ResNetSimCLR, base_model, out_dim)
    lp_model = load_model(lp_checkpoint, ResNetLP, base_model, out_dim)

    simclr_outputs = get_model_outputs(simclr_model, data_loader)
    lp_outputs = get_model_outputs(lp_model, data_loader)

    difference = np.linalg.norm(simclr_outputs - lp_outputs)
    print(f'Norm of the difference between model outputs: {difference}')

    simclr_preds = np.argmax(simclr_outputs, axis=1)
    lp_preds = np.argmax(lp_outputs, axis=1)

    cm_simclr = confusion_matrix(simclr_preds, simclr_preds)
    cm_lp = confusion_matrix(lp_preds, lp_preds)

    print('Confusion Matrix for SimCLR Model:')
    print(cm_simclr)
    print('Confusion Matrix for Linear Probing Model:')
    print(cm_lp)
    
if __name__ == "__main__":
    # Example usage
    simclr_checkpoint_path = 'path_to_simclr_checkpoint.pth.tar'
    lp_checkpoint_path = 'path_to_lp_checkpoint.pth.tar'
    base_model = 'resnet18'
    out_dim = 128

    # Assuming data_loader is defined and provides the test dataset
    data_loader = None  # Replace with actual DataLoader

    compare_models(simclr_checkpoint_path, lp_checkpoint_path, base_model, out_dim, data_loader)