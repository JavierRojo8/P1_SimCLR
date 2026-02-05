# We now want to visualize the latent space of the trained model using PCA or t-SNE.

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models.resnet_linear_probing import ResNetLP
import numpy as np
import seaborn as sns

def visualize_latent_space(model_checkpoint, base_model, out_dim, data_loader, device):
    # Load the trained model
    model = ResNetLP(base_model=base_model, out_dim=out_dim, checkpoint_path=model_checkpoint, freeze_backbone=False)
    model.to(device)
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model.backbone(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=0) 
    features_2d = tsne.fit_transform(all_features)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=all_labels, palette='tab10', legend='full', alpha=0.7)
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Classes')
    plt.show()
    
    # save the plot
    plt.savefig('latent_space_tsne.png') 
    
if __name__ == "__main__":
    # Define parameters
    model_checkpoint_path = f'runs/checkpoint_4LP/checkpoint_try_1.pth.tar'
    base_model = 'resnet18'
    out_dim = 128
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transformations and loading
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./datasets', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Visualize latent space
    visualize_latent_space(model_checkpoint_path, base_model, out_dim, test_loader, device)