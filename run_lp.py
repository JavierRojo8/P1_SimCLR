import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.resnet_linear_probing import ResNetLP
from linear_probing import LinearProbing
from data_aug.clean_dataset import CleanDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR_Linear_Probing')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main():
   # We are going to need, the dataset, the model architecture,  
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    print(f'Using device: {args.device}')
    
    dataset_name = args.dataset_name.lower()
    if dataset_name == 'stl10' or dataset_name == 'cifar10':
        args.num_classes = 10
    else:
        raise ValueError('Dataset not supported: {}'.format(args.dataset_name))
    
    dataset = CleanDataset(root_folder=args.data)
    train_loader, test_loader = dataset.get_loaders(
        name=args.dataset_name,
        batch_size=args.batch_size,
        workers=args.workers
    )   

    checkpoint_path = f'runs/checkpoint_4LP/checkpoint_try_1.pth.tar' 
    
    model = ResNetLP(base_model=args.arch, out_dim=args.out_dim, checkpoint_path=checkpoint_path, freeze_backbone=True)
    
    optimizer = torch.optim.Adam(model.backbone.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
    with torch.cuda.device(args.gpu_index):
        lp_trainer = LinearProbing(model=model,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   args=args)
        lp_trainer.train(train_loader=train_loader)  
    
if __name__ == '__main__':
    main()