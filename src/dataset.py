import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(train: bool = True, dataset_name: str = "cifar10") -> transforms.Compose:
    
    base_size = 224
    
   
    is_fashion = dataset_name.lower() == "fashion-mnist"
    
    transform_list = [
        transforms.Resize((base_size, base_size))
    ]
    
    if is_fashion:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    if train:
        
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    else:
        transform_list.append(transforms.ToTensor())

    
    if is_fashion:
        transform_list.append(transforms.Normalize(mean=[0.2860]*3, std=[0.3530]*3))
    else:
        transform_list.append(transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                   std=[0.2470, 0.2435, 0.2616]))

    return transforms.Compose(transform_list)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset_name: str = "cifar10"
) -> tuple[DataLoader, DataLoader]:
    
    
    if dataset_name.lower() == "fashion-mnist":
        dataset_class = datasets.FashionMNIST
    else:
        dataset_class = datasets.CIFAR10

    train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True, dataset_name=dataset_name),
    )
    
    val_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False, dataset_name=dataset_name),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader