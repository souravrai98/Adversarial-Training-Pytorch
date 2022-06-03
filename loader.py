import torch
import torchvision
import torchvision.transforms as transforms

def mnist_loader(batch_size):
    # Preprocess input
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.13, ), (0.3, ))])
    # Load   
    trainset = torchvision.datasets.MNIST(
        root='/home/sourav', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(
        root='/home/sourav', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

def cifar_loader(batch_size):
    # Preprocess input
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4,fill=0, padding_mode='constant'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        
    ])

    transform_test = transforms.Compose(
        [transforms.ToTensor()])
    # Load
    trainset = torchvision.datasets.CIFAR10(root='/home/sourav', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='/home/sourav', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader
