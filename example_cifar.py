import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from timm.loss import LabelSmoothingCrossEntropy
from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d
from asam import ASAM, SAM

def load_cifar(data_loader, batch_size=256, num_workers=2):
    if data_loader == CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # Transforms
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(mean, std)
                         ])
 
    # DataLoader
    train_set = data_loader(root='./data', train=True, download=True, transform=train_transform)
    test_set = data_loader(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers)
    return train_loader, test_loader
   

def train(args): 
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.batch_size)
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    # Model
    model = eval(args.model)(num_classes=num_classes).cuda()

    # Minimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay)
    minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta)
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epochs)

    # Loss Functions
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.
    for epoch in range(args.epochs):
        # Train
        model.train()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            # Ascent Step
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(model(inputs), targets).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                loss += batch_loss.sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        loss /= cnt
        accuracy *= 100. / cnt
        print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
        scheduler.step()

        # Test
        model.eval()
        loss = 0.
        accuracy = 0.
        cnt = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                predictions = model(inputs)
                loss += criterion(predictions, targets).sum().item()
                accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                cnt += len(targets)
            loss /= cnt
            accuracy *= 100. / cnt
        if best_accuracy < accuracy:
           best_accuracy = accuracy
        print(f"Epoch: {epoch}, Test accuracy:  {accuracy:6.2f} %, Test loss:  {loss:8.5f}")
    print(f"Best test accuracy: {best_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='CIFAR10', type=str, help="CIFAR10 or CIFAR100.")
    parser.add_argument("--model", default='wrn28_10', type=str, help="Name of model architecure")
    parser.add_argument("--minimizer", default='ASAM', type=str, help="ASAM or SAM.")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    args = parser.parse_args()
    assert args.dataset in ['CIFAR10', 'CIFAR100'], \
            f"Invalid data type. Please select CIFAR10 or CIFAR100"
    assert args.minimizer in ['ASAM', 'SAM'], \
            f"Invalid minimizer type. Please select ASAM or SAM"
    train(args)
