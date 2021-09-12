import sys
import torch
from torch.utils.data import DataLoader, dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

help_info: str = """
1. --info, --help    打印此帮助信息
2. --train           训练并保存模型
4. --test            加载模型并测试
5. --all             训练并测试
"""

def download_data() -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    # Downloading training data from open datasets
    training_data: datasets.FashionMNIST = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor(),
    )

    # Downloading test data from open datasets
    test_data: datasets.FashionMNIST = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor(),
    )

    return training_data, test_data

def prepare_dataloader(training_data: datasets.FashionMNIST, test_data: datasets.FashionMNIST) -> tuple[DataLoader, DataLoader]:
    batch_size: int = 64

    # Create data loaders.
    train_dataloader: DataLoader = DataLoader(dataset=training_data, batch_size=batch_size)
    test_dataloader: DataLoader = DataLoader(dataset=test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

def print_dataloader(dataloader: DataLoader) -> None:
    for X, y in dataloader:
        print("Shape of x [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break
    return

def init_model() -> NeuralNetwork:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return NeuralNetwork().to(device=device)

def save_model(model: NeuralNetwork) -> None:
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def load_model() -> NeuralNetwork:
    print("Begin to load PyTorch Model State from model.ph")
    model: NeuralNetwork = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    return model

def train(dataloader: DataLoader, model: NeuralNetwork, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    size: int = len(dataloader.dataset)
    model.train()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
    
        # Compute prediction error
        pred: torch.nn.Sequential = model(X)
        loss: torch.Tensor = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader: DataLoader, model: NeuralNetwork, loss_fn: torch.nn.Module) -> None:
    size: int = len(dataloader.dataset)
    num_batches: int = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred: torch.nn.Sequential = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f} \n")

def custom_test(model: NeuralNetwork, test_data: datasets.FashionMNIST) -> None:
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred: torch.nn.Sequential = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def argv_train(training_data: datasets.FashionMNIST, test_data: datasets.FashionMNIST):
    train_dataloader, test_dataloader = prepare_dataloader(training_data, test_data)
    print_dataloader(train_dataloader)
    print_dataloader(test_dataloader)
    model: NeuralNetwork = init_model()
    print(model)
    loss_fn: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    optimizer: torch.optim.SGD = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs: int = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    save_model(model)


def argv_test(test_data: datasets.FashionMNIST):
    loaded_model: NeuralNetwork = load_model()
    custom_test(model=loaded_model, test_data=test_data)

if __name__ == '__main__':
    training_data, test_data = download_data()
    if len(sys.argv) >= 2:
        if sys.argv[1] == '--info' or sys.argv[1] == '--help':
            print(help_info)
        elif sys.argv[1] == '--train':
            argv_train(training_data, test_data)
        elif sys.argv[1] == '--test':
            argv_test(test_data)
        elif sys.argv[1] == '--all':
            argv_train(training_data, test_data)
            argv_test(test_data)
        else:
            print('不支持的命令行参数：', sys.argv[1])
            print('请使用--help查看可用的参数列表')
    else:
        print(help_info)
    exit
