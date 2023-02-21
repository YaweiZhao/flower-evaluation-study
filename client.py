from collections import OrderedDict
import warnings
import torch
from network.DenseNet import DenseNet
from utils.dataset import load_data_one_client as load_data
import flwr as fl
import config
warnings.filterwarnings("ignore", category=UserWarning)

param = config.get_param()

data = param["data"]
DEVICE = param["bench_param"]["device"]
server_address = param["bench_param"]["server_address"]
num_round = param["bench_param"]["num_rounds"]

epochs = param["training_param"]["epochs"]
lr = param["training_param"]["learning_rate"]
loss_func = param["training_param"]["loss_func"]
optimiz = param["training_param"]["optimizer"]

if optimiz == "sgd":
    optimizer_param = param["training_param"]["optimizer_param"]
    momentum = optimizer_param["momentum"]
    dampening = optimizer_param["dampening"]
    weight_decay = optimizer_param['weight_decay']
    nesterov = optimizer_param['nesterov']
    
elif optimiz == "adam":
    optimizer_param = param["training_param"]["optimizer_param"]
    betas = optimizer_param["betas"]
    eps = optimizer_param["optimizer_param"]
    weight_decay = optimizer_param['weight_decay']
    

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    if loss_func == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss() # how to define a new loss
    else:
        criterion = torch.nn.MSELoss()
    if optimiz == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum, dampening, weight_decay, nesterov) # how to impliment a new optimizer
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr, betas, eps, weight_decay)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    if loss_func == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

# Load model and data (DenseNet, Local Data)
net = DenseNet().to(DEVICE)
trainloader, testloader, num_examples = load_data(data)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs)
        return self.get_parameters(config={}), len(trainloader.dataset), {}
    
    def model_save(self, parameters, config):
        self.set_parameters(parameters)
        print(net.state_dict().keys())
        torch.save(net.state_dict(), "../model_parameter.pkl")

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address=server_address,
    client=FlowerClient(),
)
