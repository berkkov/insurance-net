import torch
import torch.nn as nn
import torch.utils.data
from utils import InsuranceDatasetLoader
from tensorboardX import SummaryWriter

writer = SummaryWriter()


class InsuranceNet(nn.Module):
    def __init__(self, in_features=125):
        super(InsuranceNet, self).__init__()

        layers = []
        #depths = [in_features, 250, 375, 500, 750, 1500, 2250, 2250, 3750, 4500, 5225, 6000, 4000, 2400, 1500, 500, 100, 50, 8]
        # depths = [in_features, 250, 375, 500, 750, 1500, 1500, 2250, 3000, 3750, 3750, 4500, 6000,
        #          4000, 3000, 2400, 1500, 1200, 900, 750, 500, 250, 100, 50, 8]
        depths = [in_features, 250, 375, 500, 750, 1500, 1500, 2250, 2250, 3750, 3750, 3000, 2400, 1500, 1200, 900,
                  750, 500, 250, 100, 50, 8]


        print("Number of layers: ", len(depths)-1)
        for i in range(len(depths)-1):
            layers.append(nn.Linear(depths[i], depths[i+1]))
            layers.append(nn.BatchNorm1d(depths[i+1]))
            layers.append(nn.ReLU())
            if 7 <= i <= 20:
                layers.append(nn.Dropout(p=0.5))

        self.seq = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.seq(x)
        x = self.softmax(x)
        return x


def train(net, optimizer, loss_fn,epoch_no):
    net.train()
    loader = torch.utils.data.DataLoader(InsuranceDatasetLoader('train.csv'), 2048, shuffle=True)
    running_loss = 0.0
    correct_total = 0
    size = 53000
    for i, data in enumerate(loader):
        x, y = data
        optimizer.zero_grad()
        guess = net(x)

        if i % 10 == 0 and i != 0:
            print('Progress for this epoch: %', i/len(loader) * 100)
        y = y.long()
        loss = loss_fn(guess, y)
        loss.backward()
        optimizer.step()
        sum_loss = loss.sum().item()
        running_loss += sum_loss

        correct_total += get_total_correct(guess, y)


    acc = (correct_total/size) * 100
    loss_mean = running_loss/size
    writer.add_scalar('Training Loss v Epoch', loss_mean, epoch_no)
    writer.add_scalar('Training Accuracy v Epoch', acc, epoch_no)


def test(net, epoch_no):
    loader = torch.utils.data.DataLoader(InsuranceDatasetLoader('test.csv'), 2048, shuffle=True)
    net.eval()
    loss_func = nn.CrossEntropyLoss()
    total_size = 6381
    corrects_total = 0
    for i, data in enumerate(loader):
        x, y = data
        guess = net(x)
        y = y.long()
        loss = loss_func(guess, y)

        corrects_total += get_total_correct(guess, y)


    overall_score = (corrects_total/total_size) * 100
    writer.add_scalar('Test Acc v Epoch', overall_score, epoch_no)
    return overall_score


def get_total_correct(guess, y):
    """
    Calculates the number of correct guesses
    :param guess: y head of size [N, C] C: classes N: batch size
    :param y: y of size [N] each y[n€N] = c€C
    :return:
    """

    assert y.size()[0] == guess.size()[0]
    y = y.float()
    guess_pholder = torch.zeros([guess.size()[0]])
    for j in range(guess.size()[0]):
        class_num = torch.argmax(guess[j])
        guess_pholder[j] = class_num

    comp = torch.eq(guess_pholder.cpu(), y.cpu())
    corrects = comp.sum()

    return corrects.item()


if __name__ == '__main__':
    net = InsuranceNet()
    net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 20000):
        print('Epoch No: ', epoch)
        train(net, optimizer, loss_fn, epoch)
        with torch.no_grad():
            test(net, epoch)

        if epoch % 10 == 0 and epoch > 0:
            torch.save(net.state_dict(), 'checkpoint_' + str(epoch) + '.pth.tar')
            torch.save(optimizer.state_dict(), 'optimizer_' + str(epoch) + '.pth.tar')
