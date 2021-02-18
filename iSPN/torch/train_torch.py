import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from rat_torch import RatSpn, SpnArgs
import region_graph

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x.view((28 * 28,)))])

batch_size = 256

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                      transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                     transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)


rg = region_graph.RegionGraph(range(28 * 28))
for _ in range(0, 8):
    rg.random_split(2, 2)

args = SpnArgs()
args.num_sums = 20
args.num_gauss = 10
spn = RatSpn(10, region_graph=rg, name="spn", args=args).cuda()
spn.num_params()

criterion = nn.CrossEntropyLoss()
# print(list(spn.parameters()))
optimizer = optim.Adam(spn.parameters())

for epoch in range(20):
    running_loss = 0.0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = spn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        prediction = torch.argmax(outputs, 1)
        correct += sum(prediction == labels).item()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f acc %.2f' %
                  (epoch + 1, i + 1, running_loss / 200, correct / (200 * batch_size)))
            running_loss = 0.0
            correct = 0
print('done')


