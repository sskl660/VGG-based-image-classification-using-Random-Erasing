# 92퍼센트 정확도.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset  # 데이터를 묶어오는 라이브러리.
import torchvision.transforms as transforms  # 이미지를 변환하는 라이브러리.
from torch.utils.data import DataLoader  # 모델에 데이터를 전달.


def con_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2)
    )
    return model


def con_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2)
    )
    return model


def con_3_block_padding2(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=4, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, 2)
    )
    return model


def con_3_block_max(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=4, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model


class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            con_2_block(1, base_dim),
            con_2_block(base_dim, 2*base_dim),
            con_3_block_padding2(2*base_dim, 4*base_dim),
            con_3_block(4*base_dim, 8*base_dim),
            con_3_block_max(8*base_dim, 8*base_dim),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 20),
            nn.ReLU(True),
            nn.Linear(20, num_classes)
        )

    def forward(self, x):
        out = self.feature(x)
        # 텐서의 shape 을 변형해주는 함수. -1은 해당 부분은 알아서 설정하라는 의미이다.
        # 예를 들어, [10, 10]의 텐서를 tensor.view(50, -1)로 변형하면 [50, 2]로 설정 된다.
        # 즉, 여기에서는 batch_size 이외에는 모든 부분을 알아서 결정하라는 것을 의미한다.
        # 왜냐하면, linear 연산 적용하기 위해서 변형을 가하는 것이다.
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out


if __name__ == '__main__':
    batch_size = 256  # 배치사이즈 설정.
    learning_rate = 0.0001
    num_epoch = 300

    # MNIST 데이터 셋 이용. 첫 번째 인수는 경로를 의미하고, 현재 코드가 있는 위치를 이용하였다.
    # train 인수는 True 일시 학습 데이터를, False 일시 테스트 데이터를 불러온다.
    # transform 인수는 이미지 데이터를 파이토치 텐서로 변환하는 ToTensor 함수를 이용하였다.
    # target_transform 인수는 이미지 라벨에 대한 변형을 의미하는데 여기서는 None 을 이용하였다.
    # 마지막 download 는 현재 경로가 데이터가 없으면 다운로드 하겠다는 의미이다.
    mnist_train = dset.FashionMNIST("./", train=True, transform=transforms.ToTensor(),
                             target_transform=None, download=True)
    mnist_test = dset.FashionMNIST("./", train=False, transform=transforms.ToTensor(),
                            target_transform=None, download=True)

    # 첫 번째 인수는 해당 데이터, batch_size 는 batch 의 개수를 의미한다.
    # shuffle 은 셔플 여부를 의미하며, num_workers 는 데이터를 묶을 때 사용할 프로세스 개수를 의미한다.
    # drop_last 는 묶고 남는 데이터는 버릴지에 대한 여부를 의미한다.
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
                                               shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                              shuffle=False, num_workers=2, drop_last=True)

    # MNIST 데이터 셋의 형태는 [batch_size, channels, 가로, 세로]의 형태이다.

    # 모델을 초기화하고 device 를 설정 해준다.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VGG(32).to(device)
    # 손실 함수는 교차 엔트로피 오차를 이용한다. 분류 문제에서는 보통 자극을 더 강조하기 위하여 이를 이용한다.
    loss_func = nn.CrossEntropyLoss()
    # 최적화 함수는 Adam 알고리즘을 이용한다.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 진행.
    loss_arr = []
    for i in range(num_epoch):
        for j, [image, label] in enumerate(train_loader):
            x = image.to(device)
            y_ = label.to(device)

            optimizer.zero_grad()
            output = model.forward(x)
            loss = loss_func(output, y_)
            loss.backward()
            optimizer.step()

            if j % 1000 == 0:
                print(loss)
                loss_arr.append(loss.cpu().detach().numpy())
        if i % 10 == 9:
            print("<", (i + 1), "/", num_epoch, " epoch done>", sep="")

    # 모델 저장
    torch.save(model.state_dict(), )


    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in test_loader:
            x = image.to(device)
            y_ = label.to(device)

            output = model.forward(x)
            _, output_index = torch.max(output, 1)

            total += label.size(0)
            correct += (output_index == y_).sum().float()

        print("Accuracy of Test Data: {}".format(100*correct/total))

