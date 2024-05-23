import torch
import torch.nn as nn

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, input):
        # input.shape == (bs,3,n)

        bs = input.size(0)
        xb = torch.relu(self.conv1(input))
        xb = torch.relu(self.conv2(xb))
        xb = torch.relu(self.conv3(xb))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = torch.relu(self.fc1(flat))
        xb = torch.relu(self.fc2(xb))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class Pointfeat(nn.Module):
    def __init__(self):
        super(Pointfeat, self).__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

    def forward(self, input):
        assert input.size()[1] == 3
        n_pts = input.size()[2]
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        xb = torch.relu(self.conv1(xb))


        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)
        pointfeat = xb

        xb = torch.relu(self.conv2(xb))
        xb = self.conv3(xb)
        xb = torch.max(xb, 2, keepdim=True)[0]
        xb = xb.view(-1, 1024)
        xb = xb.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([xb, pointfeat], 1), matrix3x3, matrix64x64


class Surfacefeat(nn.Module):
    def __init__(self):
        super(Surfacefeat, self).__init__()
        self.input_transform = Tnet(k=3)
        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)


    def forward(self, input):
        assert input.size()[1] == 4
        n_pts = input.size()[2]
        input1 = input[:, torch.arange(input.size(1)) != 3, :]
        input2 = input[:, torch.arange(input.size(1)) == 3, :]

        matrix4x4 = self.input_transform(input1)
        # batch matrix multiplication
        xp = torch.bmm(torch.transpose(input1, 1, 2), matrix4x4).transpose(1, 2)
        xb = torch.cat((xp,input2),1)
        assert xb.size()[1] == 4
        xb = torch.relu(self.conv1(xb))
        xb = torch.relu(self.conv2(xb))
        xb = self.conv3(xb)
        xb = torch.max(xb, 2, keepdim=True)[0]
        xb = xb.view(-1, 1024)
        return xb, matrix4x4

class PINN(nn.Module):
    def __init__(self, k=1):
        super(PINN, self).__init__()
        self.k = k
        self.feat1 = Pointfeat()
        self.feat2 = Surfacefeat()
        self.conv1 = torch.nn.Conv1d(2112, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)

    def forward(self, x, y):
        assert x.size()[1] == 3
        assert y.size()[1] == 4
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, matrix3x3, matrix64x64 = self.feat1(x)
        y, matrix4x4 = self.feat2(y)
        y = y.view(-1, 1024, 1).repeat(1, 1, n_pts)
        z = torch.cat([y, x], 1)
        z = torch.relu(self.conv1(z))
        z = torch.relu(self.conv2(z))
        z = torch.relu(self.conv3(z))
        z = self.conv4(z)
        z = z.transpose(2, 1).contiguous()
        z = z.view(batchsize, n_pts, self.k)
        return z, matrix3x3, matrix4x4, matrix64x64