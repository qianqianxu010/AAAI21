
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
 data format:
 0001.jpg, 0002.jpg, 1,  0
 0034.jpg, 0023.jpg, -1, 1
 0022.jpg, 0002.jpg, 0,  1
 0001.jpg, 0045.jpg, 1,  2
 if binary == True, label1==-1 will be set 0.
'''
class myDataSet(Dataset):
    def __init__(self, txt, transform = None, target_transform = None, binary = False):
        self.transform = transform
        self.target_transform = target_transform
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.binary = binary
        self.imgs = dict()

    def __getitem__(self, index):
        line = self.lines[index]
        split = line.split(',') #split by comma
        #if not split[0] in self.imgs:
        #    img0 = Image.open(split[0]).convert("RGB")
        #    self.imgs[split[0]] = img0
        #else:
        #    img0 = self.imgs[split[0]]
        #if not split[1] in self.imgs:
        #    img1 = Image.open(split[1]).convert("RGB")
        #    self.imgs[split[1]] = img1
        #else:
        #    img1 = self.imgs[split[1]]
        #
        img0 = Image.open(split[0]).convert("RGB")
        img1 = Image.open(split[1]).convert("RGB")

        label1 = np.float32(split[2])
        label2 = np.float32(split[3])
        if self.binary: #default is False
            if int(label1) == -1:
                label1 = np.float32(0.)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, label1 + 1, label2, split[0].strip(), split[1].strip()

    def __len__(self):
        return self.lens

class singleDataSet(Dataset):
    def __init__(self, txt, transform = None, target_transform = None):
        self.transform = transform
        self.target_transform = target_transform
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        split = line.split('/') #split by comma
        name = split[-1]
        img = Image.open(line)

        if self.transform is not None:
            img = self.transform(img)

        return img, name

    def __len__(self):
        return self.lens
'''
 label is a vector
'''
class MultiLabelDataSet(Dataset):
    def __init__(self, txt, transform = None, target_transform = None, binary=True):
        self.transform = transform
        self.target_transform = target_transform
        self.lines = [line.rstrip() for line in open(txt, 'r')]
        self.lens = len(self.lines)
        self.binary = binary

    def __getitem__(self, index):
        line = self.lines[index]
        split = line.split(',') #split by comma
        img = Image.open(split[0])
        labels = []
        for i in range(1, len(split)):
            val = np.float32(split[i])
            if self.binary == True:
                if split[i] == '-1':
                    val = np.float32(0.)

            labels.append(val)

        if self.transform is not None:
            img = self.transform(img)

        return img, labels

    def __len__(self):
        return self.lens

'''
'''
class BCERankingLoss(nn.Module):
    def __init__(self):
        super(BCERankingLoss, self).__init__()

    def forward(self, score1, score2, label):
        score = score1 - score2
        prob = torch.sigmoid(score)
        w0 = (label == 0).type(score.dtype)*0.5
        w1 = (label == 1).type(score.dtype)
        w = w0 + w1
        neg_prob = torch.sigmoid(-score)

        loss1 = -w*torch.log(prob)
        loss2 = -(1-w)*torch.log(neg_prob)

        loss = loss1 + loss2
        loss = torch.mean(loss)

        return loss

'''
ranking loss
1,-1, 0
'''
class RankingLoss(nn.Module):
    def __init__(self, margin = 0.):
        super(RankingLoss, self).__init__()
        self.margin = margin

    def forward(self, score1, score2, label):
        bz = label.shape[0]
        score = score1 - score2
        w = (label == 0).type(score.dtype)
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        delta = self.margin - label * score
        dis_loss = torch.pow(w*score, 2)
        loss = dis_loss + (1-w)*torch.clamp(delta, min=0.0)
        loss = torch.mean(loss)

        return loss
'''
ranking loss type2
1, -1, 0
Loss(i, j ) = (1-w)*max(0, m1-i+j) + w*max(0, |i-j|-m2)
'''
class RankingLoss2(nn.Module):
    def __init__(self, margin1 = 1., margin2 = 0.25):
        super(RankingLoss2, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, score1, score2, label):
        score = score1-score2
        w = (label==0).type(score.dtype)
        if label.dtype != score.dtype:
            label = label.type(score.dtype)
        delta = self.margin1 - label * score
        diff_loss = (1-w)*torch.clamp(delta, min=0.0)
        delta2 = torch.pow(score,2) - self.margin2
        sim_loss = w * torch.clamp(delta2, min=0.0)
        loss = torch.mean(diff_loss + sim_loss)

        return loss
'''
 embedding loss
'''
class EmbeddingLoss(nn.Module):
    def __init__(self, margin = 0.):
        super(EmbeddingLoss, self).__init__()
        self.margin = margin
    #loss(i, j) = w*max(0, margin-D(i,j)) + (1-w)*D(i,j)
    def forward(self, x1, x2, w):
        bz, dim = x1.shape

        x1 = F.normalize(x1, p = 2, dim = 1)
        x2 = F.normalize(x2, p = 2, dim = 1)
        x = x1 - x2
        distance = torch.sum(x*x, dim = 1)
        dis_loss = (1-w)*distance
        margin_loss = w * torch.clamp(self.margin - distance, min=0.)
        loss = dis_loss + margin_loss

        return torch.sum(loss)/bz




#CDF
def uniform_cdf(t):
    return (t+1.)/2

def bradley_cdf(t):
#    a = torch.exp(t)
#    return a/(1.+a)
    return torch.sigmoid(t)

class MarginMLELoss(nn.Module):
    def __init__(self, cdf = 'bradley', device = torch.device('cuda:0')):
        super(MarginMLELoss, self).__init__()
        if cdf == 'bradley':
#            self.cdf = bradley_cdf
            self.cdf = torch.sigmoid
        elif cdf == 'uniform':
            self.cdf = uniform_cdf
        else:
            raise Exception('Unknown c.d.f type!')
        self.device = device

    def forward(self, si, sj, lamb, label):
        tmp_a = self.cdf(lamb - si + sj )
        tmp_b = self.cdf(-lamb - si + sj)
        tmp_neg_a = self.cdf(-lamb + si - sj) #for sigmoid, 1-sigmoid(x) = sigmoid(-x)
        #MLE loss
        '''
        loss = -torch.sum( (label == 1).type(si.dtype) * torch.log(1-tmp_a) +
                (label==0).type(si.dtype) * torch.log(tmp_a - tmp_b) +
                (label==-1).type(si.dtype) * torch.log(tmp_b) )
        '''
        #only for sigmoid
        eps = torch.tensor([1e-8], dtype = si.dtype, device = self.device)

        loss = -torch.mean( (label == 1).type(si.dtype) * torch.log(tmp_neg_a) +
                (label==0).type(si.dtype) * torch.log(tmp_a - tmp_b + eps) +
                (label==-1).type(si.dtype) * torch.log(tmp_b) )

        return loss


#test
if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    x = torch.randn(3)
    y = torch.FloatTensor([1,-1,0.])
    loss_fn = BCERankingLoss()
    loss = loss_fn(x,y)
    print('test BCERankingLoss-- x:', x, 'y:', y, 'loss:', loss)

    #test MarginMLELoss
    si = torch.randn(3,1)
    sj = torch.randn(3,1)
    lamb = torch.FloatTensor([1.0,1,1]).view_as(si)
    y = torch.FloatTensor([1., -1, 0])

    loss_fn = MarginMLELoss()
    loss = loss_fn(si, sj, lamb, y)
    print('Test--MLELoss\n si:', si, ' sj:', sj, ' lamb:', lamb,
            ' y:', y, ' loss:', loss)

    #test RankingLoss
    x1 = torch.randn(5)
    x2 = torch.randn(5)
    y = torch.LongTensor([1,1,-1,0,-1])
    loss_fn = RankingLoss(margin = 1.)
    loss = loss_fn(x1.cuda(), x2.cuda(), y.cuda())
    print('Test RankingLoss:\n s:', x1-x2, '|y:', y,
            '|loss:', loss)

    #test embedding loss
    x1 = torch.randn(3, 128)
    x2 = torch.randn(3, 128)
    w = torch.FloatTensor([0.8, 0, 1])
    loss_fn2 = EmbeddingLoss(margin = 2.)
    loss = loss_fn2(x1.cuda(), x2.cuda(), w.cuda())



    #test myDataset
    test_file = 'train_LFW10.txt'
    train_data = myDataSet(txt=test_file,
            transform=transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor()
                ]
                ))

    train_loader = DataLoader(dataset = train_data, batch_size = 8,
            shuffle = True, num_workers=8)


    rgb_mean = torch.zeros(3)
    rgb_std = torch.zeros(3)
    cnt = 0
    for step, (img1, img2, label1, label2) in enumerate(train_loader):
        '''
        print('step:', step, '|img1:', img1.shape, '|img2.shape:', img2.shape,
                '|label:', label1, label2)
        plt.subplot(211)
        grid = utils.make_grid(img1)
        plt.imshow(grid.numpy().transpose((1,2,0)))
        plt.subplot(212)
        grid = utils.make_grid(img2)
        plt.imshow(grid.numpy().transpose((1,2,0)))

        plt.show()

        if step > 5:
            break
        '''
        for i in range(3):
            rgb_mean[i] += torch.mean(img1[:,i,:,:]) + torch.mean(img2[:,i,:,:])
            rgb_std[i] += torch.std(img1[:,i,:,:]) + torch.std(img2[:,i,:,:])
        cnt += 1

    rgb_mean /= cnt*2
    rgb_std /= cnt*2
    print('rgb_mean:', rgb_mean)
    print('rgb_std:', rgb_std)
