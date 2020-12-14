
import torch
import os
from utils import myDataSet
from utils import RankingLoss
from utils import BCERankingLoss
from utils import RankingLoss2
from backbone.Resnet import resnet50
from backbone.alexnet import alexnet
from backbone.alexnet_gap import alexnet_gap
from backbone.alexnet_gap2 import alexnet_gap2
from backbone.vgg import vgg16_bn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from torch.nn import CrossEntropyLoss

import numpy as np
import logging as logger

#================= pretrained base net =====================
if len(sys.argv) == 8:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    SAVE_PATH = '/home/mnt1/checkpoints/{}'.format(sys.argv[3])
    if sys.argv[4] == "ce":
       rank_loss_fn = BCERankingLoss()
    else:
       rank_loss_fn = RankingLoss(margin = 1.)
    LOGS = sys.argv[5]
    gpu_id = sys.argv[6]
    USE_MODEL = sys.argv[7]
    import os
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
else:
    raise Exception('please input correct args.')
# USE_MODEL = 'alexnet'
# USE_MODEL = 'vgg'
#USE_MODEL = 'resnet'
#USE_MODEL = 'alexnet_gap'
#USE_MODEL = 'alexnet_gap2'
print(USE_MODEL)
#model
train_batch_size = 128
test_batch_size = 32
INPUT_SIZE = (224,224)
os.environ['CUDA_VISIBLE_DEVICES'] = "%s"%gpu_id

device = torch.device('cuda:0')
if USE_MODEL == 'resnet':
    model = resnet50(pretrained = True).to(device)
    model_dim = 2048
elif USE_MODEL == 'alexnet':
    model = alexnet(pretrained = True).to(device)
    model_dim = 4096
    train_batch_size = 128
    test_batch_size = 128
    INPUT_SIZE = (227,227)
elif USE_MODEL == 'vgg':
    model = vgg16_bn(pretrained = True).to(device)
    model_dim = 4096
    train_batch_size = 64
    test_batch_size = 64
elif USE_MODEL == 'alexnet_gap':
    model = alexnet_gap(pretrained=True).to(device)
    dicts = torch.load('/home/repo_disk/celeb_alexnet_gap2_pkl/alexnet_gap_checkpt_80.pkl') #0.77
    model_state_dict = dicts['model_state_dict']
    model.load_state_dict(model_state_dict)

    model_dim = 512
    train_batch_size = 128
    test_batch_size = 128
    INPUT_SIZE = (227,227)
elif USE_MODEL == 'alexnet_gap2':
    model = alexnet_gap2(pretrained=False).to(device)
    dicts = torch.load('/home/repo_disk/celeb_alexnet_gap2_pkl/alexnet_gap2_checkpt_90.pkl') #0.77
    model_state_dict = dicts['model_state_dict']
    model.load_state_dict(model_state_dict)

    model_dim = 128
    train_batch_size = 128*4
    test_batch_size = 128
    INPUT_SIZE = (227,227)
else:
    print('please select proper model such as \'alexnet\'')
    quit()
#========== load data ===================================================
kwargs = {'num_workers': 4, 'pin_memory': True}

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="%s.log"%(LOGS), filemode="w")
logger.info('Select pretrained model:%s'%USE_MODEL)
logger.info('selected device:%s'%device)
logger.info('input size:{}'.format(INPUT_SIZE))
logger.info('save path:%s'%SAVE_PATH)
logger.info("train_file:%s test_file:%s"%(train_file, test_file))

train_data = myDataSet(txt = train_file,
        transform = transforms.Compose(
            [transforms.Resize((256,256)),
             transforms.RandomCrop(INPUT_SIZE),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.4374, 0.3808, 0.3378),
#                 std=(0.2906, 0.2684, 0.2620))
#             transforms.Normalize(mean = (0.485, 0.456, 0.406),
#                 std=[0.229, 0.224, 0.225])
            ]
            ))
train_loader = DataLoader(dataset=train_data, batch_size = train_batch_size,
            shuffle=True, **kwargs)

logger.info('loading train_data:%s'%len(train_data))

test_data = myDataSet(txt = test_file,
        transform = transforms.Compose(
            [transforms.Resize(INPUT_SIZE),
             transforms.ToTensor()]
            ))
logger.info('loading test_data:%s'%len(test_data))
test_loader = DataLoader(dataset= test_data, batch_size = test_batch_size,
        shuffle = False, **kwargs)

logger.info('train_batch_size:%s test_batch_size:%s'%(train_batch_size, test_batch_size))
#============= construct network ========================
TASK_NUM = 6 #7 #1 #10
score_functions = nn.ModuleList([nn.Linear(model_dim * 2, 3).to(device) for i in range(TASK_NUM)])

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0., 0.01)
        nn.init.constant_(m.bias.data, 0.1)
for i in range(TASK_NUM):
    weight_init(score_functions[i])

#============== loss functions ===========================================
logger.info('ranking loss:{}'.format(rank_loss_fn))
#============= parameters for whole model ===============================
lr_m = 1e-5
lr_fc = 1e-2

logger.info('Initial lr-model:%s lr-fc:%s'%(lr_m, lr_fc))
params_a = [{'params': model.parameters(), 'lr':lr_m},
            {'params': score_functions.parameters(), 'lr':lr_fc}]
#update one part params.
optimizer_a = optim.SGD(params_a, lr = lr_fc, momentum=0.9, weight_decay=0.)
multisteps = [10, 20] #[40, 80, 120]
logger.info('multisteps:{}'.format(multisteps))
scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=multisteps, gamma=0.1) # best gamma = 0.1 for vgg
#show interval
log_interval = 20
SAVE_INTERVAL= 1
#======================= train phase ======================================
def train(epoch):
    model.train()
    score_functions.train()

    for step, (img1, img2, label1, label2, _, _) in enumerate(train_loader):
        img1, img2, label1, label2 = img1.to(device), img2.to(device),label1.to(device), label2.to(device)
        label2_sorted, index = torch.sort(label2)
        img1, img2, label1 = img1[index], img2[index], label1[index]

        optimizer_a.zero_grad()
        bz = label1.shape[0]
        output1 = model(img1)
        output2 = model(img2)
        pred = torch.zeros((label2.shape[0], 3), device = device)
        for k in range(bz): #for a batch
            idx = int(label2[k])
            train_score1[k] = score_functions[idx](output1[k])
            train_score2[k] = score_functions[idx](output2[k])
        cls_loss = rank_loss_fn(train_score1, train_score2, label1)

        cls_loss.backward()
        optimizer_a.step()

        #show info.
        if step % log_interval == 0:
            logger.info('Epoch:%s |iter:%s |cls_loss:%s'%(
                    epoch, step,
                    cls_loss.item()
                    ))
    if epoch-1 in multisteps:
        logger.info('adjust learning rate...')
        for param_group in optimizer_a.param_groups:
            logger.info('lr:%s'%param_group['lr'])

    #save params
    if epoch % SAVE_INTERVAL == 0:
        torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(),
			    'score_funcs_dict':score_functions.state_dict()},
                '{}/naive_{}_checkpt_{}.pkl'.format(SAVE_PATH, USE_MODEL, epoch))
#================= test phase =================================================

def test(epoch):
    model.eval()
    score_functions.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    cnt = 0
    y_true  = dict()
    y_pred1 = dict()
    y_pred2 = dict()
    for i in range(TASK_NUM):
        y_true[i] = []
        y_pred1[i] = []
        y_pred2[i] = []

    margin1 = 0.5
    margin2 = 0.
    with torch.no_grad():
        for step, (img1, img2, label1, label2, _, _) in enumerate(test_loader):
            img1, img2, label1, label2 = img1.to(device), img2.to(device),label1.to(device), label2.to(device)
            bz = img1.shape[0]
            output1 = model(img1)
            output2 = model(img2)
            test_score1 = torch.zeros(label2.shape[0], device = device)
            test_score2 = torch.zeros(label2.shape[0], device = device)
            for k in range(bz): #for a batch
                idx = int(label2[k]) #which task
                y_true[idx].append(label1[k].cpu().item())
                test_score1[k] = score_functions[idx](output1[k])
                test_score2[k] = score_functions[idx](output2[k])
                score_diff = test_score1[k] - test_score2[k]

                #margin@0.01
                pred = score_diff.clone()
                pred[pred > margin1] = 1
                pred[pred < -margin1] = -1
                pred[abs(pred) <= margin1] = 0
                y_pred1[idx].append(pred.cpu().item())
                correct += pred.eq(label1[k].view_as(pred)).item()
                #margin@1e-9
                pred = score_diff.clone()
                pred[pred > margin2] = 1
                pred[pred < -margin2] = -1
                pred[abs(pred) <= margin2] = 0
                y_pred2[idx].append(pred.cpu().item())
                correct2 += pred.eq(label1[k].view_as(pred)).item()

            cls_loss = rank_loss_fn(test_score1 , test_score2, label1)
            test_loss += cls_loss.item()
            cnt += 1

    #eval.
    test_loss /= cnt
    logger.info('\n Test set {}: average loss:{:.4f}, acc@{}:{},acc@{}:{}'.format(
            len(test_loader.dataset),test_loss, margin1,
            correct/len(test_loader.dataset), margin2,
            correct2/len(test_loader.dataset)))
    #for each task
    acc1 = [0]*TASK_NUM
    acc2 = [0]*TASK_NUM
    micro_f1 = [0]*TASK_NUM
    macro_f1 = [0]*TASK_NUM
    for i in range(TASK_NUM):
        acc1[i] = round(accuracy_score(y_true[i], y_pred1[i]), 4)
        acc2[i] = round(accuracy_score(y_true[i], y_pred2[i]), 4)
        micro_f1[i] = round(f1_score(y_true[i], y_pred1[i], average = 'micro'), 4)
        macro_f1[i] = round(f1_score(y_true[i], y_pred1[i], average = 'macro'), 4)

    logger.info('Margin1 Each task\'s acc@{}:'.format(margin1))
    logger.info("{}".format(acc1) + ' mean:%s'%round(np.mean(acc1), 4))
    logger.info('Margin2 Each task\'s acc@{}:'.format(margin2))
    logger.info("{}".format(acc2) + ' mean:%s'%round(np.mean(acc2), 4))
    logger.info('margin1 Each task\'s f1_score@{}'.format(margin1))
    logger.info('micro:{} mean:{} macro:{} mean:{}'.format(micro_f1, round(np.mean(micro_f1), 4),
            macro_f1, round(np.mean(macro_f1), 4)))
    #f1-score
    y_true = np.concatenate(list(y_true.values()))
    y_pred = np.concatenate(list(y_pred1.values()))
    y_pred2 = np.concatenate(list(y_pred2.values()))
    micro_f1 = f1_score(y_true, y_pred, average = 'micro')
    macro_f1 = f1_score(y_true, y_pred, average = 'macro')
    logger.info('Margin@0.5 micro f1:{} macro f1:{}'.format(micro_f1, macro_f1))
    micro_f1_2 = f1_score(y_true, y_pred2, average = 'micro')
    macro_f1_2 = f1_score(y_true, y_pred2, average = 'macro')
    logger.info('Margin@0. micro f1:{} macro f1:{}'.format(micro_f1_2, macro_f1_2))
#===================== training/testing =============================
EPOCHS = 30 #200
for epoch in range(1 , EPOCHS+1):
    scheduler_a.step()
    train(epoch)
    test(epoch)
