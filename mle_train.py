import os
import torch
from utils import myDataSet
from utils import MarginMLELoss
from backbone.Resnet import resnet50
from backbone.alexnet import alexnet
from backbone.alexnet_gap import alexnet_gap
from backbone.vgg import vgg16_bn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import sys
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import logging as logger


#=============== specify the dataset ============================
if len(sys.argv) == 7:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    SAVE_PATH = sys.argv[3]
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    LOGS = sys.argv[4]
    gpu_id = int(sys.argv[5])
    USE_MODEL = sys.argv[6]  # 'alexnet_gap', 'alexnet', 'vgg', 'resnet'
else:
    raise Exception('please input the train/test file')

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="%s.log"%(LOGS), filemode="w")
logger.info('save path:%s'%SAVE_PATH)
os.environ['CUDA_VISIBLE_DEVICES'] = "%s"%gpu_id
#torch.cuda.set_device(DEVICE_ID)
device = torch.device('cuda:0')
logger.info('selected device:{}'.format(device))
#================= pretrained model ==============================
EPOCHS = 30
logger.info('Select pretrained model--%s'%USE_MODEL)
#model
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 32
INPUT_SIZE = (227, 227)
if USE_MODEL == 'resnet':
    model = resnet50(pretrained = True).to(device)
    INPUT_SIZE=(224,224)

    model_dim = 2048
elif USE_MODEL == 'alexnet':
    model = alexnet(pretrained = True).to(device)
    model_dim = 4096
    TRAIN_BATCH_SIZE = 128*4
    TEST_BATCH_SIZE = 128*4
elif USE_MODEL == 'vgg':
    model = vgg16_bn(pretrained = True).to(device)
    INPUT_SIZE=(224, 224)
    model_dim = 4096
    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
elif USE_MODEL == 'alexnet_gap':
    model = alexnet_gap(pretrained=True).to(device)
    dicts = torch.load('/home/repo_disk/celeb_alexnet_gap2_pkl/alexnet_gap_checkpt_70.pkl')
    model_state_dict = dicts['model_state_dict']
    model.load_state_dict(model_state_dict)

    model_dim = 512
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
else:
    logger.info('please select proper model such as \'alexnet\'')
    quit()
#================ load data =======================================
train_data = myDataSet(txt = train_file,
        transform = transforms.Compose(
            [transforms.Resize((256,256)),
             transforms.RandomCrop(INPUT_SIZE),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.4374, 0.3808, 0.3378),
#                 std=(0.2906, 0.2684, 0.2620))
            ]
            ))
kwargs = {'num_workers': 4, 'pin_memory': True}
train_loader = DataLoader(dataset=train_data, batch_size = TRAIN_BATCH_SIZE,
            shuffle=True, **kwargs)

logger.info('loading train data:{} {} samples'.format(train_file, len(train_data)))

test_data = myDataSet(txt = test_file,
        transform = transforms.Compose(
            [transforms.Resize(INPUT_SIZE),
             transforms.ToTensor()]
            ))
logger.info('loading test data:{} {} samples'.format(test_file, len(test_data)))
test_loader = DataLoader(dataset= test_data, batch_size = TEST_BATCH_SIZE,
        shuffle = False, **kwargs)

logger.info('train batch size:%s test batch size:%s'%(train_loader.batch_size, test_loader.batch_size))

#================== fc layers for global features ===================
TASK_NUM = 10 #7 #1 #10
score_functions = nn.ModuleList([nn.Linear(model_dim, 1).to(device) for i in range(TASK_NUM)])
in_dim = 1 #model_dim
lamb_layers = nn.ModuleList([nn.Linear(in_dim,1, bias = False).to(device) for i in range(TASK_NUM)])

def weight_init(m, mean = 0., std = 0.01):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean, std)
        if m.bias:
            nn.init.constant_(m.bias.data, 0)

LAMBDA = 0.1
for i in range(TASK_NUM):
    #weight_init(score_functions[i])
    weight_init(lamb_layers[i], mean = LAMBDA, std=0.01)
logger.info('initial lambda:%s'%LAMBDA)
#=============== loss functions =====================================
rank_loss_fn = MarginMLELoss(device = device)
logger.info('loss function:{}'.format(rank_loss_fn))

#=============== parameters for whole model =========================
lr_m = 1e-3
lr_fc = 1e-3*10
lr_lamb = 1e-3  #
logger.info('Initial lr-model:%s lr-fc:%s lr-lamb:%s '%(lr_m, lr_fc, lr_lamb))
params_a = [{'params': model.parameters(), 'lr':lr_m},
            {'params': score_functions.parameters(), 'lr': lr_fc},
            {'params': lamb_layers.parameters(), 'lr': lr_lamb}
            ]
optimizer_a = optim.SGD(params_a,  momentum=0.9, weight_decay=5e-4)
#multisteps = [50, 100, 200, 300]
multisteps = [15, 20]
logger.info('Adjust learning rate steps:{}'.format(multisteps))
scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=multisteps, gamma=0.1)

#show interval
log_interval = 20
#========================= train phase ==========================
task_num = TASK_NUM
opti_b_start = 100
iters = 0
def train(epoch):
    model.train()
    score_functions.train()
    lamb_layers.train()
    global iters
    db_size = len(train_loader)
    for step, (img1, img2, label1, label2, name1, name2) in enumerate(train_loader):
        img1, img2, label1, label2 = img1.to(device), img2.to(device),label1.to(device), label2.to(device)

        niter = epoch*db_size + step
        optimizer_a.zero_grad()

        bz = label1.shape[0]
        output1 = model(img1)
        output2 = model(img2)
        train_score1 = torch.zeros(bz, device = device)
        train_score2 = torch.zeros(bz, device = device)

        lamb_in = torch.ones(bz, 1, device = device)
        lamb_out = torch.zeros(bz, 1, device = device)

        for k in range(bz): #for a batch
            idx = int(label2[k])

            train_score1[k] = score_functions[idx](output1[k])
            train_score2[k] = score_functions[idx](output2[k])
            lamb_out[k] = lamb_layers[idx](lamb_in[k])

        cls_loss = rank_loss_fn(train_score1, train_score2, lamb_out, label1)
        cls_loss.backward()
        optimizer_a.step()

        #show info.
        if step % log_interval == 0 or (epoch==1 and step==1):
            logger.info('Epoch:%s |iter:%s |cls_loss:%s '% (
                     epoch, step,
                     cls_loss.item()
                    ))
        iters += 1
    if epoch in multisteps:
        logger.info('adjust learning rate...')
        for param_group in optimizer_a.param_groups:
            logger.info('lr:%s'%param_group['lr'])

    #save params
    if epoch %5 == 0:
        file_name = '{}/{}_checkpt_{}.pkl'.format(SAVE_PATH, USE_MODEL, epoch)
        torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(),
			    'score_funcs_dict':score_functions.state_dict(),
                'lamb_layers_dict': lamb_layers.state_dict()
                },file_name)
        logger.info('save the immediate result:%s'%file_name)
#========================= test phase ======================================
def test(epoch):
    model.eval()
    score_functions.eval()
    lamb_layers.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    cnt = 0
    y_true = []
    y_pred = []
    y_pred_hard =  []
    y_pred_hard2 = []
    y_pred_ad = dict()
    y_true_ad = dict()

    for i in range(TASK_NUM):
        y_pred_ad[i] = []
        y_true_ad[i] = []

    hard_margin1, hard_margin2 = 0, 0
    with torch.no_grad():
        for step, (img1, img2, label1, label2, name1, name2) in enumerate(test_loader):
            y_true.append(label1.numpy())
            img1, img2, label1, label2 = img1.to(device), img2.to(device),label1.to(device), label2.to(device)
            output1 = model(img1)
            output2 = model(img2)
            bz = img1.shape[0]
            test_score1 = torch.zeros(label2.shape[0], device = device)
            test_score2 = torch.zeros(label2.shape[0], device = device)
            lamb_in = torch.ones(bz,1, device = device)
            lamb_out = torch.zeros(bz,1, device = device)
            for k in range(label2.shape[0]): #for a batch
                idx = int(label2[k])
                test_score1[k] = score_functions[idx](output1[k])
                test_score2[k] = score_functions[idx](output2[k])
                lamb_out[k] = lamb_layers[idx](lamb_in[k])
            cls_loss = rank_loss_fn(test_score1, test_score2, lamb_out, label1)
            test_loss += cls_loss.item()
            cnt += 1
            test_score1, test_score2 = test_score1.cpu(), test_score2.cpu()
            label1 = label1.cpu()


            #hard margin
            pred = test_score1 - test_score2
            hard_margin1 = 0.1
            pred[pred > hard_margin1] = 1
            pred[pred < -hard_margin1] = -1
            pred[abs(pred) <= hard_margin1] = 0
            pred = pred.type(torch.int)
            label1 = label1.type(torch.int)
            correct += pred.eq(label1.view_as(pred)).sum().item()
            y_pred_hard.append(pred)
            #hard margin2
            pred = test_score1 - test_score2
            hard_margin2 = 0.01
            pred[pred > hard_margin2] = 1
            pred[pred < -hard_margin2] = -1
            pred[abs(pred) <= hard_margin2] = 0
            pred = pred.type(torch.int)
            y_pred_hard2.append(pred)


            #adaptive threshold
            pred = test_score1 - test_score2
            margins = torch.zeros(bz)
            lamb_out.squeeze(0)
            for k in range(bz):
                idx = int(label2[k])
                y_true_ad[idx].append(label1[k].item())
                idx = int(label2[k])
                margin = lamb_out[k].item()
                margins[k] = margin
                pred[k][pred[k] > margin] = 1
                pred[k][pred[k] < -margin] = -1
                pred[k][abs(pred[k]) <= margin] = 0

                y_pred_ad[idx].append(pred[k].item())

            pred = pred.type(torch.int)
            y_pred.append(pred)

            correct2 += pred.eq(label1.view_as(pred)).sum().item()

    test_loss /= cnt
    logger.info('\n Test set {}: average loss:{:.4f}, acc@{}:{},acc@{}:{}'.format(
            len(test_loader.dataset),test_loss, hard_margin1,
            correct/len(test_loader.dataset), round(torch.mean(margins).item(), 4),
            correct2/len(test_loader.dataset)))
    acc_ad = [0]*TASK_NUM
    for i in range(TASK_NUM):
        acc_ad[i] = round(accuracy_score(y_true_ad[i], y_pred_ad[i]), 4)
    logger.info("MLE acc:{} |mean:{} ".format(acc_ad, np.mean(acc_ad)))


    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_pred_hard = np.concatenate(y_pred_hard)
    y_pred_hard2 = np.concatenate(y_pred_hard2)

    micro_f1 = f1_score(y_true, y_pred, average = 'micro')
    macro_f1 = f1_score(y_true, y_pred, average = 'macro')
    micro_f1_hard = f1_score(y_true, y_pred_hard, average = 'micro')
    macro_f1_hard = f1_score(y_true, y_pred_hard, average = 'macro')
    micro_f1_hard2 = f1_score(y_true, y_pred_hard2, average = 'micro')
    macro_f1_hard2 = f1_score(y_true, y_pred_hard2, average = 'macro')

    logger.info('MLE--micro f1:%.4f'%micro_f1 + ' macro f1:%.4f'%macro_f1 +
            'Hard margin@%s'%hard_margin1 + '--micro f1:%.4f'%micro_f1_hard +
            ' macro f1:%.4f'%macro_f1_hard + \
            'Har margin@%s'%hard_margin2 + '-- micro f1:%.4f'%micro_f1_hard2 + 'macro f1:%.4f'%macro_f1_hard2
            )

#============================ training/testing ============================
logger.info('Starting optimization... {} Epochs'.format(EPOCHS))
for epoch in range(EPOCHS):
    scheduler_a.step()
    train(epoch)
    test(epoch)
#    if epoch % 5==0:
#        test(epoch)

