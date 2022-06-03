""" Main module for testing optimizers """
# Load libraries and pick the CUDA device if available
import json
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
# Custom libraries

from CustomOptimizer import *

from src.attack import FastGradientSignUntargeted
#from LinfPGDAttack import *
#import torchattacks


from loader import *
from src.models import MODELS_MAP
from src.utils import *


def flat_weight_dump(model):
    """ Returns a 1-d tensor containing all the network weights """
    is_empty = True
    for _, param in model.named_parameters():
        if param.requires_grad:
            if is_empty:
                flat_tensor = param.data.flatten()
                is_empty = False
            else:
                flat_tensor = torch.cat([flat_tensor, param.data.flatten()])
    return flat_tensor


def tb_dump(epoch, net, writer1,writer2):
    """ Routine for dumping info on tensor board at the end of an epoch """
    print('=> eval on test data')
    (test_loss, test_acc, adv_test_loss, adv_test_acc, _) = test(testloader, net, device)
    writer1.add_scalar('Loss/test', test_loss, epoch)
    writer1.add_scalar('Accuracy', test_acc, epoch)
   # writer1.add_scaler('Adversarial Loss Test', adv_test_loss, epoch)
    writer1.add_scalar('Adversarial Accuracy', adv_test_acc, epoch)

    print('=> eval on train data')
    (train_loss, train_acc, adv_train_loss, adv_train_acc, _) = test(trainloader, net, device)
    writer2.add_scalar('Loss/train', train_loss, epoch)
    writer2.add_scalar('Accuracy', train_acc, epoch)
   # writer2.add_scalar('Adversarial Loss Train', adv_train_loss, epoch)
    writer2.add_scalar('Adversarial Accuracy', adv_train_acc, epoch)
    print('epoch %d done\n' % (epoch))


def test(model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1 

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output = model(data, _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                
                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = attack.perturb(data, 
                                                       pred if use_pseudo_label else label, 
                                                       'mean', 
                                                       False)

                    adv_output = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num , total_adv_acc / num


# Trains the network
def train(model,tr_loader,va_loader=None, adv_train=False):
        

        opt = optimizer
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, 
                                                         milestones=[40000, 60000], 
                                                         gamma=0.1)
        _iter = 0

        begin_time = time.time()

        for epoch in range(1, config_epochs+1):
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # When training, the adversarial example is created from a random 
                    # close point to the original data point. If in evaluation mode, 
                    # just start from the original data point.
                    adv_data = attack.perturb(data, label, 'mean', True)
                    output = model(adv_data, _eval=False)
                else:
                    output = model(data, _eval=False)

                loss = F.cross_entropy(output, label)

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % config_n_eval_step == 0:
                    t1 = time.time()

                    if adv_train:
                        with torch.no_grad():
                            stand_output = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:
                        
                        adv_data = attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        # print(label)
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    t2 = time.time()

                    logger.info(f'epoch: {epoch}, iter: {_iter}, lr={opt.param_groups[0]["lr"]}, '
                                f'spent {time.time()-begin_time:.2f} s, tr_loss: {loss.item():.3f}')

                    logger.info(f'standard acc: {std_acc:.3f}%, robustness acc: {adv_acc:.3f}%')

                    # begin_time = time()

                    # if va_loader is not None:
                    #     va_acc, va_adv_acc = self.test(model, va_loader, True)
                    #     va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                    #     logger.info('\n' + '='*30 + ' evaluation ' + '='*30)
                    #     logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    #         va_acc, va_adv_acc, time() - begin_time))
                    #     logger.info('='*28 + ' end of evaluation ' + '='*28 + '\n')

                    begin_time = time.time()

                

                _iter += 1
                # scheduler depends on training interation
                scheduler.step()

            if va_loader is not None:
                t1 = time.time()
                va_acc, va_adv_acc = test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time.time()
                logger.info('\n'+'='*20 +f' evaluation at epoch: {epoch} iteration: {_iter} ' \
                    +'='*20)
                logger.info(f'test acc: {va_acc:.3f}%, test adv acc: {va_adv_acc:.3f}%, spent: {t2-t1:.3f} s')
                logger.info('='*28+' end of evaluation '+'='*28+'\n')


# ################
parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
parser.add_argument(
    '--config', default='config.json', type=str, help='config file')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.load(config_file)
config_experiment_number = config['experiment_number']
config_dataset = config['dataset']
config_architecture = config['architecture']
config_batch_size = config['batch_size']
config_optimizer = config['optimizer']
config_lr = config['lr']
config_momentum = config['momentum']
config_prune_epoch = config["prune_epoch"]
config_unfreeze_epoch = config["unfreeze_epoch"]
config_perc_to_prune = config['perc_to_prune']
config_step_of_prune = config["step_of_prune"]
config_radius = config['radius']
config_epochs = config['epochs']
config_tb_path_test = config['tb_path_test']
config_tb_path_train = config['tb_path_train']
config_batch_statistics_freq = config['batch_statistics_freq']
config_dump_movement = bool(config['dump_movement'] == 1)
config_projected = bool(config['projected'] == 1)
config_weight_decay = config['weight_decay']
config_radius = config['radius']
config_random_seed = config['random_seed']
config_gamma0 = config['gamma0']

config_one_shot_prune = config["one_shot_prune"]
config_iterative_prune = config["iterative_prune"]
config_epochs_to_finetune = config["epochs_to_finetune"]
config_epochs_to_densetrain = config["epochs_to_densetrain"]
#config_initial_accumulator_value = config['initial_accumulator_value']
#config_beta = config['beta']
#config_eps = config['eps']

config_n_eval_step = config['n_eval_step']
config_log_folder = config['log_folder']
config_epsilon = config['epsilon']
config_k = config['k']
config_alpha = config['alpha']




use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Set random seed
torch.manual_seed(config_random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config_random_seed)

# Load data
if config_dataset == 'MNIST':
    trainloader, testloader = mnist_loader(batch_size=config_batch_size)
else:
    trainloader, testloader = cifar_loader(batch_size=config_batch_size)



model = MODELS_MAP[config_architecture]()
net = model.to(device)

criterion = nn.CrossEntropyLoss()

if config_optimizer == 0:
    optimizer = optim.SGD(
      net.parameters(), lr=config_lr,
      momentum=config_momentum, weight_decay=config_weight_decay)
elif config_optimizer == 1:
    optimizer = optim.Adagrad(
      net.parameters(), lr=config_lr, weight_decay=config_weight_decay)
elif config_optimizer == 2:
    optimizer = optim.Adam(net.parameters(), lr=config_lr, amsgrad=0, weight_decay=config_weight_decay)
elif config_optimizer == 3:
    optimizer = optim.Adam(net.parameters(), lr=config_lr, amsgrad=1, weight_decay=config_weight_decay)
elif config_optimizer == 4:
    optimizer = optim.RMSprop(net.parameters(), lr=config_lr)
elif config_optimizer == 5:
    optimizer = AdaptiveLinearCoupling(
        net.parameters(), lr=config_lr,
        weight_decay=config_weight_decay)
elif config_optimizer == 6:
    #optimizer = AdaACSA(
    #    net.parameters(), lr=config_lr, radius=1, projected=config_projected)
    optimizer = AdaACSA(
        net.parameters(), lr=config_lr, radius=config_radius,
        weight_decay=config_weight_decay, projected=config_projected,
        gamma0=config_gamma0, beta=config_beta,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 7:
    optimizer = AdaAGDplus(
        net.parameters(), lr=config_lr, radius=config_radius, projected=config_projected,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 8:
    optimizer = AdaJRGS(
        net.parameters(), lr=config_lr, radius=config_radius, projected=config_projected,
        initial_accumulator_value=config_initial_accumulator_value,
        eps=config_eps)
elif config_optimizer == 9:
    optimizer = CustomOptimizer(net.parameters(),lr=config_lr, 
    momentum=config_momentum,
    weight_decay=config_weight_decay,
    len_step = len(trainloader),
    
    one_shot_prune  = config_one_shot_prune,
    prune_epoch=config_prune_epoch,
    step_of_prune=config_step_of_prune,
    perc_to_prune = config_perc_to_prune,

    iterative_prune = config_iterative_prune,
    unfreeze_epoch=config_unfreeze_epoch,
    epochs_to_densetrain = config_epochs_to_densetrain,
    epochs_to_finetune= config_epochs_to_finetune
   )



# Writer path for display on TensorBoard
if not os.path.exists(config_tb_path_test):
    os.makedirs(config_tb_path_test)
if not os.path.exists(config_tb_path_train):
    os.makedirs(config_tb_path_train)

if not os.path.exists(config_log_folder):
    os.makedirs(config_log_folder)
#path_name = config_tb_path + \
#    str(config_experiment_number) + "_" + str(optimizer)

logger = create_logger(config_log_folder,'train','info')


# Initialize weights
net.apply(weights_init_uniform_rule)

#load the model
'''
ckp_path = 'checkpoints/epoch_model_40.pth'
checkpoint_model, start_epoch = load_ckp(ckp_path, net)
'''
attack = FastGradientSignUntargeted(net, 
                                        config_epsilon, 
                                        config_alpha, 
                                        min_val=0, 
                                        max_val=1, 
                                        max_iters=config_k, 
                                        _type='linf')    


train(net,trainloader,testloader,adv_train=True)

# Dump some info on the range of parameters after training is finished
for param in net.parameters():
    print(str(torch.min(param.data).item()) + " " + str(torch.max(param.data).item()))
