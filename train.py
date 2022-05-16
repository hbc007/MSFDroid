import logging
import sys, json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.optim import lr_scheduler

from model import Model
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders, JsonLogger, get_dataloaders
from dataset import HybridModelDataset, hybrid_model_collate_fn, FMSAMDDataset, FMSAMD_collate_fn
import os

torch.set_printoptions(profile="full")


def get_free_gpu_id():
    """
    Get the first free GPU
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [
        int(x.split()[2]) for x in open('tmp', 'r').readlines()
    ]
    return memory_available.index(max(memory_available))


# Logger
config_logging('')
models = 'hpic'

batch_size = 16
fs = 10
lr = 0.5e-2
momentum = 0.9
vote_threshold = 0.5

device = None

if len(sys.argv) == 2:
    conf = json.loads(sys.argv[1])
    models = conf['models']
    vote_threshold = float(conf['vote_threshold'])
    device = torch.device(conf['device'])
if not device:
    device = torch.device(f"cuda:{get_free_gpu_id()}")

# Dataloader
'''
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, dataset_size = prepare_dataloaders(
    dataset=HybridModelDataset,
    collate_fn=hybrid_model_collate_fn,
    batch_size=batch_size,
    fs=fs)
    
'''
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = get_dataloaders(
    ben_path="/home/user1/projects/hubochao/datasets/benign/",
    mal_path="/home/user1/projects/hubochao/datasets/malware/",
    dict_path="/home/user1/projects/hubochao/my/dict.json",
    dataset=FMSAMDDataset,
    collate_fn=FMSAMD_collate_fn,
    batch_size=batch_size,
    num_workers=24,
    fs=fs)
# Model

model = Model(models=models,
              ada_voting=vote_threshold != 0,
              vote_threshold=vote_threshold,
              device=device)

model = model.to(device)
num_params = 0
for param in model.parameters():
    num_params += param.numel()
print('Params:', num_params)


# Train process
def train(model, device, train_loader, val_loader):
    jsonlogger = JsonLogger(models, vote_threshold)
    jsonlogger.log('models', models)
    jsonlogger.log('batch_size', batch_size)
    jsonlogger.log('vote_threshold', vote_threshold)

    logging.info(
        f"batch_size:{batch_size} fs:{fs} lr:{lr} momentum:{momentum}")
    saver = BestSaver(f"{models}_adav{vote_threshold}")

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum)  #1e-2
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs = 100
    for epoch in range(1, epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))
        total_losses = AverageMeter()
        sv_losses = AverageMeter()  #软投票损失
        header_losses = AverageMeter()
        pxx_losses = AverageMeter()
        ip_losses = AverageMeter()
        cb_losses = AverageMeter()
        # Train phase
        model.train()
        for batch_num, batch in enumerate(train_loader, 1):
            header, pxx, ip, is_mal = batch
            #print(ip[0])
            #exit()
            #print(header.shape, pxx.shape, ip.shape, is_mal.shape)
            # [16, 128] [16, 128] [16, 4380] [16]

            header = header.to(model.device)
            pxx = pxx.to(model.device)
            ip = ip.to(model.device)
            is_mal = is_mal.float().to(model.device)

            ##exit()
            (_,
             header_loss), (_,
                            pxx_loss), (_,
                                        ip_loss), (_,
                                                   cb_loss), (_,
                                                              sv_loss) = model(
                                                                  header, pxx,
                                                                  ip, is_mal)
            total_loss = header_loss + pxx_loss + ip_loss + cb_loss + sv_loss
            header_losses.update(header_loss.item(), batch[0].shape[0])
            pxx_losses.update(pxx_loss.item(), batch[0].shape[0])
            ip_losses.update(ip_loss.item(), batch[0].shape[0])
            cb_losses.update(cb_loss.item(), batch[0].shape[0])
            sv_losses.update(sv_loss.item(), batch[0].shape[0])
            total_losses.update(total_loss.item(), batch[0].shape[0])
            # Backpropagation
            model.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            if batch_num % 300 == 0:
                model.update_temperature()
            if batch_num % 10 == 0:
                logging.info(
                    "epoch:{}/{} batch:{}/{} # Loss: total: {:.4f} header: {:.4f} pxx: {:.4f} ip: {:.4f} cb: {:.4f} sv: {:.4f}"
                    .format(epoch, epochs, batch_num,
                            len(train_dataset) // batch_size, total_losses.val,
                            header_losses.val, pxx_losses.val, ip_losses.val,
                            cb_losses.val, sv_losses.val))
        logging.info("Train Loss (total_loss): {:.4f}".format(
            total_losses.avg))
        scheduler.step()

        # Valid Phase
        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        total_losses = AverageMeter()
        header_losses = AverageMeter()
        pxx_losses = AverageMeter()
        ip_losses = AverageMeter()
        cb_losses = AverageMeter()
        sv_losses = AverageMeter()
        header_predicts = []
        pxx_predicts = []
        ip_predicts = []
        cb_predicts = []
        sv_predicts = []
        targets = []
        for batch_num, batch in enumerate(val_loader, 1):
            header, pxx, ip, is_mal = batch
            header = header.to(model.device)
            pxx = pxx.to(model.device)
            ip = ip.to(model.device)
            is_mal = is_mal.float().to(model.device)
            target = is_mal
            with torch.no_grad():
                (header_predict, header_loss), (pxx_predict, pxx_loss), (
                    ip_predict,
                    ip_loss), (cb_predict,
                               cb_loss), (sv_predict, sv_loss) = model.forward(
                                   header, pxx, ip, is_mal)
            # Loss Update
            total_loss = header_loss + pxx_loss + ip_loss + cb_loss + sv_loss
            header_losses.update(header_loss.item(), batch[0].shape[0])
            pxx_losses.update(pxx_loss.item(), batch[0].shape[0])
            ip_losses.update(ip_loss.item(), batch[0].shape[0])
            cb_losses.update(cb_loss.item(), batch[0].shape[0])
            sv_losses.update(sv_loss.item(), batch[0].shape[0])
            total_losses.update(total_loss.item(), batch[0].shape[0])
            # Prediction Update
            header_predicts.append(header_predict)
            pxx_predicts.append(pxx_predict)
            ip_predicts.append(ip_predict)
            cb_predicts.append(cb_predict)
            sv_predicts.append(sv_predict)
            # Target Update
            targets.append(target)

        sv_weight = model.get_vote_weight().cpu().data.numpy()
        sv_weight_log = ''
        sv_weight_i = 0
        if header_loss > 0:
            sv_weight_log += 'header: {:.4f} '.format(sv_weight[sv_weight_i])
            sv_weight_i += 1
        if pxx_loss > 0:
            sv_weight_log += 'pxx: {:.4f} '.format(sv_weight[sv_weight_i])
            sv_weight_i += 1
        if ip_loss > 0:
            sv_weight_log += 'ip: {:.4f} '.format(sv_weight[sv_weight_i])
            sv_weight_i += 1
        if cb_loss > 0:
            sv_weight_log += 'cb: {:.4f} '.format(sv_weight[sv_weight_i])
            sv_weight_i += 1

        logging.info(f"SoftVoting Weight: " + sv_weight_log)
        logging.info("Valid Loss (total_loss): {:.4f}".format(
            total_losses.avg))

        targets = torch.cat(targets).cpu().data.numpy()

        outputs = torch.cat(sv_predicts).cpu().data.numpy()
        auc = metrics.roc_auc_score(targets, outputs)

        header_auc = 0
        pxx_auc = 0
        ip_auc = 0
        cb_auc = 0
        if 'h' in models:
            header_predicts = torch.cat(header_predicts).cpu().data.numpy()
            header_auc = metrics.roc_auc_score(targets, header_predicts)

        if 'p' in models:
            pxx_predicts = torch.cat(pxx_predicts).cpu().data.numpy()
            pxx_auc = metrics.roc_auc_score(targets, pxx_predicts)

        if 'i' in models:
            ip_predicts = torch.cat(ip_predicts).cpu().data.numpy()
            ip_auc = metrics.roc_auc_score(targets, ip_predicts)

        if 'c' in models:
            cb_predicts = torch.cat(cb_predicts).cpu().data.numpy()
            cb_auc = metrics.roc_auc_score(targets, cb_predicts)
        logging.info(
            "AUC:{:.4f}, header:{:.4f}, pxx:{:.4f}, ip:{:.4f}, cb:{:.4f}".
            format(auc, header_auc, pxx_auc, ip_auc, cb_auc))

        predicts = np.where(outputs > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(targets, predicts)
        #accuracy = metrics.accuracy_score(predicts, targets)
        logging.info("Accuracy@0.5:{:.4f}".format(accuracy))
        positive_loss = -np.log(outputs[targets == 1]).mean()
        f1 = metrics.f1_score(targets, predicts)
        logging.info("F1:{:.4f}".format(f1))

        recall = metrics.recall_score(targets, predicts)
        logging.info("Recall:{:.4f}".format(recall))

        cm = metrics.confusion_matrix(targets, predicts)
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        TP = cm[1][1]
        '''
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        '''
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        logging.info("TPR:{:.4f}".format(TPR))
        logging.info("FPR:{:.4f}".format(FPR))
        logging.info("Positive loss:{:.4f}".format(positive_loss))
        positive_acc = sum(outputs[targets == 1] > 0.5) / len(outputs)
        logging.info("Positive accuracy:{:.4f}".format(positive_acc))

        if jsonlogger.get('best_auc') is None or auc > jsonlogger.get(
                'best_auc'):
            jsonlogger.log('best_auc', auc)
            jsonlogger.log('best_header_auc', header_auc)
            jsonlogger.log('best_pxx_auc', pxx_auc)
            jsonlogger.log('best_ip_auc', ip_auc)
            jsonlogger.log('best_cb_auc', cb_auc)
            jsonlogger.log('best_acc', accuracy)
            jsonlogger.log('best_f1', f1)
            jsonlogger.log('best_recall', recall)
            jsonlogger.log('best_tpr', TPR)
            jsonlogger.log('best_fpr', FPR)
            jsonlogger.log('best_positive_acc', positive_acc)
            jsonlogger.log('best_epoch', epoch)

        jsonlogger.log('epoch', epoch)
        jsonlogger.append('header_auc', header_auc)
        jsonlogger.append('pxx_auc', pxx_auc)
        jsonlogger.append('ip_auc', ip_auc)
        jsonlogger.append('cb_auc', cb_auc)
        jsonlogger.append('auc', auc)
        jsonlogger.append('acc', accuracy)
        jsonlogger.append('f1', f1)
        jsonlogger.append('recall', recall)
        jsonlogger.append('tpr', TPR)
        jsonlogger.append('fpr', FPR)

        jsonlogger.append('header_loss', header_losses.avg)
        jsonlogger.append('pxx_loss', pxx_losses.avg)
        jsonlogger.append('ip_loss', ip_losses.avg)
        jsonlogger.append('cb_loss', cb_losses.avg)
        jsonlogger.append('sv_loss', sv_losses.avg)
        jsonlogger.append('sv_weight', sv_weight.tolist())
        jsonlogger.append('total_loss', total_losses.avg)
        # Save best model
        saver.save(auc, model.state_dict(), epoch)
        jsonlogger.save()


def main():
    train(model, device, train_loader, val_loader)


if __name__ == "__main__":
    main()
