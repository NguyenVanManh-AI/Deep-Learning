from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34, resnet18
from model.basenet import AlexNetBase, VGGBase, MLP_Classifier, GCN
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser(description='TRUNGQUANG_AI')
parser.add_argument('--steps', type=int, default=10000, metavar='N',
                    help='maximum number of iterations '' to train (default: 50000)')
parser.add_argument('--method', type=str, default='Using GCN Classifier')
parser.add_argument('--lr', type=float, default=0.01,
                    metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1,
                    metavar='MLT', help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05,
                    metavar='T', help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1,
                    metavar='LAM', help='value of lamda')
parser.add_argument('--save_check', action='store_true',
                    default=False, help='save checkpoint or not')
parser.add_argument('--checkpath', type=str,
                    default='./save_model_ssda', help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging ''training status')
parser.add_argument('--save_interval', type=int, default=500,
                    metavar='N', help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--data', type=str, default='amazon', help='source domain')
parser.add_argument('--dataset', type=str, default='office',
                    help='the name of dataset')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvement ''before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false',
                    default=True, help='early stopping on validation or not')

args = parser.parse_args()
print('Dataset %s Data %s Network %s' % (args.dataset, args.data, args.net))

use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir, '%s_net_%s_%s' %
                           (args.method, args.net, args.data))

labeled_data_loader, unlabeled_data_loader_test, class_list = return_dataset(
    args)

torch.cuda.manual_seed(args.seed)

if args.net == 'resnet18':
    G = resnet18()
    inc = 512
elif args.net == "resnet34":
    G = resnet34()
    inc = 512
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

GCN = GCN(in_features=inc, edge_features=512, out_feature=512, device=DEVICE)

MLP_Classifier = MLP_Classifier(num_class=len(class_list), inc=inc)

weights_init(MLP_Classifier)
lr = args.lr
G.to(DEVICE)
MLP_Classifier.to(DEVICE)

im_data_labeled = torch.FloatTensor(1)
im_data_unlabeled = torch.FloatTensor(1)
gt_labels_labeled = torch.LongTensor(1)
gt_labels_unlabeled = torch.LongTensor(1)

im_data_labeled = im_data_labeled.to(DEVICE)
im_data_unlabeled = im_data_unlabeled.to(DEVICE)
gt_labels_labeled = gt_labels_labeled.to(DEVICE)
gt_labels_unlabeled = gt_labels_unlabeled.to(DEVICE)

im_data_labeled = Variable(im_data_labeled)
im_data_unlabeled = Variable(im_data_unlabeled)
gt_labels_labeled = Variable(gt_labels_labeled)
gt_labels_unlabeled = Variable(gt_labels_unlabeled)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)


def train():
    G.train()
    MLP_Classifier.train()
    GCN.train()

    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_gcn = optim.SGD(
        list(GCN.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(MLP_Classifier.parameters(
    )), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_gcn.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])

    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    param_lr_gcn = []
    for param_group in optimizer_gcn.param_groups:
        param_lr_gcn.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    criterion_gedge = nn.BCELoss(reduction='mean').to(DEVICE)

    all_step = args.steps

    data_iter_labeled = iter(labeled_data_loader)
    len_train_labeled = len(labeled_data_loader)

    best_acc = 0
    counter = 0
    best_acc_test = 0
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(
            param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(
            param_lr_f, optimizer_f, step, init_lr=args.lr)
        optimizer_gcn = inv_lr_scheduler(
            param_lr_gcn, optimizer_gcn, step, init_lr=args.lr)

        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_labeled == 0:
            data_iter_labeled = iter(labeled_data_loader)

        data_labeled = next(data_iter_labeled)

        im_data_labeled.resize_(data_labeled[0].size()).copy_(data_labeled[0])
        gt_labels_labeled.resize_(
            data_labeled[1].size()).copy_(data_labeled[1])

        zero_grad_all()

        # Extract features (representations)
        output = G(im_data_labeled)

        # make forward pass for gnn head
        gcn_features, edge_sim = GCN(output)

        # compute edge loss
        edge_gt, edge_mask = GCN.label2edge(gt_labels_labeled.unsqueeze(dim=0))

        # training edge network using Binary Cross Entropy
        edge_loss = criterion_gedge(edge_sim.masked_select(
            edge_mask), edge_gt.masked_select(edge_mask))

        prediction = MLP_Classifier(gcn_features)

        # training edge network using Cross-Entropy Loss
        node_loss = criterion(prediction, gt_labels_labeled)

        GCN_loss = node_loss + 0.3*edge_loss

        GCN_loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        optimizer_gcn.step()
        zero_grad_all()

        log_train = 'Data {} Train Ep: {} lr{} \t ' 'Node Loss Classification: {:.4f} Edge Loss Classification: {:.4f} Method {}\n'\
            .format(args.data, step, lr, node_loss.data, edge_loss.data, args.method)

        G.zero_grad()
        MLP_Classifier.zero_grad()
        GCN.zero_grad()

        zero_grad_all()

        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step > 0:
            loss_test, acc_test = test(unlabeled_data_loader_test)

            G.train()
            MLP_Classifier.train()
            GCN.train()

            if acc_test >= best_acc:
                best_acc = acc_test

            print('best acc test %f' % best_acc)
            print('record %s' % record_file)

            G.train()
            MLP_Classifier.train()
            GCN.train()

            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(), os.path.join(
                    args.checkpath, "G_iter_model_{}_{}_step_{}.pth.tar".format(args.method, args.data, step)))
                torch.save(MLP_Classifier.state_dict(), os.path.join(
                    args.checkpath, "MLP_Classifier_iter_model_{}_{}_step_{}.pth.tar".format(args.method, args.data, step)))


def test(loader):
    G.eval()
    MLP_Classifier.eval()
    GCN.eval()

    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    confusion_matrix = torch.zeros(num_class, num_class)

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_unlabeled.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_unlabeled.resize_(data_t[1].size()).copy_(data_t[1])

            feat = G(im_data_unlabeled)
            aggre_feature, _ = GCN(feat)
            output1 = MLP_Classifier(aggre_feature)

            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_unlabeled.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_unlabeled.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_unlabeled.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_unlabeled) / len(loader)
    print('\nTest set: Average loss: {:.4f}, ''Accuracy: {}/{} MLP_Classifier ({:.0f}%)\n'.format(
        test_loss, correct, size, 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()
