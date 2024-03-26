from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
from collections import OrderedDict


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


class AlexNetBase(nn.Module):
    def __init__(self, pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pret)
        self.features = nn.Sequential(*list(model_alexnet.
                                            features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x



class MLP_Classifier(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(MLP_Classifier, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out


class Discriminator(nn.Module):
    def __init__(self, inc=4096):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(inc, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 1)

    def forward(self, x, reverse=True, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2_1(x))
        x_out = F.sigmoid(self.fc3_1(x))
        return x_out


class GCN(nn.Module):
    def __init__(self, in_features, edge_features, out_feature, device):
        super(GCN, self).__init__()

        self.edge_net = EdgeNet(in_features=in_features,
                                num_features=edge_features,
                                device=device)
        # set edge to node
        self.node_net = NodeNet(in_features=in_features,
                                num_features=out_feature,
                                device=device)
        # mask value for no-gradient edges
        self.mask_val = -1

    def label2edge(self, targets):
        ''' convert node labels to affinity mask for backprop'''
        num_sample = targets.size()[1]
        label_i = targets.unsqueeze(-1).repeat(1, 1, num_sample)
        label_j = label_i.transpose(1, 2)
        edge = torch.eq(label_i, label_j).float()
        target_edge_mask = (torch.eq(label_i, self.mask_val) + torch.eq(label_j, self.mask_val)).type(torch.bool)
        source_edge_mask = ~target_edge_mask
        init_edge = edge * source_edge_mask.float()
        return init_edge[0], source_edge_mask

    def forward(self, init_node_feat):
        #  compute normalized and not normalized affinity matrix
        edge_feat, edge_sim = self.edge_net(init_node_feat)
        # compute node features and class logits
        logits_gnn = self.node_net(init_node_feat, edge_feat)
        return logits_gnn, edge_sim


class NodeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(NodeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        self.device = device
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l - 1] if l > 0 else in_features * 2,
                out_channels=num_features_list[l],
                kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            if l < (len(num_features_list) - 1):
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        self.network = nn.Sequential(layer_list).to(device)

    def forward(self, node_feat, edge_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        # get eye matrix (batch_size x node_size x node_size) only use inter dist.
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)
        # set diagonal as zero and normalize
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        # compute attention and aggregate
        aggr_feat = torch.bmm(edge_feat.squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, aggr_feat], -1).transpose(1, 2)
        # non-linear transform
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2)
        node_feat = node_feat.squeeze(-1).squeeze(0)
        return node_feat


class EdgeNet(nn.Module):
    def __init__(self, in_features, num_features, device, ratio=(2, 1)):
        super(EdgeNet, self).__init__()
        num_features_list = [num_features * r for r in ratio]
        self.device = device
        # define layers
        layer_list = OrderedDict()
        for l in range(len(num_features_list)):
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=num_features_list[l - 1] if l > 0 else in_features,
                out_channels=num_features_list[l], kernel_size=1, bias=False
            )
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=num_features_list[l])
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()
        # add final similarity kernel
        layer_list['conv_out'] = nn.Conv2d(in_channels=num_features_list[-1],
                                           out_channels=1, kernel_size=1)
        self.sim_network = nn.Sequential(layer_list).to(device)

    def forward(self, node_feat):
        node_feat = node_feat.unsqueeze(dim=0)
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)
        # compute similarity/dissimilarity (batch_size x feat_size x num_samples x num_samples)
        sim_val = torch.sigmoid(self.sim_network(x_ij)).squeeze(1).squeeze(0).to(self.device)
        # normalize affinity matrix
        force_edge_feat = torch.eye(num_data).unsqueeze(0).repeat(num_tasks, 1, 1).to(self.device)
        edge_feat = sim_val + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1)
        return edge_feat, sim_val
