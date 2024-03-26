import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args):
    base_path = '/home/D2/home/AI_VietNam/GCN_2/GCN/data/txt/office'
    root = '/home/D/hung/code/data/office31/'
    image_set_file_labeled = os.path.join(base_path, 'labeled_images_' + args.data + '.txt')
    image_set_file_unlabeled = os.path.join(base_path, 'unlabeled_images_' + args.data + '.txt')

    print(image_set_file_labeled)
    print(image_set_file_unlabeled)

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    labeled_dataset = Imagelists_VISDA(image_set_file_labeled, root=root, transform=data_transforms['train'])
    dataset_test = Imagelists_VISDA(image_set_file_unlabeled, root=root, transform=data_transforms['test'])

    class_list = return_classlist(image_set_file_labeled)

    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24

    labeled_data_loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=bs, num_workers=3, shuffle=True, drop_last=True)
    unlabeled_data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=bs * 2, num_workers=3, shuffle=True, drop_last=True)

    return labeled_data_loader, unlabeled_data_loader_test, class_list


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
