import torch
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from networks.resnet import resnet50
from collections import OrderedDict
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import warnings
from sklearn.metrics import precision_score, recall_score


# Code for generating Heatmaps

warnings.filterwarnings('ignore')

transforms_ = transforms.Compose([
    transforms.Lambda(lambda img: custom_resize(img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_accuracy(targets, preds, batch_size):
    correct = sum(targets == preds).cpu()
    return correct / batch_size

def precision(targets, preds):
    targets = targets.detach().cpu().flatten()
    preds = preds.detach().cpu().flatten()
    return precision_score(targets, preds)

def recall(targets, preds):
    targets = targets.detach().cpu().flatten()
    preds = preds.detach().cpu().flatten()
    return recall_score(targets, preds)

def custom_resize(img):
    # interp = sample_discrete('bilinear')
    return transforms.functional.resize(img, 256, interpolation=transforms.InterpolationMode.BILINEAR)

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def data_parser(path, batch_size):
    instances = os.listdir(path)
    instances = [x for x in instances if x.split('.')[-1] in ['jpg', 'png']]

    # batch_instances = list(chunks(instances, 1 + len(instances) // batch_size))
    batch_instances = list(chunks(instances, batch_size))

    return batch_instances

def data_loader(path, img_list, im_size=(256, 256)):
    image_batch = list()
    names = list()
    for i, img_p in enumerate(img_list):
        img = Image.open(os.path.join(path, img_p))
        np_img = np.array(img) / 255.
        # original_img = cv2.imread(os.path.join(path, img_p))
        # img = cv2.resize(img, im_size)
        # img = np.float32(img) / 255.
        if i == 0:
            # input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_tensor = transforms_(img).unsqueeze(0)
        else:
            input_tensor_ = transforms_(img).unsqueeze(0)
            input_tensor = torch.cat((input_tensor, input_tensor_), dim=0)

        image_batch.append(np_img)
        names.append(img_p)
    
    return image_batch, input_tensor, names

def im_saver(save, image_batch, output_tensor, name_batch):
    for i in range(len(image_batch)):
        image_batch[i] = cv2.resize(image_batch[i], (256, 256))
        vis = show_cam_on_image(image_batch[i], output_tensor[i, :], use_rgb=True)
        cv2.imwrite(os.path.join(save, name_batch[i]), vis)

def inference(args, target_category=0):

    model = resnet50(num_classes=1)
    model = model.cuda()
    checkpoint = torch.load(args.ckpt)['model']

    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        k = k.replace('module.', '')
        new_checkpoint[k] = v
    model.load_state_dict(new_checkpoint)
    print('Loading model is complete')

    target_layers = [model.layer4[-1]]
    # input_tensor = # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    accuracy = 0
    batch_size = 0

    losses, acces, precisions, recalls = [], [], [], []

    instances = data_parser(args.path, args.batch_size)
    for batch in tqdm(instances):
        # print(f'Current Batch: {batch_idx + 1} / {len(instances)}')
        image_batch, input_tensor, names = data_loader(args.path, batch)
        input_tensor = input_tensor.cuda()
        # input_tensor = input_transform(rgb_img).cuda()
        # input_tensor = input_tensor.unsqueeze(0)
        # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        with torch.no_grad():
            preds = model(input_tensor)
            preds = preds.sigmoid()

            if args.label == 'fake':
                labels = torch.zeros(preds.size()[0], device=preds.device)
            elif args.label == 'real':
                labels = torch.ones(preds.size()[0], device=preds.device)
            else:
                raise NotImplementedError


            # _, pred = preds.max(dim=1)
            preds = preds.view(-1)
            # loss = torch.nn.functional.cross_entropy(preds, labels)
            preds[preds >= 0.5] = 1
            preds[preds < 0.5] = 0

            accu = get_accuracy(labels, preds, input_tensor.size(0))
            pre = precision(labels, preds)
            rec = recall(labels, preds)

            # losses.append(loss.item())
            acces.append(accu)
            precisions.append(pre)
            recalls.append(rec)


            # acc = preds == labels
            # acc = acc.float().sum().item()
            
            # using scikit-learn package

        # accuracy += acc
        # print(f"Current Acc: {acc / input_tensor.size(0)}")
        batch_size += input_tensor.size(0)

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        # In this example grayscale_cam has only one image in the batch:
        im_saver(args.save, image_batch, grayscale_cam, names)
    accuracy = accuracy / batch_size
    print(f'Accuracy: {accuracy}%')

    print(
        "[+] Test result\n",
        # "{:10s}: {:2.8f}\n".format('Loss', np.mean(losses)),
        "{:10s}: {:2.8f}\n".format('Accuracy', np.mean(acces)),
        "{:10s}: {:2.8f}\n".format('Precision', np.mean(precisions)),
        "{:10s}: {:2.8f}\n".format('Recall', np.mean(recalls)),
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('label', type=str, choices=['fake', 'real'])
    parser.add_argument('--save', type=str, default='./out')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--ckpt', type=str, default='checkpoints_baseline/blur_jpg_prob0.1/model_epoch_best.pth')
    parser.add_argument('--target-category', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.save, exist_ok=True)

    if args.label == 'fake':
        args.target_category = 0
    elif args.label == 'real':
        args.target_category = 1
    else:
        raise NotImplementedError

    inference(args)