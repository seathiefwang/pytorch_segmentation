import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict

matches = [800, 100, 200, 300, 400, 500, 600, 700]


def multi_plus_predict(models, image, num_classes, device):
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))
    image = image.to(device)
    for model in models:
        prediction = model(image)
        total_predictions += prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(models)+1
    return total_predictions

def multi_vote_predict(models, image, num_classes, device):
    total_predictions = []
    image = image.to(device)
    for model in models:
        prediction = model(image)
        prediction = F.softmax(prediction.squeeze(0), dim=0).argmax(0).cpu().numpy()
        total_predictions.append(prediction)
    
    total_predictions = np.asarray(total_predictions)
    results = []
    for key in range(num_classes):
        mask = total_predictions == key
        val = mask.sum(axis=0)
        results.append(val)
    results = np.asarray(results).argmax(0)
    return results


def save_images(image, mask, output_path, image_file, num_classes):
	# Saves the image, the model output and the results after the post processing
    # w, h = image.size
    # image_file = os.path.basename(image_file).split('.')[0]
    # colorized_mask = colorize_mask(mask, palette)
    # colorized_mask.save(os.path.join(output_path, image_file+'.png'))
    # output_im = Image.new('RGB', (w*2, h))
    # output_im.paste(image, (0,0))
    # output_im.paste(colorized_mask, (w,0))
    # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
    # mask_img = Image.fromarray(mask, 'L')
    # mask_img.save(os.path.join(output_path, image_file+'.png'))

    h, w = mask.shape
    save_mask = np.zeros((w, h), dtype=np.int32)
    for i in range(num_classes):
        save_mask[mask == i] = matches[i]
    save_mask = Image.fromarray(save_mask)
    image_file = os.path.basename(image_file).split('.')[0]
    save_mask.save(os.path.join(output_path, image_file+'.png'))

def load_model(model_name, model_path, num_classes, device):

    # Model
    model = getattr(models, model_name)(num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def main():
    args = parse_arguments()
    config = json.load(open(args.config))
    config2 = json.load(open(args.config2))


    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'RSI']
    if dataset_type == 'CityScapes': 
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
    else:
        scales = [0.75, 1.0, 1.25, 1.5]
    loader = getattr(dataloaders, config['train_loader']['type'])(**config['train_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    # palette = loader.dataset.palette
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    models = []
    for model_path in args.models:
        model = load_model(config['arch']['type'], model_path, num_classes, device)
        models.append(model)

    for model_path in args.models2:
        model = load_model(config2['arch']['type'], model_path, num_classes, device)
        models.append(model)
    print("model num:", len(models))

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
    with torch.no_grad():
        tbar = tqdm(image_files, ncols=100)
        for img_file in tbar:
            image = Image.open(img_file).convert('RGB')
            input = normalize(to_tensor(image)).unsqueeze(0)
            
            if args.mode == 'vote':
                prediction = multi_vote_predict(models, input, num_classes, device)
            else:
                prediction = model(input.to(device))
                prediction = prediction.squeeze(0).cpu().numpy()
                prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
            save_images(image, prediction, args.output, img_file, num_classes)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-c2', '--config2', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='vote', type=str,
                        help='Mode used for prediction: either [multiflip, multiscale, sliding]')
    parser.add_argument('-m', '--models', default='model_weights.pth', nargs='+',
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-m2', '--models2', default='model_weights.pth', nargs='+',
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='tif', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
