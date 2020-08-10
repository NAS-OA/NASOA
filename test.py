# -*- coding:utf-8 -*-
"""Test code of ETNas."""
import argparse
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from etnas import ETNas, MODEL_MAPPINGS


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet test code')
    parser.add_argument('--data-dir', default='imagenet', 
                        help='Location of ImageNet, which has subfolders train and val')
    parser.add_argument('--model-name', default='ET-NAS-A',
                        help='Name of the model to run, default: ET-NAS-A')
    parser.add_argument('--model-zoo-dir', default="./model_zoo",
                        help='Path of the save model')
    parser.add_argument('--batch-size', default=256, type=int, 
                        help='Batch size of dataloader')
    parser.add_argument('--num-workers', default=4, type=int, 
                        help='Number of workers for dataloader')
    parser.add_argument('--device', default='cuda', 
                        help='Device used ("cuda" or "cpu")')
    return parser.parse_args()


def main():
    args = parse_args()
    val_dir = os.path.join(args.data_dir, 'val')
    transform = transforms.Compose([
        transforms.Resize(256, 3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    dataset = ImageFolder(val_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)    
    model = ETNas(MODEL_MAPPINGS[args.model_name]).to(args.device).eval()
    saved_state_dict = torch.load(os.path.join(args.model_zoo_dir, args.model_name, "{}.pth".format(MODEL_MAPPINGS[args.model_name])))
    model.load_state_dict(saved_state_dict)

    total_correct, total_images, total_steps = 0, 0, len(dataloader)
    for step, (images, labels) in enumerate(dataloader):
        images, labels = images.to(args.device), labels.to(args.device)
        logits = model(images)
        preds = torch.argmax(logits, 1)
        total_correct += torch.sum(torch.eq(preds, labels).long()).item()
        total_images += images.shape[0]

        if (step + 1) % 100 == 0:
            accuracy = total_correct / total_images
            print("Step [{}/{}]: [{}/{}] correct, accuracy = {:.2f}%".format(
                step + 1, total_steps, total_correct, total_images, accuracy * 100))

    accuracy = total_correct / total_images
    print("Final: [{}/{}] correct, accuracy = {:.2f}%".format(
        total_correct, total_images, accuracy * 100))


if __name__ == "__main__":
    main()

