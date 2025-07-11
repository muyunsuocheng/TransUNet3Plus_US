import os
import torch
import torchvision
from torchvision.io import read_image
import torch.nn as nn

from model import TransUNet3plus
from dataset import *
from trainfunc import train
from argparse import ArgumentParser
from torch.utils.data import random_split
from torchvision.transforms.functional import to_pil_image

torch.autograd.set_detect_anomaly(True)
# 检查CUDA是否可用
print("CUDA可用性:", torch.cuda.is_available())

# 创建参数解析器
parser = ArgumentParser(
    prog='TransUNET3+',
    description='Train a TransUNET3+ model on a train and test set using COCO-style labels',
)

# 参数定义 - 将training_set设为必需参数
parser.add_argument("-training_set", required=True, help="Path to directory containing the training folder with Images and Masks subdirectories")
parser.add_argument("--test_set", default=None, help="Path to test dataset. Or use --train_test_split to split training set")
parser.add_argument("--train_test_split", default=None, help="If test_set is empty, enter a number in 0-100 range to split training set")

parser.add_argument("--img_size", default=256, type=int, help="Size training images are resized to")
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--device", default=0, help="Device used for training. Defaults to cuda:0. -1 for cpu", type=int)


parser.add_argument("--save_model_directory", default='./')
parser.add_argument("--model_first_output_channels", default=16, type=int)
parser.add_argument("--model_depth", default=4, type=int)
parser.add_argument("--model_input_channels", default=3, type=int)
parser.add_argument("--model_up_feature_channels", default=32, help="Number of channels of decoder stage outputs", type=int)
parser.add_argument("--model_side_mask_size", default=256, help="Size of side mask outputs of the decoder stage", type=int)

# 解析参数
args = parser.parse_args()


# 验证训练集路径
if not os.path.exists(args.training_set):
    raise FileNotFoundError(f"训练集路径不存在: {args.training_set}")

# 设置设备
device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 构建图像和掩码路径
image_dir = os.path.join(args.training_set, 'Images')
mask_dir = os.path.join(args.training_set, 'Masks')

# 验证路径是否存在
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"图像目录不存在: {image_dir}")
if not os.path.exists(mask_dir):
    raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")

print(f"加载训练集: {args.training_set}")
print(f"图像目录: {image_dir}")
print(f"掩码目录: {mask_dir}")

# 创建数据集
dataset = COCODataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_shape=[args.img_size] * 2,
    _device=device
)

# 处理测试集
Dset_test = None
Dset_train = dataset
if args.test_set:
    print(f"加载独立测试集: {args.test_set}")
    Dset_test = COCODataset(
        image_dir=os.path.join(args.test_set, 'Images'),
        mask_dir=None,  # 不传mask_dir
        image_shape=[args.img_size] * 2,
        _device=device
    )
elif args.train_test_split:
    ratio = float(args.train_test_split) / 100.0
    if ratio <= 0 or ratio >= 1:
        raise ValueError("train_test_split 必须在 0-100 之间")
    
    train_size = int((1 - ratio) * len(dataset))
    test_size = len(dataset) - train_size
    print(f"分割数据集: 训练集={train_size}, 测试集={test_size}")
    
    Dset_train, Dset_test = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
    )
else:
    print("警告: 未提供测试集，也未指定分割比例，将使用完整数据集进行训练")

# 创建模型
model = TransUNet3plus(
    in_channels=args.model_input_channels,
    n_classes=len(dataset.categories),
    depth=args.model_depth,
    first_output_channels=args.model_first_output_channels,
    upwards_feature_channels=args.model_up_feature_channels,
    sideways_mask_shape=[args.model_side_mask_size, args.model_side_mask_size]
).to(device)

print(f"模型已创建，参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练模型
train(
    model,
    Dset_train,
    Dset_test,
    batch_size=args.batch_size,
    epochs=args.epochs,
    input_image_size=args.img_size,
    save_model_directory=args.save_model_directory
)

# 加载最优权重
best_model_path = os.path.join(args.save_model_directory, "best_model.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"已加载最优模型权重: {best_model_path}")
else:
    print("未找到最优模型权重，使用当前模型参数进行推理。")

# 推理并保存分割结果
if Dset_test is not None:
    # 构建测试集路径

    test_img_dir = os.path.join('.', 'data', 'test_set', 'Images')
    save_dir = os.path.join('.', 'data', 'test_set', 'segmentation')
    
    if os.path.exists(test_img_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"在测试集上进行推理: {test_img_dir}")
        print(f"结果保存到: {save_dir}")

        model.eval()
        with torch.no_grad():
            for img_name in os.listdir(test_img_dir):
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
                    continue

                img_path = os.path.join(test_img_dir, img_name)
                img = read_image(img_path).float().to(device)
                img = torchvision.transforms.Resize([args.img_size, args.img_size])(img)
                img = img.unsqueeze(0) / 255.0  # 归一化

                pred = model(img)
                pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().byte()

                # 先定义保存路径
                save_path = os.path.join(save_dir, os.path.splitext(img_name)[0] + '.png')

                # 保存为png
                to_pil_image(pred_mask*255).save(save_path)
                # 可选：也可以用torchvision保存
                # torchvision.utils.save_image(pred_mask.float(), save_path)
        print("推理完成!")
    else:
        print(f"警告: 测试集目录不存在: {test_img_dir}")
else:
    print("警告: 无测试集可用，跳过推理步骤")

print("训练和推理流程完成!Dumb Ass Bitch！")