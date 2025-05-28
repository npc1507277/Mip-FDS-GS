# ...原有import部分保持不变...
import gc
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# 不再需要ssim
# from utils.loss_utils import ssim
# 不再需要lpips
# from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImagesInBatches(renders_dir, gt_dir, batch_size=1):
    """按批次读取图像，减少显存占用"""
    image_names = sorted(os.listdir(renders_dir))
    return image_names  # 只返回文件名列表，不加载图像

def evaluate(model_paths, preset_image_names, scale, batch_size=1):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / f"gt_{scale}"
                renders_dir = method_dir / f"test_preds_{scale}"
                
                # 只获取文件名列表
                image_names = sorted(os.listdir(renders_dir))
                
                # 处理图像名称过滤
                if len(preset_image_names) > 0:
                    filtered_names = [name for name in image_names if name in preset_image_names]
                    image_names = filtered_names
                
                # 只保留PSNR
                psnrs = []
                
                # 批量处理图像
                for i in tqdm(range(0, len(image_names), batch_size), desc="Metric evaluation progress"):
                    batch_names = image_names[i:i+batch_size]
                    batch_renders = []
                    batch_gts = []
                    
                    # 只加载当前批次的图像
                    for fname in batch_names:
                        render = Image.open(renders_dir / fname)
                        gt = Image.open(gt_dir / fname)
                        batch_renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
                        batch_gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
                    
                    # 为当前批次计算指标，只计算PSNR
                    for j in range(len(batch_names)):
                        psnrs.append(psnr(batch_renders[j], batch_gts[j]))
                    
                    # 释放显存
                    del batch_renders, batch_gts
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # 使用正确的格式打印结果，只打印PSNR
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "PSNR": torch.tensor(psnrs).mean().item()
                })
                
                per_view_dict[scene_dir][method].update({
                    "PSNR": {name: psnr_val.item() for psnr_val, name in zip(psnrs, image_names)}
                })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 命令行参数设置
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--image_names', '-i', required=False, nargs="+", type=str, default=[])
    parser.add_argument('--resolution', '-r', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=1, 
                       help="Number of images to process at once (decrease for large images)")
    
    args = parser.parse_args()
    evaluate(args.model_paths, args.image_names, args.resolution, args.batch_size)