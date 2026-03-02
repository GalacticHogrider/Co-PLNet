import torch
import random
import numpy as np
import os
import os.path as osp
import time
import datetime
import argparse
import logging
import json
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import cv2

import hawp
from hawp.base.utils.comm import to_device
from hawp.base.utils.logger import setup_logger
from hawp.base.utils.metric_logger import MetricLogger
from hawp.base.utils.miscellaneous import save_config
from hawp.base.utils.checkpoint import DetectronCheckpointer
from hawp.base.utils.metric_evaluation import TPFP, AP

from hawp.fsl.dataset import build_train_dataset, build_test_dataset
from hawp.fsl.config import cfg
from hawp.fsl.config.paths_catalog import DatasetCatalog
from hawp.fsl.model.build import build_model
from hawp.fsl.solver import make_lr_scheduler, make_optimizer

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')
THRESHOLDS = [5, 10, 15]

def get_output_dir(root, basename):
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    return os.path.join(root,basename,timestamp)

def convert_coords_to_image_space(feature_coords, feature_size, image_size):
    """
    将特征图坐标转换为图像坐标 - 修正版
    
    Args:
        feature_coords: [N, 2] or [N, 4] 特征图坐标
        feature_size: (H, W) 特征图尺寸  
        image_size: (H, W) 图像尺寸 (实际tensor尺寸，不是原始标注尺寸)
    
    Returns:
        image_coords: [N, 2] or [N, 4] 图像坐标
    """
    if len(feature_coords) == 0:
        return feature_coords
    
    # 确保输入是numpy数组
    if isinstance(feature_coords, torch.Tensor):
        feature_coords = feature_coords.detach().cpu().numpy()
    
    feature_coords = np.array(feature_coords, dtype=np.float32)
        
    # 计算缩放因子
    feature_h, feature_w = feature_size
    image_h, image_w = image_size
    
    sx = image_w / feature_w  # width缩放因子
    sy = image_h / feature_h  # height缩放因子
    
    print(f"Debug coords conversion:")
    print(f"  Feature size: {feature_size} (H={feature_h}, W={feature_w})")
    print(f"  Image size: {image_size} (H={image_h}, W={image_w})")
    print(f"  Scale factors: sx={sx:.2f}, sy={sy:.2f}")
    
    # 如果尺寸相同，直接返回原坐标
    if sx == 1.0 and sy == 1.0:
        print(f"  No scaling needed, returning original coordinates")
        return feature_coords
    
    # 复制坐标进行转换
    image_coords = feature_coords.copy()
    
    if feature_coords.shape[1] == 2:  # 关键点 [x, y]
        print(f"  Converting {len(feature_coords)} points")
        if len(feature_coords) > 0:
            print(f"  First point before: ({feature_coords[0, 0]:.2f}, {feature_coords[0, 1]:.2f})")
            image_coords[:, 0] *= sx  # x坐标缩放
            image_coords[:, 1] *= sy  # y坐标缩放
            print(f"  First point after: ({image_coords[0, 0]:.2f}, {image_coords[0, 1]:.2f})")
            
    elif feature_coords.shape[1] == 4:  # 线段 [x1, y1, x2, y2]
        print(f"  Converting {len(feature_coords)} lines")
        if len(feature_coords) > 0:
            print(f"  First line before: ({feature_coords[0, 0]:.2f}, {feature_coords[0, 1]:.2f}) -> ({feature_coords[0, 2]:.2f}, {feature_coords[0, 3]:.2f})")
            image_coords[:, 0] *= sx  # x1坐标缩放
            image_coords[:, 1] *= sy  # y1坐标缩放
            image_coords[:, 2] *= sx  # x2坐标缩放
            image_coords[:, 3] *= sy  # y2坐标缩放
            print(f"  First line after: ({image_coords[0, 0]:.2f}, {image_coords[0, 1]:.2f}) -> ({image_coords[0, 2]:.2f}, {image_coords[0, 3]:.2f})")
    
    return image_coords

def visualize_debug_info(images, backbone_debug, model_output, step, output_dir, annotations):
    """
    可视化调试信息 - 修正坐标转换逻辑
    """
    if backbone_debug is None:
        return
        
    try:
        # 创建可视化目录
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 只可视化batch中的第一张图像
        batch_idx = 0
        image = images[batch_idx].detach()  # [C, H, W]
        annotation = annotations[batch_idx]
        
        # 转换图像到numpy格式 [H, W, C]
        if image.shape[0] == 1:  # 灰度图
            img_np = image[0].cpu().numpy()
            img_np = np.stack([img_np, img_np, img_np], axis=-1)  # 转为RGB
        else:  # RGB图
            img_np = image.permute(1, 2, 0).cpu().numpy()
        
        # 归一化到[0,1]
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        
        # 关键修正：使用实际tensor的图像尺寸，而不是annotation中的原始尺寸
        tensor_height, tensor_width = img_np.shape[:2]  # 实际处理的图像尺寸
        actual_image_size = (tensor_height, tensor_width)
        
        # 获取原始标注尺寸（仅用于信息显示）
        annotation_height = annotation['height']  # 原始图像高度
        annotation_width = annotation['width']    # 原始图像宽度
        annotation_size = (annotation_height, annotation_width)
        
        print(f"\nDebug visualization info:")
        print(f"  Tensor image shape: {image.shape}")
        print(f"  Numpy image shape: {img_np.shape}")
        print(f"  Actual tensor size: {actual_image_size} (used for coordinate conversion)")
        print(f"  Annotation size: {annotation_size} (original size, for reference only)")
        
        # 获取调试信息
        debug_info = backbone_debug[batch_idx]
        feature_size = debug_info['feature_size']  # (H, W)
        
        print(f"  Feature size from debug: {feature_size}")
        
        # 提取prompt数据
        prompt_lines = debug_info['lines']          # [N_lines, 4] 特征图坐标
        prompt_points = debug_info['points']        # [N_points, 2] 特征图坐标
        prompt_confidences = debug_info['point_confidences']  # [N_points]
        
        print(f"  Raw prompt data:")
        print(f"    Lines: {len(prompt_lines)} items, type: {type(prompt_lines)}")
        print(f"    Points: {len(prompt_points)} items, type: {type(prompt_points)}")
        
        # 打印一些原始坐标用于调试
        if len(prompt_lines) > 0:
            print(f"    First few lines (feature coords): {prompt_lines[:3]}")
        if len(prompt_points) > 0:
            print(f"    First few points (feature coords): {prompt_points[:3]}")
        
        # 关键修正：使用实际tensor图像尺寸进行坐标转换
        if len(prompt_lines) > 0:
            prompt_lines_img = convert_coords_to_image_space(prompt_lines, feature_size, actual_image_size)
        else:
            prompt_lines_img = np.array([]).reshape(0, 4)
            
        if len(prompt_points) > 0:
            prompt_points_img = convert_coords_to_image_space(prompt_points, feature_size, actual_image_size)
            # 确保confidences是numpy数组
            if isinstance(prompt_confidences, torch.Tensor):
                prompt_confidences = prompt_confidences.detach().cpu().numpy()
        else:
            prompt_points_img = np.array([]).reshape(0, 2)
            prompt_confidences = np.array([])
        
        # 创建图像 (训练模式下只显示prompt结果)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Training Step {step} - Prompt Visualization\n'
                    f'Feature: {feature_size}, Tensor: {actual_image_size}, Original: {annotation_size}', 
                    fontsize=14)
        
        # === 子图1: 原始图像 ===
        ax1 = axes[0]
        ax1.imshow(img_np)
        ax1.set_title('Original Image', fontsize=12)
        ax1.axis('off')
        
        # === 子图2: 原始图像 + Prompt解码结果 ===
        ax2 = axes[1]
        ax2.imshow(img_np)
        ax2.set_title(f'Prompt Results\nLines: {len(prompt_lines_img)}, Points: {len(prompt_points_img)}', fontsize=12)
        ax2.axis('off')
        
        # 设置坐标轴范围确保完整显示（使用实际tensor尺寸）
        ax2.set_xlim(0, tensor_width)
        ax2.set_ylim(tensor_height, 0)  # 注意Y轴是倒置的
        
        # 可视化prompt线段
        if len(prompt_lines_img) > 0:
            # 限制显示数量避免过于拥挤
            max_lines_to_show = min(100, len(prompt_lines_img))
            for i in range(max_lines_to_show):
                line = prompt_lines_img[i]
                x1, y1, x2, y2 = line
                
                # 检查坐标是否在合理范围内（使用实际tensor尺寸）
                if (0 <= x1 <= tensor_width and 0 <= y1 <= tensor_height and 
                    0 <= x2 <= tensor_width and 0 <= y2 <= tensor_height):
                    ax2.plot([x1, x2], [y1, y2], 'r-', linewidth=1.5, alpha=0.7)
                else:
                    print(f"    Warning: Line {i} out of bounds: ({x1:.1f},{y1:.1f}) -> ({x2:.1f},{y2:.1f})")
        
        # 可视化prompt关键点
        if len(prompt_points_img) > 0:
            max_points_to_show = min(200, len(prompt_points_img))
            for i in range(max_points_to_show):
                point = prompt_points_img[i]
                conf = prompt_confidences[i]
                x, y = point
                
                # 检查坐标是否在合理范围内（使用实际tensor尺寸）
                if 0 <= x <= tensor_width and 0 <= y <= tensor_height:
                    size = 20 + conf * 80  # 根据置信度调整大小
                    ax2.scatter(x, y, s=size, c='blue', alpha=0.8, marker='o')
                else:
                    print(f"    Warning: Point {i} out of bounds: ({x:.1f},{y:.1f})")
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label=f'Prompt Lines ({len(prompt_lines_img)})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=8, label=f'Prompt Points ({len(prompt_points_img)})')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # 保存图像
        plt.tight_layout()
        filename = f'train_visualization_step_{step:06d}.png'
        filepath = os.path.join(vis_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Training visualization saved: {filepath}")
        print(f"  Feature map size: {feature_size}")
        print(f"  Tensor image size: {actual_image_size} (used for conversion)")
        print(f"  Original annotation size: {annotation_size}")
        print(f"  Scale factors: sx={actual_image_size[1]/feature_size[1]:.2f}, sy={actual_image_size[0]/feature_size[0]:.2f}")
        print(f"  Prompt: {len(prompt_lines_img)} lines, {len(prompt_points_img)} points")
        
    except Exception as e:
        print(f"Visualization error at step {step}: {str(e)}")
        import traceback
        traceback.print_exc()

def compute_sap(result_list, annotations_dict, threshold):
    tp_list, fp_list, scores_list = [],[],[]
    n_gt = 0
    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        lines_pred = np.array(res['lines_pred'],dtype=np.float32)
        scores = np.array(res['lines_score'],dtype=np.float32)
        sort_idx = np.argsort(-scores)
        
        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]
        # import pdb; pdb.set_trace()
        lines_pred[:,0] *= 128/float(res['width'])
        lines_pred[:,1] *= 128/float(res['height'])
        lines_pred[:,2] *= 128/float(res['width'])
        lines_pred[:,3] *= 128/float(res['height'])

        lines_gt   = np.array(gt['lines'],dtype=np.float32)
        lines_gt[:,0]  *= 128/float(gt['width'])
        lines_gt[:,1]  *= 128/float(gt['height'])
        lines_gt[:,2]  *= 128/float(gt['width'])
        lines_gt[:,3]  *= 128/float(gt['height'])

        tp, fp = TPFP(lines_pred,lines_gt,threshold)
        
        n_gt += lines_gt.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx])/n_gt
    fp = np.cumsum(fp_list[idx])/n_gt
    rcs = tp
    pcs = tp/np.maximum(tp+fp,1e-9)
    sAP = AP(tp,fp)*100
    return sAP, pcs, rcs

def sAPEval(result_list, annotations_dict, threshold):
    tp_list, fp_list, scores_list = [],[],[]
    n_gt = 0
    for res in result_list:
        filename = res['filename']
        gt = annotations_dict[filename]
        lines_pred = np.array(res['lines_pred'],dtype=np.float32)
        scores = np.array(res['lines_score'],dtype=np.float32)
        sort_idx = np.argsort(-scores)
        
        lines_pred = lines_pred[sort_idx]
        scores = scores[sort_idx]
        lines_pred[:,0] *= 128/float(res['width'])
        lines_pred[:,1] *= 128/float(res['height'])
        lines_pred[:,2] *= 128/float(res['width'])
        lines_pred[:,3] *= 128/float(res['height'])

        lines_gt   = np.array(gt['lines'],dtype=np.float32)
        lines_gt[:,0]  *= 128/float(gt['width'])
        lines_gt[:,1]  *= 128/float(gt['height'])
        lines_gt[:,2]  *= 128/float(gt['width'])
        lines_gt[:,3]  *= 128/float(gt['height'])
        
        assert gt['width'] == res['width'] and gt['height'] == res['height']
        
        tp, fp = TPFP(lines_pred,lines_gt,threshold)
        n_gt += lines_gt.shape[0]
        tp_list.append(tp)
        fp_list.append(fp)
        scores_list.append(scores)

    tp_list = np.concatenate(tp_list)
    fp_list = np.concatenate(fp_list)
    scores_list = np.concatenate(scores_list)
    idx = np.argsort(scores_list)[::-1]
    tp = np.cumsum(tp_list[idx])/n_gt
    fp = np.cumsum(fp_list[idx])/n_gt
    rcs = tp
    pcs = tp/np.maximum(tp+fp,1e-9)
    F_list = (2*rcs*pcs/(rcs+pcs+1e-9))
    F_list = np.nan_to_num(F_list, 0)
    F = F_list.max()
    
    P = pcs[F_list.argmax()]
    R = rcs[F_list.argmax()]
    sAP = AP(tp,fp)

    return sAP, P, R, F

def validate(cfg, model, val_datasets, epoch, writer, logger):
    """验证函数，参考benchmark.py的逻辑"""
    model.eval()
    
    val_metrics = {}
    
    for name, dataset in val_datasets:
        logger.info(f'Validating on {name} dataset')
        results = []
        
        data_list = []
        for i, (images, annotations) in enumerate(dataset):
            data_list.append((images.to(cfg.MODEL.DEVICE), annotations))
            
        for i, (images, annotations) in enumerate(tqdm(data_list, desc=f'Val {name}', ncols=80)):
            with torch.no_grad():
                output, extra_info = model(images, annotations=annotations)
            output = to_device(output,'cpu')
        
            for k in output.keys():
                if isinstance(output[k], torch.Tensor):
                    output[k] = output[k].tolist()
            results.append(output)

        # 评估结果
        ann_file = DatasetCatalog.get(name)['args']['ann_file']
        with open(ann_file,'r') as _ann:
            annotations_list = json.load(_ann)
        annotations_dict = {
            ann['filename']: ann for ann in annotations_list
        }
        
        dataset_metrics = {}
        for threshold in THRESHOLDS:
            sAP, P, R, F = sAPEval(results, annotations_dict, threshold)
            dataset_metrics[f'sAP{threshold}'] = sAP * 100
            dataset_metrics[f'sF{threshold}'] = F * 100
            
            logger.info(f'{name} - sAP{threshold} = {sAP*100:.1f}')
            logger.info(f'{name} - sF{threshold} = {F*100:.1f}')
            
            # 记录到tensorboard (仅当writer不为None时)
            if writer is not None:
                writer.add_scalar(f'val_{name}/sAP{threshold}', sAP*100, epoch)
                writer.add_scalar(f'val_{name}/sF{threshold}', F*100, epoch)
        
        val_metrics[name] = dataset_metrics
    
    return val_metrics

class LossReducer(object):
    def __init__(self,cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
    
    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k]*loss_dict[k] 
        for k in self.loss_weights.keys()])
        
        return total_loss

def train(cfg, model, train_dataset, val_datasets, optimizer, scheduler, loss_reducer, checkpointer, arguments):
    logger = logging.getLogger("hawp.trainer")
    device = cfg.MODEL.DEVICE
    model = model.to(device)
    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    num_epochs = arguments['max_epoch'] - start_epoch
    epoch_size = len(train_dataset)
    
    epoch = arguments['epoch'] +1

    total_iterations = num_epochs*epoch_size
    step = 0

    writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'tensorboard'))
    
    # 用于记录最佳性能
    best_sap = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch+1, start_epoch+num_epochs+1):
        model.train()            
        loss_meters = MetricLogger(" ")
        aux_meters = MetricLogger(" ")
        sys_meters = MetricLogger(" ")
        
        # 创建epoch级别的进度条
        epoch_pbar = tqdm(
            enumerate(train_dataset),
            total=len(train_dataset),
            desc=f'Epoch {epoch}/{start_epoch+num_epochs}',
            unit='batch',
            ncols=120,
            leave=True
        )
        
        for it, (images, annotations) in epoch_pbar:
            data_time = time.time() - end
            images = images.to(device)
            annotations = to_device(annotations,device)
            
            # 确定可视化频率
            if epoch == 1:  # 第一个epoch
                vis_interval = 10
            else:  # 其他epoch
                vis_interval = 100
            
            # 可视化决策
            should_visualize = (it % vis_interval == 0)
            
            # 正常训练，如果需要可视化则获取debug信息
            if should_visualize:
                loss_dict, extra_info, debug_info = model(images, annotations, return_debug=True)
                
                # 提取backbone debug信息进行可视化
                backbone_debug = debug_info.get('backbone_debug', None)
                if backbone_debug is not None:
                    try:
                        # 在训练模式下可视化，不需要切换模式
                        # 由于我们在训练模式，没有最终检测结果，所以创建一个模拟的输出用于可视化框架
                        mock_output = {
                            'lines_pred': [],  # 训练模式下没有最终检测结果
                            'lines_score': [],
                            'juncs_pred': [],
                            'juncs_score': [],
                        }
                        visualize_debug_info(images, backbone_debug, mock_output, step, cfg.OUTPUT_DIR, annotations)
                    except Exception as e:
                        logger.warning(f"Visualization failed at step {step}: {str(e)}")
            else:
                # 正常训练，不获取debug信息
                loss_dict, extra_info = model(images, annotations)
            
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = total_loss.item()
                loss_meters.update(loss=loss_reduced, **loss_dict_reduced)
                aux_meters.update(**extra_info)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            sys_meters.update(time=batch_time, data=data_time)

            total_iterations -= 1
            step +=1
            
            eta_seconds = sys_meters.time.global_avg*total_iterations
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            # 更新进度条显示信息
            progress_info = {
                'Loss': f'{loss_reduced:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'ETA': eta_string.split('.')[0],  # 去掉毫秒部分
                'GPU': f'{torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}MB'
            }
            
            if should_visualize:
                progress_info['VIS'] = '✓'
                
            epoch_pbar.set_postfix(progress_info)

            if it % 20 == 0 or it+1 == len(train_dataset):
                logger.info(
                    "".join(
                        [
                            "eta: {eta} ",
                            "epoch: {epoch} ",
                            "iter: {iter} ",
                            "lr: {lr:.6f} ",
                            "max mem: {memory:.0f}\n",
                            "RUNTIME: {sys_meters}\n",
                            "LOSSES: {loss_meters}\n",
                            "AUXINFO: {aux_meters}\n"
                            "WorkingDIR: {wdir}\n"
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        loss_meters=str(loss_meters),
                        sys_meters=str(sys_meters),
                        aux_meters=str(aux_meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        wdir = cfg.OUTPUT_DIR
                    )
                )

            writer.add_scalar("loss", loss_reduced, step)

        # 关闭当前epoch的进度条
        epoch_pbar.close()
        
        # 打印epoch总结
        logger.info(f'Epoch {epoch} completed - Average Loss: {loss_meters.loss.global_avg:.4f}')
        
        # 验证阶段
        if val_datasets is not None:
            val_metrics = validate(cfg, model, val_datasets, epoch, writer, logger)
            
            # 使用york数据集的sAP10作为主要指标（如果存在）
            # 如果不存在york，则使用wireframe作为后备
            if 'york_test' in val_metrics:
                current_sap = val_metrics['york_test']['sAP10']
                dataset_name = 'york_test'
            elif 'wireframe_test' in val_metrics:
                current_sap = val_metrics['wireframe_test']['sAP10']
                dataset_name = 'wireframe_test'
            else:
                # 如果都没有，使用第一个可用的数据集
                dataset_name = list(val_metrics.keys())[0]
                current_sap = val_metrics[dataset_name]['sAP10']
            
            if current_sap > best_sap:
                best_sap = current_sap
                best_epoch = epoch
                # 保存最佳模型
                checkpointer.save('model_best')
                logger.info(f'New best model saved! {dataset_name} sAP10: {best_sap:.1f}')
            
            logger.info(f'Current best: Epoch {best_epoch}, {dataset_name} sAP10: {best_sap:.1f}')
        
        scheduler.step()
        checkpointer.save('model_{:05d}'.format(epoch))
                
    writer.close()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (num_epochs)
        )
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='HAWPv2 Training')

    parser.add_argument("config",
                        # metavar="FILE",
                        help="path to config file",
                        type=str,
                        )

    parser.add_argument('--logdir',required=True, type=str)
    parser.add_argument('--resume',default=None, type=str)
    parser.add_argument("--clean",
                        default=False,
                        action='store_true')
    parser.add_argument("--seed",
                        default=42,
                        type=int)
    parser.add_argument('--val-dataset', 
                        default='wireframe', 
                        choices=['wireframe', 'york', 'both'],
                        help='validation dataset')
    
    parser.add_argument('--tf32', default=False, action='store_true', help='toggle on the TF32 of pytorch')
    parser.add_argument('--dtm', default=True, choices=[True, False], help='toggle the deterministic option of CUDNN. This option will affect the replication of experiments')

    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = args.dtm

    assert args.config.endswith('yaml') or args.config.endswith('yml')
    config_basename = os.path.basename(args.config)
    if config_basename.endswith('yaml'):
        config_basename = config_basename[:-5]
    else:
        config_basename = config_basename[:-4]

    cfg.merge_from_file(args.config)

    output_dir = get_output_dir(args.logdir,config_basename)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir)
    

    logger = setup_logger('hawp', output_dir, out_file='train.log')

    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config))

    with open(args.config,"r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(output_dir, 'config.yaml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = build_model(cfg)

    device = cfg.MODEL.DEVICE
    model = model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    loss_reducer = LossReducer(cfg)

    arguments = {}
    arguments["epoch"] = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arguments["max_epoch"] = max_epoch

    checkpointer = DetectronCheckpointer(cfg,
                                         model,
                                         optimizer,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)
    if args.resume:
        state_dict = torch.load(args.resume,map_location='cpu')
        model.load_state_dict(state_dict['model'],strict=False)
        logger.info('loading the pretrained model from {}'.format(args.resume))
        
    train_dataset = build_train_dataset(cfg)
    logger.info('epoch size = {}'.format(len(train_dataset)))
    
    # 构建验证数据集 - 修改为使用york数据集
    val_datasets = None
    if args.val_dataset != 'none':
        if args.val_dataset == 'wireframe':
            cfg.DATASETS.TEST = ('wireframe_test',)
        elif args.val_dataset == 'york':
            cfg.DATASETS.TEST = ('york_test',)
        elif args.val_dataset == 'both':
            cfg.DATASETS.TEST = ('wireframe_test', 'york_test')
        
        val_datasets = build_test_dataset(cfg)
        logger.info('Validation datasets: {}'.format([name for name, _ in val_datasets]))
        
        # 在训练开始前先进行一轮验证，检查一切是否正常工作
        logger.info('Performing initial validation before training starts...')
        initial_val_metrics = validate(cfg, model, val_datasets, 0, None, logger)
        logger.info('Initial validation completed successfully!')
    
    train(cfg, model, train_dataset, val_datasets, optimizer, scheduler, loss_reducer, checkpointer, arguments)    

                                         
    
    import pdb; pdb.set_trace()