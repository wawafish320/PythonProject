#!/usr/bin/env python3
"""
诊断 contact_plan GRU 的长期稳定性：
固定 cond 输入，跑 1000+ 步，观察 contacts_plan 是否：
1. 周期稳定（好）
2. 塌缩到常数（坏 - 信息丢失）
3. 发散/振荡（坏 - 动力学不稳定）
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train.models import EventMotionModel


def diagnose_plan_stability(
    model: EventMotionModel,
    cond_fixed: torch.Tensor,  # [1, cond_dim] - 固定的控制输入
    steps: int = 1000,
    device: str = 'cpu',
):
    """
    固定 cond，纯 GRU rollout，观察 plan_z 和 contacts_plan 的演化。
    """
    model.eval()
    model.to(device)
    cond_fixed = cond_fixed.to(device)

    if not model.contact_plan_enable:
        print("模型未启用 contact_plan，无法诊断")
        return None

    h_plan = int(model.contact_plan_hidden)
    contact_dim = int(model.contact_dim)

    # 初始化
    plan_z = torch.zeros((1, h_plan), device=device, dtype=torch.float32)

    # 记录轨迹
    plan_z_history = []
    contacts_plan_history = []

    with torch.no_grad():
        for t in range(steps):
            # GRU step
            plan_z = model.contact_plan_cell(cond_fixed, plan_z)

            # Decode to contacts
            logits = model.contact_plan_head(plan_z)
            contacts_plan = torch.sigmoid(logits)

            # 记录
            plan_z_history.append(plan_z.cpu().numpy()[0])  # [h_plan]
            contacts_plan_history.append(contacts_plan.cpu().numpy()[0])  # [contact_dim]

    plan_z_history = np.array(plan_z_history)  # [T, h_plan]
    contacts_plan_history = np.array(contacts_plan_history)  # [T, contact_dim]

    # 分析
    analysis = analyze_stability(plan_z_history, contacts_plan_history, steps)

    return {
        'plan_z': plan_z_history,
        'contacts_plan': contacts_plan_history,
        'analysis': analysis,
    }


def analyze_stability(plan_z_seq, contacts_seq, steps):
    """
    分析长期稳定性的指标。
    """
    T, h_plan = plan_z_seq.shape
    T, contact_dim = contacts_seq.shape

    # 1. Hidden state 范数变化
    z_norms = np.linalg.norm(plan_z_seq, axis=1)
    z_norm_mean = z_norms.mean()
    z_norm_std = z_norms.std()
    z_norm_trend = np.polyfit(np.arange(T), z_norms, deg=1)[0]  # 线性趋势

    # 2. Contacts 的统计
    c_mean = contacts_seq.mean(axis=0)  # [contact_dim]
    c_std = contacts_seq.std(axis=0)
    c_range = contacts_seq.max(axis=0) - contacts_seq.min(axis=0)

    # 3. 检测周期性（用自相关）
    autocorr_lags = []
    for dim in range(contact_dim):
        signal = contacts_seq[:, dim]
        # 简单自相关（去均值）
        signal_centered = signal - signal.mean()
        autocorr = np.correlate(signal_centered, signal_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # 只看正lag
        autocorr = autocorr / autocorr[0]  # 归一化

        # 找第一个大于0.5的非零lag（可能是周期）
        peaks = []
        for lag in range(10, min(200, len(autocorr))):
            if autocorr[lag] > 0.5:
                peaks.append(lag)

        if peaks:
            autocorr_lags.append(peaks[0])
        else:
            autocorr_lags.append(-1)  # 无明显周期

    # 4. 检测塌缩（方差过小）
    collapsed = (c_std < 0.05).sum()

    # 5. 检测发散（范数增长）
    diverging = z_norm_trend > 0.01  # 每步增长 > 0.01

    analysis = {
        'hidden_norm_mean': float(z_norm_mean),
        'hidden_norm_std': float(z_norm_std),
        'hidden_norm_trend': float(z_norm_trend),
        'contacts_mean': c_mean.tolist(),
        'contacts_std': c_std.tolist(),
        'contacts_range': c_range.tolist(),
        'autocorr_periods': autocorr_lags,
        'collapsed_dims': int(collapsed),
        'diverging': bool(diverging),
    }

    # 判断稳定性
    if diverging:
        stability = 'DIVERGING (坏)'
    elif collapsed > contact_dim // 2:
        stability = 'COLLAPSED (坏)'
    elif max([p for p in autocorr_lags if p > 0], default=0) > 0:
        stability = f'PERIODIC (好 - 周期约{max(autocorr_lags)}帧)'
    elif z_norm_std / z_norm_mean < 0.1:
        stability = 'STABLE_ATTRACTOR (可能好)'
    else:
        stability = 'UNCERTAIN (需人工检查曲线)'

    analysis['stability_verdict'] = stability

    return analysis


def plot_results(result, out_path):
    """
    绘制诊断结果。
    """
    plan_z = result['plan_z']
    contacts = result['contacts_plan']
    analysis = result['analysis']

    T, h_plan = plan_z.shape
    T, contact_dim = contacts.shape

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 1. Hidden state 范数
    ax = axes[0]
    z_norms = np.linalg.norm(plan_z, axis=1)
    ax.plot(z_norms, label='||plan_z||')
    ax.set_title(f"Hidden State Norm (trend={analysis['hidden_norm_trend']:.4f})")
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 Norm')
    ax.legend()
    ax.grid(True)

    # 2. Contacts 演化
    ax = axes[1]
    for dim in range(min(4, contact_dim)):  # 最多画4个维度
        ax.plot(contacts[:, dim], label=f'contact_{dim}', alpha=0.7)
    ax.set_title('Contacts Plan Evolution')
    ax.set_xlabel('Step')
    ax.set_ylabel('Contact Probability')
    ax.legend()
    ax.grid(True)

    # 3. Hidden state 前3个维度
    ax = axes[2]
    for dim in range(min(3, h_plan)):
        ax.plot(plan_z[:, dim], label=f'z_{dim}', alpha=0.7)
    ax.set_title('Hidden State (first 3 dims)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"诊断图保存到: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--cond-json', type=str, help='固定cond的JSON文件（可选，默认用全0）')
    parser.add_argument('--steps', type=int, default=1000, help='rollout步数')
    parser.add_argument('--out', type=str, default='debug_output/plan_stability.json', help='输出JSON')
    parser.add_argument('--plot', type=str, default='debug_output/plan_stability.png', help='输出图像')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.model}")
    ckpt = torch.load(args.model, map_location='cpu')
    model_state = ckpt.get('model_state_dict', ckpt)

    # 从checkpoint推断模型配置（简化版，实际需要config）
    # 这里假设你有保存config在checkpoint中
    cfg = ckpt.get('config', {})
    if not cfg:
        print("警告：checkpoint中无config，使用默认配置")
        cfg = {
            'in_state_dim': 200,
            'cond_dim': 5,
            'out_motion_dim': 150,
            'contact_plan_enable': True,
            'contact_plan_hidden': 64,
        }

    model = EventMotionModel(**cfg)
    model.load_state_dict(model_state, strict=False)
    model.to(args.device)

    # 构造固定cond
    cond_dim = cfg.get('cond_dim', 5)
    if args.cond_json:
        with open(args.cond_json) as f:
            cond_data = json.load(f)
        cond_fixed = torch.tensor(cond_data, dtype=torch.float32).unsqueeze(0)
    else:
        # 默认：walk forward at medium speed
        # 假设 cond = [action_onehot(3), dir_x, dir_y, speed]
        # walk=1, forward=(0,1), speed=0.8
        cond_fixed = torch.tensor([[0, 1, 0, 0, 1, 0.8]], dtype=torch.float32)

    print(f"使用 cond: {cond_fixed.numpy()}")

    # 诊断
    print(f"开始 {args.steps} 步 rollout...")
    result = diagnose_plan_stability(model, cond_fixed, args.steps, args.device)

    if result is None:
        return

    # 输出分析
    analysis = result['analysis']
    print("\n=== 稳定性分析 ===")
    print(f"Hidden Norm 均值: {analysis['hidden_norm_mean']:.4f}")
    print(f"Hidden Norm 趋势: {analysis['hidden_norm_trend']:.6f} (>0.01为发散)")
    print(f"Contacts 塌缩维度: {analysis['collapsed_dims']}/{len(analysis['contacts_mean'])}")
    print(f"检测到的周期: {analysis['autocorr_periods']}")
    print(f"\n结论: {analysis['stability_verdict']}")

    # 保存结果
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'analysis': analysis,
            'config': {
                'steps': args.steps,
                'cond': cond_fixed.tolist(),
            }
        }, f, indent=2)
    print(f"\n结果保存到: {out_path}")

    # 绘图
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(result, plot_path)


if __name__ == '__main__':
    main()
