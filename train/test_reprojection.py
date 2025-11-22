#!/usr/bin/env python3
"""
测试相对化重投影功能

运行方式:
    python train/test_reprojection.py
"""

import torch
import math


def test_reproject_cond_to_local_frame():
    """测试条件重投影功能"""

    # 模拟 Trainer 的 _reproject_cond_to_local_frame 方法
    def _reproject_cond_to_local_frame(cond_raw, yaw_gt, yaw_pred):
        """将条件信息重投影到模型预测的局部坐标系"""
        if cond_raw is None:
            return None

        device = cond_raw.device
        dtype = cond_raw.dtype

        # 确保 yaw 是 [B] 形状
        if yaw_gt.dim() > 1:
            yaw_gt = yaw_gt.squeeze(-1)
        if yaw_pred.dim() > 1:
            yaw_pred = yaw_pred.squeeze(-1)

        # 计算朝向偏差
        delta_yaw = yaw_pred - yaw_gt
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        # 解析 cond_raw
        cond_dim = cond_raw.shape[-1]
        if cond_dim < 3:
            return cond_raw

        action_dim = cond_dim - 3
        cond_reprojected = cond_raw.clone()

        # 提取方向分量
        dir_world = cond_raw[..., action_dim:action_dim + 2]

        # 旋转到局部坐标系
        cos_delta = torch.cos(-delta_yaw)
        sin_delta = torch.sin(-delta_yaw)

        dir_local_x = dir_world[..., 0] * cos_delta - dir_world[..., 1] * sin_delta
        dir_local_y = dir_world[..., 0] * sin_delta + dir_world[..., 1] * cos_delta

        cond_reprojected[..., action_dim] = dir_local_x
        cond_reprojected[..., action_dim + 1] = dir_local_y

        return cond_reprojected

    print("=" * 60)
    print("测试 1: 基本重投影 - 模型向左偏 10°")
    print("=" * 60)

    # 测试场景 1: 模型向左偏了 10 度
    batch_size = 2
    yaw_gt = torch.tensor([0.0, 0.0])  # GT 朝向: 正北 (0°)
    yaw_pred = torch.tensor([-10.0, -10.0]) * (math.pi / 180.0)  # 模型向左偏 10°

    # 目标方向: 正前方 (1, 0)，速度 1.0
    cond_raw = torch.tensor([
        [1.0, 0.0, 1.0],  # [dir_x, dir_y, speed]
        [1.0, 0.0, 1.0]
    ])

    cond_reprojected = _reproject_cond_to_local_frame(cond_raw, yaw_gt, yaw_pred)

    print(f"GT 朝向: {yaw_gt[0].item():.2f} rad (0°)")
    print(f"模型朝向: {yaw_pred[0].item():.2f} rad (-10°)")
    print(f"原始目标方向: {cond_raw[0, :2].tolist()}")
    print(f"重投影后方向: {cond_reprojected[0, :2].tolist()}")

    # 验证: 应该向右旋转 10°
    expected_angle = 10.0 * (math.pi / 180.0)
    actual_angle = math.atan2(cond_reprojected[0, 1].item(), cond_reprojected[0, 0].item())
    print(f"期望角度: {expected_angle * 180 / math.pi:.2f}°")
    print(f"实际角度: {actual_angle * 180 / math.pi:.2f}°")

    assert abs(actual_angle - expected_angle) < 0.01, "重投影角度不正确"
    print("✓ 测试通过: 重投影角度正确\n")

    print("=" * 60)
    print("测试 2: 模型向右偏 15°")
    print("=" * 60)

    yaw_gt = torch.tensor([0.0])
    yaw_pred = torch.tensor([15.0]) * (math.pi / 180.0)  # 向右偏 15°
    cond_raw = torch.tensor([[1.0, 0.0, 1.5]])  # 目标: 正前方，速度 1.5

    cond_reprojected = _reproject_cond_to_local_frame(cond_raw, yaw_gt, yaw_pred)

    print(f"GT 朝向: {yaw_gt[0].item():.2f} rad (0°)")
    print(f"模型朝向: {yaw_pred[0].item():.2f} rad (15°)")
    print(f"原始目标方向: {cond_raw[0, :2].tolist()}")
    print(f"重投影后方向: {cond_reprojected[0, :2].tolist()}")

    # 验证: 应该向左旋转 15°
    expected_angle = -15.0 * (math.pi / 180.0)
    actual_angle = math.atan2(cond_reprojected[0, 1].item(), cond_reprojected[0, 0].item())
    print(f"期望角度: {expected_angle * 180 / math.pi:.2f}°")
    print(f"实际角度: {actual_angle * 180 / math.pi:.2f}°")

    assert abs(actual_angle - expected_angle) < 0.01, "重投影角度不正确"
    print("✓ 测试通过: 重投影角度正确\n")

    print("=" * 60)
    print("测试 3: 复杂目标方向 (45°)")
    print("=" * 60)

    yaw_gt = torch.tensor([0.0])
    yaw_pred = torch.tensor([-20.0]) * (math.pi / 180.0)  # 向左偏 20°

    # 目标方向: 45° (右前方)
    target_angle = 45.0 * (math.pi / 180.0)
    dir_x = math.cos(target_angle)
    dir_y = math.sin(target_angle)
    cond_raw = torch.tensor([[dir_x, dir_y, 2.0]])

    cond_reprojected = _reproject_cond_to_local_frame(cond_raw, yaw_gt, yaw_pred)

    print(f"GT 朝向: {yaw_gt[0].item():.2f} rad (0°)")
    print(f"模型朝向: {yaw_pred[0].item():.2f} rad (-20°)")
    print(f"原始目标方向: {cond_raw[0, :2].tolist()}")
    print(f"重投影后方向: {cond_reprojected[0, :2].tolist()}")

    # 验证: 应该是 45° + 20° = 65°
    expected_angle = 65.0 * (math.pi / 180.0)
    actual_angle = math.atan2(cond_reprojected[0, 1].item(), cond_reprojected[0, 0].item())
    print(f"期望角度: {expected_angle * 180 / math.pi:.2f}°")
    print(f"实际角度: {actual_angle * 180 / math.pi:.2f}°")

    assert abs(actual_angle - expected_angle) < 0.01, "重投影角度不正确"
    print("✓ 测试通过: 重投影角度正确\n")

    print("=" * 60)
    print("测试 4: 速度不变性")
    print("=" * 60)

    # 验证速度分量不受影响
    yaw_gt = torch.tensor([0.0])
    yaw_pred = torch.tensor([30.0]) * (math.pi / 180.0)
    cond_raw = torch.tensor([[1.0, 0.0, 3.5]])

    cond_reprojected = _reproject_cond_to_local_frame(cond_raw, yaw_gt, yaw_pred)

    print(f"原始速度: {cond_raw[0, 2].item():.2f}")
    print(f"重投影后速度: {cond_reprojected[0, 2].item():.2f}")

    assert abs(cond_raw[0, 2].item() - cond_reprojected[0, 2].item()) < 1e-6, "速度不应改变"
    print("✓ 测试通过: 速度保持不变\n")

    print("=" * 60)
    print("✅ 所有测试通过！相对化重投影功能正常工作")
    print("=" * 60)


if __name__ == "__main__":
    test_reproject_cond_to_local_frame()
