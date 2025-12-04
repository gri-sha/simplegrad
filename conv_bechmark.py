import numpy as np
import time
import simplegrad as sg
import torch
import torch.nn as nn


def benchmark_conv(batch_size, in_channels, img_size, out_channels, kernel_size, n_runs=50):
    """Benchmark a single convolution configuration."""
    # Create input tensor
    x = sg.random((batch_size, in_channels, img_size, img_size), dtype="float32")

    # Create conv layer
    conv = sg.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        pad_width=1,
    )

    # PyTorch setup
    x_torch = torch.tensor(x.values, requires_grad=True)
    conv_torch = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    # Warmup simplegrad
    for _ in range(5):
        y = conv(x)
        y.zero_grad()
        y.backward()

    # Warmup pytorch
    for _ in range(5):
        y_torch = conv_torch(x_torch)
        y_torch.sum().backward()
        conv_torch.zero_grad()
        x_torch.grad = None

    # Benchmark simplegrad forward
    sg_forward_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y = conv(x)
        end = time.perf_counter()
        sg_forward_times.append(end - start)

    # Benchmark simplegrad backward
    sg_backward_times = []
    for _ in range(n_runs):
        y = conv(x)
        y.zero_grad()
        start = time.perf_counter()
        y.backward()
        end = time.perf_counter()
        sg_backward_times.append(end - start)

    # Benchmark pytorch forward
    torch_forward_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y_torch = conv_torch(x_torch)
        end = time.perf_counter()
        torch_forward_times.append(end - start)

    # Benchmark pytorch backward
    torch_backward_times = []
    for _ in range(n_runs):
        y_torch = conv_torch(x_torch)
        start = time.perf_counter()
        y_torch.sum().backward()
        end = time.perf_counter()
        conv_torch.zero_grad()
        x_torch.grad = None
        torch_backward_times.append(end - start)

    sg_total = (np.mean(sg_forward_times) + np.mean(sg_backward_times)) * 1000
    sg_total_std = np.sqrt(np.var(sg_forward_times) + np.var(sg_backward_times)) * 1000
    torch_total = (np.mean(torch_forward_times) + np.mean(torch_backward_times)) * 1000
    torch_total_std = np.sqrt(np.var(torch_forward_times) + np.var(torch_backward_times)) * 1000

    delta = sg_total - torch_total
    delta_std = np.sqrt(sg_total_std**2 + torch_total_std**2)

    return {
        "sg_fwd": np.mean(sg_forward_times) * 1000,
        "sg_fwd_std": np.std(sg_forward_times) * 1000,
        "sg_bwd": np.mean(sg_backward_times) * 1000,
        "sg_bwd_std": np.std(sg_backward_times) * 1000,
        "sg_total": sg_total,
        "sg_total_std": sg_total_std,
        "torch_fwd": np.mean(torch_forward_times) * 1000,
        "torch_fwd_std": np.std(torch_forward_times) * 1000,
        "torch_bwd": np.mean(torch_backward_times) * 1000,
        "torch_bwd_std": np.std(torch_backward_times) * 1000,
        "torch_total": torch_total,
        "torch_total_std": torch_total_std,
        "delta": delta,
        "delta_std": delta_std,
        "output_shape": y.shape,
    }


def run_benchmarks():
    # MNIST-like configuration
    img_size = 28
    in_channels = 1
    out_channels = 32

    kernel_sizes = [2, 3, 5, 7]
    batch_sizes = [1, 32, 128]

    results = {}

    for batch_size in batch_sizes:
        print(f"\n{'=' * 140}")
        print(f"Input Shape: ({batch_size}, {in_channels}, {img_size}, {img_size})")
        print("=" * 140)

        print(
            f"\n{'Kernel':<8} {'Shape':<20} {'SG Fwd (ms)':<15} {'PT Fwd (ms)':<15} {'SG Bwd (ms)':<15} {'PT Bwd (ms)':<15} {'SG Total':<15} {'PT Total':<15} {'Delta':<15}"
        )
        print("-" * 140)

        for kernel_size in kernel_sizes:
            result = benchmark_conv(
                batch_size=batch_size,
                in_channels=in_channels,
                img_size=img_size,
                out_channels=out_channels,
                kernel_size=kernel_size,
                n_runs=30,
            )

            key = (batch_size, kernel_size)
            results[key] = result

            kernel_str = f"{kernel_size}x{kernel_size}"
            sg_fwd_str = f"{result['sg_fwd']:.2f}±{result['sg_fwd_std']:.2f}"
            pt_fwd_str = f"{result['torch_fwd']:.2f}±{result['torch_fwd_std']:.2f}"
            sg_bwd_str = f"{result['sg_bwd']:.2f}±{result['sg_bwd_std']:.2f}"
            pt_bwd_str = f"{result['torch_bwd']:.2f}±{result['torch_bwd_std']:.2f}"
            sg_tot_str = f"{result['sg_total']:.2f}±{result['sg_total_std']:.2f}"
            pt_tot_str = f"{result['torch_total']:.2f}±{result['torch_total_std']:.2f}"
            delta_str = f"{result['delta']:.2f}±{result['delta_std']:.2f}"

            print(
                f"{kernel_str:<8} "
                f"{str(result['output_shape']):<20} "
                f"{sg_fwd_str:<15} "
                f"{pt_fwd_str:<15} "
                f"{sg_bwd_str:<15} "
                f"{pt_bwd_str:<15} "
                f"{sg_tot_str:<15} "
                f"{pt_tot_str:<15} "
                f"{delta_str:<15}"
            )

    # Additional benchmark: varying image sizes
    print(f"\n\n{'=' * 140}")
    print("Varying Image Sizes (kernel=3x3, batch=32)")
    print("=" * 140)

    img_sizes = [14, 28, 56, 112]
    kernel_size = 3
    batch_size = 32

    print(
        f"\n{'Size':<8} {'Shape':<20} {'SG Fwd (ms)':<15} {'PT Fwd (ms)':<15} {'SG Bwd (ms)':<15} {'PT Bwd (ms)':<15} {'SG Total (ms)':<15} {'PT Total (ms)':<15} {'Delta (ms)':<15}"
    )
    print("-" * 140)

    for img_size in img_sizes:
        result = benchmark_conv(
            batch_size=batch_size,
            in_channels=in_channels,
            img_size=img_size,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_runs=20,
        )

        img_str = f"{img_size}x{img_size}"
        sg_fwd_str = f"{result['sg_fwd']:.2f}±{result['sg_fwd_std']:.2f}"
        pt_fwd_str = f"{result['torch_fwd']:.2f}±{result['torch_fwd_std']:.2f}"
        sg_bwd_str = f"{result['sg_bwd']:.2f}±{result['sg_bwd_std']:.2f}"
        pt_bwd_str = f"{result['torch_bwd']:.2f}±{result['torch_bwd_std']:.2f}"
        sg_tot_str = f"{result['sg_total']:.2f}±{result['sg_total_std']:.2f}"
        pt_tot_str = f"{result['torch_total']:.2f}±{result['torch_total_std']:.2f}"
        delta_str = f"{result['delta']:.2f}±{result['delta_std']:.2f}"

        print(
            f"{img_str:<8} "
            f"{str(result['output_shape']):<20} "
            f"{sg_fwd_str:<15} "
            f"{pt_fwd_str:<15} "
            f"{sg_bwd_str:<15} "
            f"{pt_bwd_str:<15} "
            f"{sg_tot_str:<15} "
            f"{pt_tot_str:<15} "
            f"{delta_str:<15}"
        )

    # Additional benchmark: varying output channels
    print(f"\n\n{'=' * 140}")
    print("Varying Output Channels (kernel=3x3, batch=32, img=28x28)")
    print("=" * 140)

    out_channels_list = [8, 16, 32, 64, 128]
    kernel_size = 3
    batch_size = 32
    img_size = 28

    print(
        f"\n{'Shape':<20} {'SG Fwd (ms)':<15} {'PT Fwd (ms)':<15} {'SG Bwd (ms)':<15} {'PT Bwd (ms)':<15} {'SG Total (ms)':<15} {'PT Total (ms)':<15} {'Delta (ms)':<15}"
    )
    print("-" * 140)

    for out_ch in out_channels_list:
        result = benchmark_conv(
            batch_size=batch_size,
            in_channels=in_channels,
            img_size=img_size,
            out_channels=out_ch,
            kernel_size=kernel_size,
            n_runs=20,
        )

        sg_fwd_str = f"{result['sg_fwd']:.2f}±{result['sg_fwd_std']:.2f}"
        pt_fwd_str = f"{result['torch_fwd']:.2f}±{result['torch_fwd_std']:.2f}"
        sg_bwd_str = f"{result['sg_bwd']:.2f}±{result['sg_bwd_std']:.2f}"
        pt_bwd_str = f"{result['torch_bwd']:.2f}±{result['torch_bwd_std']:.2f}"
        sg_tot_str = f"{result['sg_total']:.2f}±{result['sg_total_std']:.2f}"
        pt_tot_str = f"{result['torch_total']:.2f}±{result['torch_total_std']:.2f}"
        delta_str = f"{result['delta']:.2f}±{result['delta_std']:.2f}"

        print(
            f"{str(result['output_shape']):<20} "
            f"{sg_fwd_str:<15} "
            f"{pt_fwd_str:<15} "
            f"{sg_bwd_str:<15} "
            f"{pt_bwd_str:<15} "
            f"{sg_tot_str:<15} "
            f"{pt_tot_str:<15} "
            f"{delta_str:<15}"
        )


if __name__ == "__main__":
    run_benchmarks()
