#!/usr/bin/env python3
"""
Simple inference script for quick sample generation with likelihood tracking.
This is a lightweight version for basic usage.

Usage:
    python simple_inference.py --checkpoint /path/to/checkpoint.pt --samples 4 --steps 100
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math

# Add this at the top to handle import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.flows.elemwise import LogitTransform


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 2:
        gen_sde = checkpoint[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gen_sde = gen_sde.to(device)
        gen_sde.eval()
        return gen_sde, device
    else:
        raise ValueError("Invalid checkpoint format")


def get_dataset_info(checkpoint_path):
    """Try to infer dataset information from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    
    # Look for args.txt
    args_file = checkpoint_dir / 'args.txt'
    if args_file.exists():
        with open(args_file, 'r') as f:
            try:
                config = json.load(f)
                dataset = config.get('dataset', 'cifar')
                real = config.get('real', True)
                return dataset, real
            except:
                pass
    
    # Fallback defaults
    return 'cifar', True


def calculate_likelihood(sde, y, num_dimensions):
    """Calculate log-likelihood and BPD for given samples."""
    try:
        y_calc = y.clone().requires_grad_(True)
        elbo = sde.elbo_random_t_slice(y_calc)
        log_likelihood = elbo.mean().item()
        bpd = -log_likelihood / (num_dimensions * math.log(2)) + 8
        return log_likelihood, bpd
    except Exception as e:
        return np.nan, np.nan


def generate_samples_simple(sde, num_samples, num_steps, dataset='cifar', use_real_transform=True, device='cpu'):
    """Generate samples with likelihood tracking."""
    
    # Set up dimensions and transforms
    if dataset == 'mnist':
        input_channels, input_height = 1, 28
        num_dimensions = 1 * 28 * 28
    else:  # cifar or default
        input_channels, input_height = 3, 32
        num_dimensions = 3 * 32 * 32
    
    # Initialize logit transform
    logit = LogitTransform(alpha=0.05) if use_real_transform else None
    
    # Generate samples
    print(f"Generating {num_samples} samples with {num_steps} steps...")
    
    shape = (num_samples, input_channels, input_height, input_height)
    y = torch.randn(shape, device=device)
    
    delta = sde.T / num_steps
    ts = torch.linspace(0, 1, num_steps + 1, device=device) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1, device=device)
    
    # Track likelihood
    log_likelihoods = []
    bpds = []
    
    with torch.no_grad():
        for i in range(num_steps):
            # Calculate likelihood at current step
            log_lik, bpd = calculate_likelihood(sde, y, num_dimensions)
            log_likelihoods.append(log_lik)
            bpds.append(bpd)
            
            if (i + 1) % max(1, num_steps // 10) == 0:
                print(f"  Step {i+1}/{num_steps}: Log-likelihood = {log_lik:.1f}, BPD = {bpd:.3f}")
            
            # Diffusion step
            mu = sde.mu(ones * ts[i], y)
            sigma = sde.sigma(ones * ts[i], y)
            y = y + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y)
        
        # Final step
        log_lik, bpd = calculate_likelihood(sde, y, num_dimensions)
        log_likelihoods.append(log_lik)
        bpds.append(bpd)
        print(f"  Final: Log-likelihood = {log_lik:.1f}, BPD = {bpd:.3f}")
    
    # Apply reverse transform
    if logit is not None:
        y = logit.reverse(y)
    
    y = torch.clamp(y, 0, 1)
    
    return y, np.array(log_likelihoods), np.array(bpds)


def plot_results(samples, log_likelihoods, bpds, save_path=None):
    """Plot samples and likelihood trajectory."""
    fig = plt.figure(figsize=(15, 10))
    
    # Create sample grid
    num_samples = samples.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Plot samples
    ax1 = plt.subplot(2, 2, (1, 2))
    
    # Arrange samples in grid
    grid_samples = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < num_samples:
                sample = samples[idx].cpu().numpy()
                if sample.shape[0] == 1:  # Grayscale
                    sample = sample.squeeze(0)
                else:  # Color
                    sample = sample.transpose(1, 2, 0)
            else:
                # Pad with zeros
                if samples.shape[1] == 1:
                    sample = np.zeros((samples.shape[2], samples.shape[3]))
                else:
                    sample = np.zeros((samples.shape[2], samples.shape[3], samples.shape[1]))
            row.append(sample)
        grid_samples.append(row)
    
    # Create the full grid
    if samples.shape[1] == 1:  # Grayscale
        full_grid = np.block(grid_samples)
        ax1.imshow(full_grid, cmap='gray')
    else:  # Color
        # Stack horizontally and vertically
        rows = []
        for row in grid_samples:
            rows.append(np.hstack(row))
        full_grid = np.vstack(rows)
        ax1.imshow(full_grid)
    
    ax1.set_title(f'Generated Samples ({num_samples} samples)', fontsize=14)
    ax1.axis('off')
    
    # Plot log-likelihood trajectory
    ax2 = plt.subplot(2, 2, 3)
    steps = np.arange(len(log_likelihoods))
    valid_mask = ~np.isnan(log_likelihoods)
    
    if np.any(valid_mask):
        ax2.plot(steps[valid_mask], log_likelihoods[valid_mask], 'b-', linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Diffusion Step')
        ax2.set_ylabel('Log-Likelihood')
        ax2.set_title('Log-Likelihood Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Add improvement text
        valid_ll = log_likelihoods[valid_mask]
        improvement = valid_ll[-1] - valid_ll[0]
        ax2.text(0.02, 0.98, f'Improvement: {improvement:.1f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot BPD trajectory
    ax3 = plt.subplot(2, 2, 4)
    valid_bpd_mask = ~np.isnan(bpds)
    
    if np.any(valid_bpd_mask):
        ax3.plot(steps[valid_bpd_mask], bpds[valid_bpd_mask], 'r-', linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('Diffusion Step')
        ax3.set_ylabel('Bits per Dimension')
        ax3.set_title('BPD Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Add improvement text
        valid_bpd = bpds[valid_bpd_mask]
        bpd_change = valid_bpd[-1] - valid_bpd[0]
        ax3.text(0.02, 0.98, f'Change: {bpd_change:.3f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()


def print_summary(log_likelihoods, bpds, num_samples, num_steps):
    """Print summary statistics."""
    print(f"\n{'='*50}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*50}")
    print(f"Samples generated: {num_samples}")
    print(f"Diffusion steps: {num_steps}")
    
    valid_ll = log_likelihoods[~np.isnan(log_likelihoods)]
    valid_bpd = bpds[~np.isnan(bpds)]
    
    if len(valid_ll) > 0:
        print(f"\nLog-Likelihood:")
        print(f"  Initial: {valid_ll[0]:.2f}")
        print(f"  Final:   {valid_ll[-1]:.2f}")
        print(f"  Change:  {valid_ll[-1] - valid_ll[0]:.2f}")
        if valid_ll[-1] > valid_ll[0]:
            print(f"  → Model confidence IMPROVED during sampling ✓")
        else:
            print(f"  → Model confidence decreased during sampling")
    
    if len(valid_bpd) > 0:
        print(f"\nBits per Dimension:")
        print(f"  Initial: {valid_bpd[0]:.4f}")
        print(f"  Final:   {valid_bpd[-1]:.4f}")
        print(f"  Change:  {valid_bpd[-1] - valid_bpd[0]:.4f}")
        if valid_bpd[-1] < valid_bpd[0]:
            print(f"  → Sample quality IMPROVED during diffusion ✓")
        else:
            print(f"  → Sample quality decreased during diffusion")
    
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="Simple sample generation with likelihood tracking")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint.pt')
    parser.add_argument('--samples', type=int, default=9, help='Number of samples (default: 9)')
    parser.add_argument('--steps', type=int, default=200, help='Number of diffusion steps (default: 200)')
    parser.add_argument('--output', type=str, default=None, help='Output file path (optional)')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset (mnist/cifar, auto-detected if not provided)')
    parser.add_argument('--no-transform', action='store_true', help='Disable logit transform')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    try:
        sde, device = load_model(args.checkpoint)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Get dataset info
    if args.dataset:
        dataset = args.dataset
        use_real_transform = not args.no_transform
    else:
        dataset, use_real_transform = get_dataset_info(args.checkpoint)
        if args.no_transform:
            use_real_transform = False
    
    print(f"Dataset: {dataset}")
    print(f"Using real transform: {use_real_transform}")
    
    # Generate samples
    try:
        samples, log_likelihoods, bpds = generate_samples_simple(
            sde, args.samples, args.steps, dataset, use_real_transform, device
        )
        
        # Print summary
        print_summary(log_likelihoods, bpds, args.samples, args.steps)
        
        # Plot results
        output_path = args.output
        if output_path is None:
            checkpoint_name = Path(args.checkpoint).stem
            output_path = f"inference_results_{checkpoint_name}.png"
        
        plot_results(samples, log_likelihoods, bpds, save_path=output_path)
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())