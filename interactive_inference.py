#!/usr/bin/env python3
"""
Interactive inference script for exploring diffusion models.
This script provides functions for Jupyter notebook or interactive Python sessions.

Usage in Jupyter/IPython:
    %run interactive_inference.py
    model = load_diffusion_model('path/to/checkpoint.pt')
    samples, trajectory = generate_with_likelihood_tracking(model, num_samples=4, num_steps=100)
    plot_diffusion_process(samples, trajectory)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import math
from typing import Tuple, Dict, Optional, List
import warnings

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class DiffusionModel:
    """Wrapper class for easy interaction with diffusion models."""
    
    def __init__(self, sde, dataset='cifar', use_real_transform=True, device='cpu'):
        self.sde = sde
        self.dataset = dataset
        self.use_real_transform = use_real_transform
        self.device = device
        
        # Set up dataset parameters
        if dataset == 'mnist':
            self.input_channels, self.input_height = 1, 28
            self.num_dimensions = 784
        else:  # cifar or default
            self.input_channels, self.input_height = 3, 32
            self.num_dimensions = 3072
        
        # Set up transforms
        if use_real_transform:
            from lib.flows.elemwise import LogitTransform
            self.logit = LogitTransform(alpha=0.05)
        else:
            self.logit = None
    
    def calculate_likelihood(self, y: torch.Tensor) -> Tuple[float, float]:
        """Calculate log-likelihood and BPD for samples."""
        try:
            y_calc = y.clone().requires_grad_(True)
            elbo = self.sde.elbo_random_t_slice(y_calc)
            log_likelihood = elbo.mean().item()
            bpd = -log_likelihood / (self.num_dimensions * math.log(2)) + 8
            return log_likelihood, bpd
        except Exception as e:
            return np.nan, np.nan
    
    def single_diffusion_step(self, y: torch.Tensor, t: torch.Tensor, lambda_param: float = 0.0) -> torch.Tensor:
        """Perform a single diffusion step."""
        batch_size = y.shape[0]
        ones = torch.ones(batch_size, 1, 1, 1, device=self.device)
        
        mu = self.sde.mu(ones * t, y, lmbd=lambda_param)
        sigma = self.sde.sigma(ones * t, y, lmbd=lambda_param)
        
        delta = self.sde.T / 1000  # Assume 1000 total steps for delta calculation
        noise = torch.randn_like(y)
        
        return y + delta * mu + delta ** 0.5 * sigma * noise
    
    def generate_samples(self, num_samples: int, num_steps: int = 200, 
                        lambda_param: float = 0.0, temperature: float = 1.0,
                        track_likelihood: bool = True, 
                        save_intermediate: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Generate samples with optional likelihood tracking."""
        
        # Initialize
        shape = (num_samples, self.input_channels, self.input_height, self.input_height)
        y = torch.randn(shape, device=self.device) * temperature
        
        delta = self.sde.T / num_steps
        ts = torch.linspace(0, 1, num_steps + 1, device=self.device) * self.sde.T
        ones = torch.ones(num_samples, 1, 1, 1, device=self.device)
        
        # Tracking
        trajectory = None
        if track_likelihood or save_intermediate:
            trajectory = {
                'log_likelihoods': [],
                'bpds': [],
                'steps': [],
                'intermediate_samples': [] if save_intermediate else None
            }
        
        with torch.no_grad():
            for i in range(num_steps):
                # Track likelihood
                if track_likelihood:
                    log_lik, bpd = self.calculate_likelihood(y)
                    trajectory['log_likelihoods'].append(log_lik)
                    trajectory['bpds'].append(bpd)
                    trajectory['steps'].append(i)
                
                # Save intermediate
                if save_intermediate:
                    trajectory['intermediate_samples'].append(y.clone().cpu())
                
                # Diffusion step
                mu = self.sde.mu(ones * ts[i], y, lmbd=lambda_param)
                sigma = self.sde.sigma(ones * ts[i], y, lmbd=lambda_param)
                y = y + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y)
            
            # Final tracking
            if track_likelihood:
                log_lik, bpd = self.calculate_likelihood(y)
                trajectory['log_likelihoods'].append(log_lik)
                trajectory['bpds'].append(bpd)
                trajectory['steps'].append(num_steps)
            
            if save_intermediate:
                trajectory['intermediate_samples'].append(y.clone().cpu())
        
        # Apply reverse transform
        if self.logit is not None:
            y = self.logit.reverse(y)
        y = torch.clamp(y, 0, 1)
        
        # Convert tracking lists to arrays
        if trajectory:
            if track_likelihood:
                trajectory['log_likelihoods'] = np.array(trajectory['log_likelihoods'])
                trajectory['bpds'] = np.array(trajectory['bpds'])
                trajectory['steps'] = np.array(trajectory['steps'])
        
        return y, trajectory


def load_diffusion_model(checkpoint_path: str, dataset: Optional[str] = None, 
                        use_real_transform: Optional[bool] = None) -> DiffusionModel:
    """Load a diffusion model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 2:
        sde = checkpoint[0].to(device)
        sde.eval()
    else:
        raise ValueError("Invalid checkpoint format")
    
    # Auto-detect dataset and transform settings
    if dataset is None or use_real_transform is None:
        args_file = checkpoint_path.parent / 'args.txt'
        if args_file.exists():
            try:
                with open(args_file, 'r') as f:
                    config = json.load(f)
                    if dataset is None:
                        dataset = config.get('dataset', 'cifar')
                    if use_real_transform is None:
                        use_real_transform = config.get('real', True)
            except:
                warnings.warn("Could not load config file, using defaults")
    
    # Set defaults
    if dataset is None:
        dataset = 'cifar'
    if use_real_transform is None:
        use_real_transform = True
    
    print(f"Dataset: {dataset}, Real transform: {use_real_transform}, Device: {device}")
    
    return DiffusionModel(sde, dataset, use_real_transform, device)


def generate_with_likelihood_tracking(model: DiffusionModel, num_samples: int = 4, 
                                     num_steps: int = 200, **kwargs) -> Tuple[torch.Tensor, Dict]:
    """Generate samples with likelihood tracking (convenience function)."""
    return model.generate_samples(num_samples, num_steps, track_likelihood=True, **kwargs)


def plot_samples_grid(samples: torch.Tensor, title: str = "Generated Samples", 
                     figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """Plot samples in a grid."""
    num_samples = samples.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    if figsize is None:
        figsize = (grid_size * 2, grid_size * 2)
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < num_samples:
            sample = samples[i].cpu().numpy()
            if sample.shape[0] == 1:  # Grayscale
                ax.imshow(sample.squeeze(0), cmap='gray')
            else:  # Color
                ax.imshow(sample.transpose(1, 2, 0))
        ax.axis('off')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_likelihood_trajectory(trajectory: Dict, title: str = "Likelihood Evolution", 
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """Plot likelihood trajectory."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    steps = trajectory['steps']
    log_likelihoods = trajectory['log_likelihoods']
    bpds = trajectory['bpds']
    
    # Remove NaN values
    valid_mask = ~(np.isnan(log_likelihoods) | np.isnan(bpds))
    
    if np.any(valid_mask):
        valid_steps = steps[valid_mask]
        valid_ll = log_likelihoods[valid_mask]
        valid_bpd = bpds[valid_mask]
        
        # Plot log-likelihood
        ax1.plot(valid_steps, valid_ll, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_ylabel('Log-Likelihood', fontsize=12)
        ax1.set_title('Log-Likelihood During Diffusion', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if len(valid_ll) > 1:
            improvement = valid_ll[-1] - valid_ll[0]
            ax1.annotate(f'Δ = {improvement:.1f}', 
                        xy=(valid_steps[-1], valid_ll[-1]), xytext=(10, 10),
                        textcoords='offset points', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot BPD
        ax2.plot(valid_steps, valid_bpd, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Diffusion Step', fontsize=12)
        ax2.set_ylabel('Bits per Dimension', fontsize=12)
        ax2.set_title('BPD During Diffusion', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add BPD change annotation
        if len(valid_bpd) > 1:
            bpd_change = valid_bpd[-1] - valid_bpd[0]
            ax2.annotate(f'Δ = {bpd_change:.3f}', 
                        xy=(valid_steps[-1], valid_bpd[-1]), xytext=(10, 10),
                        textcoords='offset points', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig


def plot_diffusion_process(samples: torch.Tensor, trajectory: Dict, 
                          figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """Plot both samples and likelihood trajectory."""
    fig = plt.figure(figsize=figsize)
    
    # Create layout
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 1, 1])
    
    # Plot samples
    ax_samples = fig.add_subplot(gs[0, :])
    num_samples = samples.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create sample grid manually for better control
    sample_grid = []
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
                # Pad with white
                if samples.shape[1] == 1:
                    sample = np.ones((samples.shape[2], samples.shape[3]))
                else:
                    sample = np.ones((samples.shape[2], samples.shape[3], samples.shape[1]))
            row.append(sample)
        sample_grid.append(row)
    
    # Display grid
    if samples.shape[1] == 1:  # Grayscale
        full_grid = np.block(sample_grid)
        ax_samples.imshow(full_grid, cmap='gray')
    else:  # Color
        rows = [np.hstack(row) for row in sample_grid]
        full_grid = np.vstack(rows)
        ax_samples.imshow(np.clip(full_grid, 0, 1))
    
    ax_samples.set_title(f'Generated Samples ({num_samples} samples)', fontsize=16)
    ax_samples.axis('off')
    
    # Plot likelihood trajectory
    if trajectory and 'log_likelihoods' in trajectory:
        ax_ll = fig.add_subplot(gs[1, 0])
        ax_bpd = fig.add_subplot(gs[1, 1])
        
        steps = trajectory['steps']
        log_likelihoods = trajectory['log_likelihoods']
        bpds = trajectory['bpds']
        
        valid_mask = ~(np.isnan(log_likelihoods) | np.isnan(bpds))
        
        if np.any(valid_mask):
            valid_steps = steps[valid_mask]
            valid_ll = log_likelihoods[valid_mask]
            valid_bpd = bpds[valid_mask]
            
            # Log-likelihood
            ax_ll.plot(valid_steps, valid_ll, 'b-', linewidth=2, marker='o', markersize=3)
            ax_ll.set_xlabel('Step')
            ax_ll.set_ylabel('Log-Likelihood')
            ax_ll.set_title('LL Evolution')
            ax_ll.grid(True, alpha=0.3)
            
            # BPD
            ax_bpd.plot(valid_steps, valid_bpd, 'r-', linewidth=2, marker='s', markersize=3)
            ax_bpd.set_xlabel('Step')
            ax_bpd.set_ylabel('BPD')
            ax_bpd.set_title('BPD Evolution')
            ax_bpd.grid(True, alpha=0.3)
        
        # Summary statistics
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.axis('off')
        
        if np.any(valid_mask) and len(valid_ll) > 1:
            stats_text = f"""Summary:
            
Initial LL: {valid_ll[0]:.1f}
Final LL: {valid_ll[-1]:.1f}
LL Change: {valid_ll[-1] - valid_ll[0]:.1f}

Initial BPD: {valid_bpd[0]:.3f}
Final BPD: {valid_bpd[-1]:.3f}
BPD Change: {valid_bpd[-1] - valid_bpd[0]:.3f}

Steps: {len(valid_steps)}
Samples: {num_samples}"""
            
            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def compare_lambda_parameters(model: DiffusionModel, lambda_values: List[float] = [0.0, 0.5, 1.0],
                             num_samples: int = 4, num_steps: int = 100) -> plt.Figure:
    """Compare different lambda parameters (family of reverse SDEs)."""
    results = {}
    
    print("Generating samples with different lambda parameters...")
    for lam in lambda_values:
        print(f"  Lambda = {lam}")
        samples, trajectory = model.generate_samples(
            num_samples, num_steps, lambda_param=lam, track_likelihood=True
        )
        results[lam] = (samples, trajectory)
    
    # Create comparison plot
    fig = plt.figure(figsize=(20, 12))
    n_lambda = len(lambda_values)
    
    # Plot samples for each lambda
    for i, lam in enumerate(lambda_values):
        samples, trajectory = results[lam]
        
        # Samples
        ax = plt.subplot(3, n_lambda, i + 1)
        sample_to_show = samples[0].cpu().numpy()  # Show first sample
        if sample_to_show.shape[0] == 1:
            ax.imshow(sample_to_show.squeeze(0), cmap='gray')
        else:
            ax.imshow(sample_to_show.transpose(1, 2, 0))
        ax.set_title(f'λ = {lam}', fontsize=14)
        ax.axis('off')
        
        # Log-likelihood trajectory
        ax_ll = plt.subplot(3, n_lambda, n_lambda + i + 1)
        if trajectory and 'log_likelihoods' in trajectory:
            valid_mask = ~np.isnan(trajectory['log_likelihoods'])
            if np.any(valid_mask):
                ax_ll.plot(trajectory['steps'][valid_mask], 
                          trajectory['log_likelihoods'][valid_mask], 
                          linewidth=2, marker='o', markersize=3)
        ax_ll.set_ylabel('Log-Likelihood')
        ax_ll.set_title(f'LL Evolution (λ={lam})')
        ax_ll.grid(True, alpha=0.3)
        
        # BPD trajectory
        ax_bpd = plt.subplot(3, n_lambda, 2 * n_lambda + i + 1)
        if trajectory and 'bpds' in trajectory:
            valid_mask = ~np.isnan(trajectory['bpds'])
            if np.any(valid_mask):
                ax_bpd.plot(trajectory['steps'][valid_mask], 
                           trajectory['bpds'][valid_mask], 
                           linewidth=2, marker='s', markersize=3, color='red')
        ax_bpd.set_xlabel('Diffusion Step')
        ax_bpd.set_ylabel('BPD')
        ax_bpd.set_title(f'BPD Evolution (λ={lam})')
        ax_bpd.grid(True, alpha=0.3)
    
    fig.suptitle('Comparison of Lambda Parameters (λ=0: standard SDE, λ=1: ODE)', fontsize=16)
    plt.tight_layout()
    return fig


# Convenience functions for quick exploration
def quick_sample(checkpoint_path: str, num_samples: int = 9, num_steps: int = 100):
    """Quick sampling function for exploration."""
    model = load_diffusion_model(checkpoint_path)
    samples, trajectory = generate_with_likelihood_tracking(model, num_samples, num_steps)
    return plot_diffusion_process(samples, trajectory)


def explore_lambda(checkpoint_path: str, num_samples: int = 4, num_steps: int = 100):
    """Explore different lambda parameters."""
    model = load_diffusion_model(checkpoint_path)
    return compare_lambda_parameters(model, [0.0, 0.5, 1.0], num_samples, num_steps)


# Example usage documentation
def show_usage_examples():
    """Print usage examples."""
    print("""
=== Interactive Diffusion Model Inference ===

Basic Usage:
    # Load model
    model = load_diffusion_model('path/to/checkpoint.pt')
    
    # Generate samples with likelihood tracking
    samples, trajectory = generate_with_likelihood_tracking(model, num_samples=9, num_steps=200)
    
    # Plot results
    plot_diffusion_process(samples, trajectory)

Quick Functions:
    # Quick sampling
    quick_sample('path/to/checkpoint.pt', num_samples=9, num_steps=200)
    
    # Explore lambda parameters
    explore_lambda('path/to/checkpoint.pt', num_samples=4, num_steps=100)

Advanced Usage:
    # Generate with custom parameters
    samples, trajectory = model.generate_samples(
        num_samples=16, 
        num_steps=500,
        lambda_param=0.5,  # Interpolate between SDE and ODE
        temperature=1.2,   # Higher temperature = more diverse samples
        save_intermediate=True  # Save intermediate steps
    )
    
    # Plot just samples
    plot_samples_grid(samples, "My Generated Samples")
    
    # Plot just likelihood trajectory
    plot_likelihood_trajectory(trajectory, "Likelihood Evolution")
    
    # Compare different settings
    compare_lambda_parameters(model, [0.0, 0.25, 0.5, 0.75, 1.0])

Understanding the Output:
    - Log-Likelihood: Higher values = better model confidence
    - BPD (Bits per Dimension): Lower values = better compression/quality
    - Lambda parameter: 0.0 = stochastic SDE, 1.0 = deterministic ODE
    - Temperature: Controls diversity (1.0 = standard, >1.0 = more diverse)
    """)


if __name__ == "__main__":
    show_usage_examples()