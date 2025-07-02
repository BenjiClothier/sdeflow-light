#!/usr/bin/env python3
"""
Inference script for sdeflow-light with log-likelihood tracking.
Generates samples from trained diffusion models and displays likelihood evolution.

Usage:
    python inference.py --checkpoint_path /path/to/checkpoint.pt --num_samples 16 --num_steps 1000
"""

import argparse
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pathlib import Path
import json
from PIL import Image

# Import model components
from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.flows.elemwise import LogitTransform
from lib.models.unet import UNet
from lib.helpers import create


def get_args():
    parser = argparse.ArgumentParser(description="Generate samples with likelihood tracking")
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint (checkpoint.pt)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to args.txt config file (auto-detected if not provided)')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='Number of diffusion steps for generation')
    parser.add_argument('--grid_size', type=int, default=None,
                        help='Grid size for display (auto-calculated if not provided)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                        help='Directory to save outputs')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate samples during diffusion')
    parser.add_argument('--save_animation', action='store_true',
                        help='Create animation of the diffusion process')
    parser.add_argument('--show_plots', action='store_true',
                        help='Display plots interactively')
    
    # Advanced options
    parser.add_argument('--lambda_param', type=float, default=0.0,
                        help='Lambda parameter for family of reverse SDEs (0=standard, 1=ODE)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (1.0=standard)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


class InferenceEngine:
    """Main inference engine for generating samples with likelihood tracking."""
    
    def __init__(self, checkpoint_path, config_path=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration and checkpoint
        self.load_model()
        
        # Calculate dataset dimensions
        if hasattr(self, 'args'):
            if self.args.dataset == 'mnist':
                self.num_dimensions = 1 * 28 * 28
                self.input_channels = 1
                self.input_height = 28
            elif self.args.dataset == 'cifar':
                self.num_dimensions = 3 * 32 * 32
                self.input_channels = 3
                self.input_height = 32
        else:
            # Fallback defaults
            self.num_dimensions = 3072
            self.input_channels = 3
            self.input_height = 32
    
    def load_model(self):
        """Load model from checkpoint and configuration."""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract model components
        if isinstance(checkpoint, (list, tuple)) and len(checkpoint) >= 2:
            gen_sde = checkpoint[0]
            self.gen_sde = gen_sde.to(self.device)
        else:
            raise ValueError("Invalid checkpoint format")
        
        # Load configuration if available
        if self.config_path is None:
            # Try to find args.txt in the same directory
            potential_config = self.checkpoint_path.parent / 'args.txt'
            if potential_config.exists():
                self.config_path = potential_config
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
                # Convert to namespace for easier access
                self.args = argparse.Namespace(**config_dict)
            print(f"Loaded configuration from {self.config_path}")
        else:
            print("Warning: No configuration file found. Using defaults.")
            # Create minimal default args
            self.args = argparse.Namespace(
                dataset='cifar',
                real=True,
                num_steps=1000
            )
        
        # Initialize logit transform if needed
        if hasattr(self.args, 'real') and self.args.real:
            self.logit = LogitTransform(alpha=0.05)
            self.reverse_transform = self.logit.reverse
        else:
            self.logit = None
            self.reverse_transform = None
        
        self.gen_sde.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def bpd_to_log_likelihood(self, bpd):
        """Convert BPD to log-likelihood."""
        return -bpd * self.num_dimensions * np.log(2)
    
    def calculate_likelihood_at_step(self, y, requires_grad=True):
        """Calculate log-likelihood at a given step."""
        try:
            if requires_grad:
                y_calc = y.clone().requires_grad_(True)
            else:
                y_calc = y.clone()
            
            with torch.set_grad_enabled(requires_grad):
                elbo = self.gen_sde.elbo_random_t_slice(y_calc)
                log_likelihood = elbo.mean().item()
                bpd = -log_likelihood / (self.num_dimensions * np.log(2)) + 8
                
            return log_likelihood, bpd
        except Exception as e:
            print(f"Warning: Likelihood calculation failed: {e}")
            return np.nan, np.nan
    
    def generate_samples_with_tracking(self, num_samples, num_steps, lambda_param=0.0, 
                                     temperature=1.0, save_intermediate=False):
        """
        Generate samples while tracking likelihood at each step.
        
        Returns:
            final_samples: Generated samples
            trajectory_data: Dictionary with tracking information
        """
        print(f"Generating {num_samples} samples with {num_steps} steps")
        
        # Initialize
        shape = (num_samples, self.input_channels, self.input_height, self.input_height)
        y0 = torch.randn(shape, device=self.device) * temperature
        
        delta = self.gen_sde.T / num_steps
        ts = torch.linspace(0, 1, num_steps + 1).to(y0) * self.gen_sde.T
        ones = torch.ones(num_samples, 1, 1, 1, device=self.device)
        
        # Tracking arrays
        log_likelihoods = []
        bpds = []
        step_samples = []
        step_times = []
        
        print("Starting diffusion process...")
        
        with torch.no_grad():
            for i in range(num_steps):
                if (i + 1) % (num_steps // 10) == 0 or i < 10:
                    print(f"  Step {i+1}/{num_steps}")
                
                # Calculate likelihood at current step
                log_lik, bpd = self.calculate_likelihood_at_step(y0, requires_grad=True)
                log_likelihoods.append(log_lik)
                bpds.append(bpd)
                step_times.append(ts[i].item())
                
                # Save intermediate sample if requested
                if save_intermediate:
                    step_samples.append(y0.clone().cpu())
                
                # Perform diffusion step
                mu = self.gen_sde.mu(ones * ts[i], y0, lmbd=lambda_param)
                sigma = self.gen_sde.sigma(ones * ts[i], y0, lmbd=lambda_param)
                y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)
            
            # Final step
            log_lik, bpd = self.calculate_likelihood_at_step(y0, requires_grad=True)
            log_likelihoods.append(log_lik)
            bpds.append(bpd)
            step_times.append(ts[-1].item())
            
            if save_intermediate:
                step_samples.append(y0.clone().cpu())
        
        # Apply reverse transform if needed
        final_samples = y0.clone()
        if self.reverse_transform is not None:
            final_samples = self.reverse_transform(final_samples)
        
        # Clip to valid range
        final_samples = torch.clamp(final_samples, 0, 1)
        
        trajectory_data = {
            'log_likelihoods': np.array(log_likelihoods),
            'bpds': np.array(bpds),
            'step_times': np.array(step_times),
            'step_samples': step_samples if save_intermediate else None,
            'num_steps': num_steps,
            'num_samples': num_samples,
            'lambda_param': lambda_param,
            'temperature': temperature
        }
        
        print("Generation complete!")
        return final_samples, trajectory_data
    
    def plot_likelihood_trajectory(self, trajectory_data, save_path=None, show=True):
        """Plot likelihood trajectory during diffusion."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        steps = np.arange(len(trajectory_data['log_likelihoods']))
        log_likelihoods = trajectory_data['log_likelihoods']
        bpds = trajectory_data['bpds']
        
        # Remove NaN values for plotting
        valid_mask = ~(np.isnan(log_likelihoods) | np.isnan(bpds))
        
        if np.any(valid_mask):
            # Plot log-likelihood
            ax1.plot(steps[valid_mask], log_likelihoods[valid_mask], 'b-', linewidth=2, marker='o', markersize=3)
            ax1.set_ylabel('Log-Likelihood', fontsize=12)
            ax1.set_title(f'Log-Likelihood During Diffusion Sampling ({trajectory_data["num_samples"]} samples)', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text
            valid_log_lik = log_likelihoods[valid_mask]
            initial_ll = valid_log_lik[0]
            final_ll = valid_log_lik[-1]
            change_ll = final_ll - initial_ll
            
            stats_text = f'Initial: {initial_ll:.1f}\nFinal: {final_ll:.1f}\nChange: {change_ll:.1f}'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Plot BPD
            ax2.plot(steps[valid_mask], bpds[valid_mask], 'r-', linewidth=2, marker='s', markersize=3)
            ax2.set_xlabel('Diffusion Step', fontsize=12)
            ax2.set_ylabel('Bits per Dimension', fontsize=12)
            ax2.set_title('BPD During Diffusion Sampling', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add BPD statistics
            valid_bpd = bpds[valid_mask]
            initial_bpd = valid_bpd[0]
            final_bpd = valid_bpd[-1]
            change_bpd = final_bpd - initial_bpd
            
            bpd_stats_text = f'Initial: {initial_bpd:.3f}\nFinal: {final_bpd:.3f}\nChange: {change_bpd:.3f}'
            ax2.text(0.02, 0.98, bpd_stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Likelihood trajectory saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_sample_grid(self, samples, grid_size=None):
        """Create a grid visualization of generated samples."""
        num_samples = samples.shape[0]
        
        if grid_size is None:
            grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        # Pad with zeros if necessary
        samples_padded = samples
        total_needed = grid_size * grid_size
        if num_samples < total_needed:
            padding_shape = (total_needed - num_samples,) + samples.shape[1:]
            padding = torch.zeros(padding_shape, device=samples.device)
            samples_padded = torch.cat([samples, padding], dim=0)
        elif num_samples > total_needed:
            samples_padded = samples[:total_needed]
        
        # Reshape and permute for grid display
        grid = samples_padded.view(grid_size, grid_size, *samples.shape[1:])
        
        if samples.shape[1] == 1:  # Grayscale
            grid = grid.squeeze(2)  # Remove channel dimension
            grid = grid.permute(0, 2, 1, 3).contiguous()
            grid = grid.view(grid_size * samples.shape[2], grid_size * samples.shape[3])
        else:  # Color
            grid = grid.permute(0, 2, 1, 3, 4).contiguous()
            grid = grid.view(samples.shape[1], grid_size * samples.shape[2], grid_size * samples.shape[3])
            grid = grid.permute(1, 2, 0).contiguous()
        
        return grid.cpu().numpy()
    
    def plot_samples(self, samples, trajectory_data, save_path=None, show=True):
        """Plot generated samples alongside likelihood information."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
        
        # Plot samples
        ax_samples = fig.add_subplot(gs[0, :])
        grid = self.create_sample_grid(samples)
        
        if len(grid.shape) == 2:  # Grayscale
            ax_samples.imshow(grid, cmap='gray')
        else:  # Color
            ax_samples.imshow(grid)
        
        ax_samples.set_title(f'Generated Samples ({trajectory_data["num_samples"]} samples, {trajectory_data["num_steps"]} steps)', 
                           fontsize=16)
        ax_samples.axis('off')
        
        # Plot likelihood trajectory
        ax_likelihood = fig.add_subplot(gs[1, 0])
        steps = np.arange(len(trajectory_data['log_likelihoods']))
        log_likelihoods = trajectory_data['log_likelihoods']
        valid_mask = ~np.isnan(log_likelihoods)
        
        if np.any(valid_mask):
            ax_likelihood.plot(steps[valid_mask], log_likelihoods[valid_mask], 'b-', linewidth=2)
            ax_likelihood.set_xlabel('Diffusion Step')
            ax_likelihood.set_ylabel('Log-Likelihood')
            ax_likelihood.set_title('Log-Likelihood Evolution')
            ax_likelihood.grid(True, alpha=0.3)
        
        # Plot BPD trajectory
        ax_bpd = fig.add_subplot(gs[1, 1])
        bpds = trajectory_data['bpds']
        valid_bpd_mask = ~np.isnan(bpds)
        
        if np.any(valid_bpd_mask):
            ax_bpd.plot(steps[valid_bpd_mask], bpds[valid_bpd_mask], 'r-', linewidth=2)
            ax_bpd.set_xlabel('Diffusion Step')
            ax_bpd.set_ylabel('BPD')
            ax_bpd.set_title('BPD Evolution')
            ax_bpd.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Sample plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_diffusion_animation(self, trajectory_data, save_path, fps=10):
        """Create animation showing the diffusion process."""
        if trajectory_data['step_samples'] is None:
            print("No intermediate samples saved. Cannot create animation.")
            return
        
        print(f"Creating diffusion animation...")
        
        step_samples = trajectory_data['step_samples']
        log_likelihoods = trajectory_data['log_likelihoods']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Show current sample
            current_samples = step_samples[frame]
            if self.reverse_transform is not None:
                current_samples = self.reverse_transform(current_samples)
            current_samples = torch.clamp(current_samples, 0, 1)
            
            grid = self.create_sample_grid(current_samples)
            
            if len(grid.shape) == 2:  # Grayscale
                ax1.imshow(grid, cmap='gray')
            else:  # Color
                ax1.imshow(grid)
            
            ax1.set_title(f'Diffusion Step {frame}/{len(step_samples)-1}')
            ax1.axis('off')
            
            # Show likelihood trajectory up to current step
            steps_so_far = np.arange(frame + 1)
            ll_so_far = log_likelihoods[:frame + 1]
            valid_mask = ~np.isnan(ll_so_far)
            
            if np.any(valid_mask):
                ax2.plot(steps_so_far[valid_mask], ll_so_far[valid_mask], 'b-', linewidth=2, marker='o')
                ax2.set_xlim(0, len(log_likelihoods))
                ax2.set_ylim(np.nanmin(log_likelihoods), np.nanmax(log_likelihoods))
                ax2.set_xlabel('Diffusion Step')
                ax2.set_ylabel('Log-Likelihood')
                ax2.set_title(f'Log-Likelihood: {ll_so_far[-1]:.1f}')
                ax2.grid(True, alpha=0.3)
        
        anim = FuncAnimation(fig, animate, frames=len(step_samples), interval=1000//fps, repeat=True)
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
        plt.close()


def main():
    args = get_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    engine = InferenceEngine(args.checkpoint_path, args.config_path)
    
    # Calculate grid size
    grid_size = args.grid_size
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(args.num_samples)))
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples, trajectory_data = engine.generate_samples_with_tracking(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        lambda_param=args.lambda_param,
        temperature=args.temperature,
        save_intermediate=args.save_intermediate or args.save_animation
    )
    
    # Save outputs
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S') if 'pd' in globals() else 'output'
    
    # Plot and save likelihood trajectory
    likelihood_plot_path = output_dir / f'likelihood_trajectory_{timestamp}.png'
    engine.plot_likelihood_trajectory(trajectory_data, save_path=likelihood_plot_path, show=args.show_plots)
    
    # Plot and save samples with likelihood info
    samples_plot_path = output_dir / f'samples_with_likelihood_{timestamp}.png'
    engine.plot_samples(samples, trajectory_data, save_path=samples_plot_path, show=args.show_plots)
    
    # Save raw samples
    samples_path = output_dir / f'samples_{timestamp}.pt'
    torch.save(samples.cpu(), samples_path)
    print(f"Raw samples saved to {samples_path}")
    
    # Save trajectory data
    trajectory_path = output_dir / f'trajectory_data_{timestamp}.npz'
    np.savez(trajectory_path, **trajectory_data)
    print(f"Trajectory data saved to {trajectory_path}")
    
    # Create animation if requested
    if args.save_animation and trajectory_data['step_samples'] is not None:
        animation_path = output_dir / f'diffusion_animation_{timestamp}.gif'
        engine.create_diffusion_animation(trajectory_data, animation_path)
    
    # Print summary statistics
    print(f"\n=== Generation Summary ===")
    print(f"Number of samples: {args.num_samples}")
    print(f"Number of diffusion steps: {args.num_steps}")
    print(f"Lambda parameter: {args.lambda_param}")
    print(f"Temperature: {args.temperature}")
    
    valid_ll = trajectory_data['log_likelihoods'][~np.isnan(trajectory_data['log_likelihoods'])]
    if len(valid_ll) > 0:
        print(f"Initial log-likelihood: {valid_ll[0]:.2f}")
        print(f"Final log-likelihood: {valid_ll[-1]:.2f}")
        print(f"Log-likelihood change: {valid_ll[-1] - valid_ll[0]:.2f}")
    
    valid_bpd = trajectory_data['bpds'][~np.isnan(trajectory_data['bpds'])]
    if len(valid_bpd) > 0:
        print(f"Initial BPD: {valid_bpd[0]:.4f}")
        print(f"Final BPD: {valid_bpd[-1]:.4f}")
        print(f"BPD change: {valid_bpd[-1] - valid_bpd[0]:.4f}")
    
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()