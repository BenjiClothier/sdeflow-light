#!/usr/bin/env python3
"""
Enhanced analysis script for BPD trajectories with likelihood calculations.
Usage: python enhanced_analyze_bpd.py <path_to_bpd_trajectories_folder>
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
from pathlib import Path
import argparse


def bpd_to_log_likelihood(bpd, num_dimensions):
    """Convert BPD to log-likelihood."""
    return -bpd * num_dimensions * math.log(2)


def bpd_to_likelihood(bpd, num_dimensions):
    """Convert BPD to likelihood (with numerical stability)."""
    log_likelihood = bpd_to_log_likelihood(bpd, num_dimensions)
    return math.exp(min(log_likelihood, 700))  # Prevent overflow


def load_bpd_trajectories(bpd_data_path):
    """Load all BPD trajectory files and try to infer dataset dimensions."""
    bpd_data_path = Path(bpd_data_path)
    trajectories = {}
    
    # Try to load summary file first to get dataset info
    summary_file = bpd_data_path / "bpd_trajectories_summary.npz"
    dataset_info = None
    
    if summary_file.exists():
        try:
            summary_data = np.load(summary_file, allow_pickle=True)
            dataset_info = {
                'dataset': str(summary_data['dataset']),
                'num_steps': int(summary_data['num_steps'])
            }
            print(f"Detected dataset: {dataset_info['dataset']}")
        except:
            pass
    
    # Load individual trajectory files
    for npy_file in bpd_data_path.glob("bpd_trajectory_iter_*.npy"):
        filename = npy_file.stem
        if 'sample' in filename:
            continue
        
        try:
            parts = filename.split('_')
            iteration = int(parts[-1])
            trajectory = np.load(npy_file)
            trajectories[iteration] = trajectory
        except (ValueError, IndexError):
            print(f"Warning: Could not parse iteration from {filename}")
    
    print(f"Loaded {len(trajectories)} BPD trajectories")
    return trajectories, dataset_info


def infer_dimensions(dataset_info):
    """Infer number of dimensions from dataset information."""
    if dataset_info and 'dataset' in dataset_info:
        dataset = dataset_info['dataset'].lower()
        if 'mnist' in dataset:
            return 1 * 28 * 28  # 784
        elif 'cifar' in dataset:
            return 3 * 32 * 32  # 3072
    
    # Default fallback - ask user or use CIFAR-10
    print("Could not infer dataset dimensions. Assuming CIFAR-10 (3072 dimensions).")
    print("Use --dimensions argument to specify manually.")
    return 3072


def plot_bpd_and_likelihood_evolution(trajectories, num_dimensions, save_path=None):
    """Plot both BPD and likelihood evolution during training."""
    if not trajectories:
        print("No trajectories to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort iterations
    sorted_iterations = sorted(trajectories.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_iterations)))
    
    # Plot BPD evolution
    for i, iteration in enumerate(sorted_iterations):
        trajectory = trajectories[iteration]
        steps = np.arange(len(trajectory))
        valid_mask = ~np.isnan(trajectory)
        
        if np.any(valid_mask):
            ax1.plot(steps[valid_mask], trajectory[valid_mask], 
                    color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Bits per Dimension')
    ax1.set_title('BPD Evolution During Training')
    ax1.grid(True, alpha=0.3)
    
    # Plot log-likelihood evolution
    for i, iteration in enumerate(sorted_iterations):
        trajectory = trajectories[iteration]
        steps = np.arange(len(trajectory))
        valid_mask = ~np.isnan(trajectory)
        
        if np.any(valid_mask):
            log_likelihood_trajectory = np.array([
                bpd_to_log_likelihood(bpd, num_dimensions) for bpd in trajectory[valid_mask]
            ])
            ax2.plot(steps[valid_mask], log_likelihood_trajectory, 
                    color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood Evolution During Training')
    ax2.grid(True, alpha=0.3)
    
    # Calculate and plot final values over training iterations
    final_bpds = []
    final_log_likelihoods = []
    initial_bpds = []
    initial_log_likelihoods = []
    
    for iteration in sorted_iterations:
        trajectory = trajectories[iteration]
        valid_trajectory = trajectory[~np.isnan(trajectory)]
        
        if len(valid_trajectory) > 0:
            final_bpds.append(valid_trajectory[-1])
            final_log_likelihoods.append(bpd_to_log_likelihood(valid_trajectory[-1], num_dimensions))
            initial_bpds.append(valid_trajectory[0])
            initial_log_likelihoods.append(bpd_to_log_likelihood(valid_trajectory[0], num_dimensions))
    
    # Plot final BPD over training
    ax3.plot(sorted_iterations, final_bpds, 'b-o', linewidth=2, markersize=5, label='Final BPD')
    ax3.plot(sorted_iterations, initial_bpds, 'r-s', linewidth=2, markersize=5, label='Initial BPD')
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('BPD')
    ax3.set_title('Initial vs Final BPD Over Training')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot final log-likelihood over training
    ax4.plot(sorted_iterations, final_log_likelihoods, 'b-o', linewidth=2, markersize=5, label='Final Log-Likelihood')
    ax4.plot(sorted_iterations, initial_log_likelihoods, 'r-s', linewidth=2, markersize=5, label='Initial Log-Likelihood')
    ax4.set_xlabel('Training Iteration')
    ax4.set_ylabel('Log-Likelihood')
    ax4.set_title('Initial vs Final Log-Likelihood Over Training')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add colorbars
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=min(sorted_iterations), 
                                               vmax=max(sorted_iterations)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], location='right', shrink=0.8)
    cbar.set_label('Training Iteration')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined evolution plot to {save_path}")
    else:
        plt.show()
    plt.close()


def analyze_likelihood_convergence(trajectories, num_dimensions):
    """Analyze convergence patterns in both BPD and likelihood space."""
    if not trajectories:
        print("No trajectories to analyze")
        return
    
    print("=== BPD and Likelihood Analysis ===")
    print(f"Dataset dimensions: {num_dimensions}")
    print(f"Total trajectories analyzed: {len(trajectories)}")
    
    iterations = sorted(trajectories.keys())
    print(f"Training iterations: {min(iterations)} to {max(iterations)}")
    
    # Analyze trajectories
    final_bpds = []
    final_log_likelihoods = []
    final_likelihoods = []
    bpd_changes = []
    log_likelihood_changes = []
    
    print(f"\n{'Iteration':<10} {'Initial BPD':<12} {'Final BPD':<12} {'BPD Change':<12} {'Final Log-Lik':<15} {'Log-Lik Change':<15}")
    print("-" * 90)
    
    for iteration in iterations:
        trajectory = trajectories[iteration]
        valid_bpd = trajectory[~np.isnan(trajectory)]
        
        if len(valid_bpd) > 1:
            initial_bpd = valid_bpd[0]
            final_bpd = valid_bpd[-1]
            bpd_change = final_bpd - initial_bpd
            
            final_log_lik = bpd_to_log_likelihood(final_bpd, num_dimensions)
            initial_log_lik = bpd_to_log_likelihood(initial_bpd, num_dimensions)
            log_lik_change = final_log_lik - initial_log_lik
            
            final_bpds.append(final_bpd)
            final_log_likelihoods.append(final_log_lik)
            final_likelihoods.append(bpd_to_likelihood(final_bpd, num_dimensions))
            bpd_changes.append(bpd_change)
            log_likelihood_changes.append(log_lik_change)
            
            print(f"{iteration:<10} {initial_bpd:<12.3f} {final_bpd:<12.3f} {bpd_change:<12.3f} {final_log_lik:<15.1f} {log_lik_change:<15.1f}")
    
    if final_bpds:
        print(f"\n=== Summary Statistics ===")
        print(f"Final BPD - Mean: {np.mean(final_bpds):.4f}, Std: {np.std(final_bpds):.4f}")
        print(f"Final Log-Likelihood - Mean: {np.mean(final_log_likelihoods):.1f}, Std: {np.std(final_log_likelihoods):.1f}")
        print(f"BPD Change - Mean: {np.mean(bpd_changes):.4f}, Std: {np.std(bpd_changes):.4f}")
        print(f"Log-Likelihood Change - Mean: {np.mean(log_likelihood_changes):.1f}, Std: {np.std(log_likelihood_changes):.1f}")
        
        # Interpretation
        avg_bpd_change = np.mean(bpd_changes)
        avg_loglik_change = np.mean(log_likelihood_changes)
        
        print(f"\n=== Interpretation ===")
        if avg_bpd_change < -0.1:
            print("🔽 BPD decreases significantly during sampling → Quality improves")
        elif avg_bpd_change > 0.1:
            print("🔼 BPD increases during sampling → Potential quality degradation")
        else:
            print("➡️  BPD remains relatively stable during sampling")
            
        if avg_loglik_change > 100:
            print("📈 Log-likelihood increases significantly → Model confidence improves")
        elif avg_loglik_change < -100:
            print("📉 Log-likelihood decreases → Model confidence drops")
        else:
            print("📊 Log-likelihood remains relatively stable")
        
        # Best and worst iterations
        best_iter = iterations[np.argmax(final_log_likelihoods)]
        worst_iter = iterations[np.argmin(final_log_likelihoods)]
        print(f"\nBest final log-likelihood: Iteration {best_iter} ({max(final_log_likelihoods):.1f})")
        print(f"Worst final log-likelihood: Iteration {worst_iter} ({min(final_log_likelihoods):.1f})")


def plot_single_trajectory_with_likelihood(trajectory, iteration, num_dimensions, save_path=None):
    """Plot both BPD and likelihood for a single trajectory."""
    steps = np.arange(len(trajectory))
    valid_mask = ~np.isnan(trajectory)
    
    if not np.any(valid_mask):
        print("No valid data points in trajectory")
        return
    
    # Calculate likelihood trajectory
    log_likelihood_trajectory = np.array([
        bpd_to_log_likelihood(bpd, num_dimensions) if not np.isnan(bpd) else np.nan
        for bpd in trajectory
    ])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot BPD
    ax1.plot(steps[valid_mask], trajectory[valid_mask], 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_ylabel('Bits per Dimension')
    ax1.set_title(f'BPD Trajectory at Training Iteration {iteration}')
    ax1.grid(True, alpha=0.3)
    
    # Add BPD statistics
    valid_bpd = trajectory[valid_mask]
    bpd_stats = f'Mean: {valid_bpd.mean():.3f}\nStd: {valid_bpd.std():.3f}\nChange: {valid_bpd[-1] - valid_bpd[0]:.3f}'
    ax1.text(0.02, 0.98, bpd_stats, transform=ax1.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Plot log-likelihood
    valid_loglik_mask = ~np.isnan(log_likelihood_trajectory)
    ax2.plot(steps[valid_loglik_mask], log_likelihood_trajectory[valid_loglik_mask], 
             'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title(f'Log-Likelihood Trajectory at Training Iteration {iteration}')
    ax2.grid(True, alpha=0.3)
    
    # Add log-likelihood statistics
    valid_loglik = log_likelihood_trajectory[valid_loglik_mask]
    loglik_stats = f'Mean: {valid_loglik.mean():.1f}\nStd: {valid_loglik.std():.1f}\nChange: {valid_loglik[-1] - valid_loglik[0]:.1f}'
    ax2.text(0.02, 0.98, loglik_stats, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze BPD trajectories with likelihood calculations")
    parser.add_argument("bpd_data_path", help="Path to BPD trajectory data directory")
    parser.add_argument("--dimensions", type=int, help="Number of data dimensions (auto-inferred if not provided)")
    parser.add_argument("--iteration", type=int, help="Plot specific iteration")
    parser.add_argument("--report", action="store_true", help="Generate full analysis report")
    parser.add_argument("--evolution", action="store_true", help="Plot trajectory evolution")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bpd_data_path):
        print(f"Error: Path {args.bpd_data_path} does not exist")
        return 1
    
    # Load trajectories and dataset info
    trajectories, dataset_info = load_bpd_trajectories(args.bpd_data_path)
    
    if not trajectories:
        print("No BPD trajectories found. Make sure the path contains .npy files with BPD data.")
        return 1
    
    # Determine number of dimensions
    if args.dimensions:
        num_dimensions = args.dimensions
        print(f"Using manually specified dimensions: {num_dimensions}")
    else:
        num_dimensions = infer_dimensions(dataset_info)
    
    if args.report:
        # Generate full report with likelihood analysis
        output_dir = Path(args.bpd_data_path) / "enhanced_analysis_report"
        output_dir.mkdir(exist_ok=True)
        print(f"Creating enhanced analysis report in {output_dir}")
        
        # Generate comprehensive plots
        plot_bpd_and_likelihood_evolution(trajectories, num_dimensions, 
                                        save_path=output_dir / "bpd_likelihood_evolution.png")
        
        # Generate individual trajectory plots for key iterations
        iterations = sorted(trajectories.keys())
        key_iterations = [iterations[0]]
        if len(iterations) > 1:
            key_iterations.append(iterations[len(iterations)//2])
            key_iterations.append(iterations[-1])
        
        for iteration in key_iterations:
            if iteration in trajectories:
                plot_single_trajectory_with_likelihood(
                    trajectories[iteration], iteration, num_dimensions,
                    save_path=output_dir / f"trajectory_likelihood_iter_{iteration}.png"
                )
        
        # Generate analysis
        print("\n" + "="*80)
        analyze_likelihood_convergence(trajectories, num_dimensions)
        print("="*80)
        
        # Save numerical results
        results_file = output_dir / "likelihood_analysis_results.txt"
        with open(results_file, 'w') as f:
            # Redirect stdout to file temporarily
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print(f"Enhanced BPD and Likelihood Analysis")
            print(f"Dataset dimensions: {num_dimensions}")
            print(f"Analysis timestamp: {pd.Timestamp.now()}" if 'pd' in globals() else "")
            print("\n")
            analyze_likelihood_convergence(trajectories, num_dimensions)
            
            sys.stdout = original_stdout
        
        print(f"\nEnhanced analysis report saved to: {output_dir}")
        
    elif args.iteration:
        # Plot specific iteration with likelihood
        if args.iteration in trajectories:
            plot_single_trajectory_with_likelihood(trajectories[args.iteration], args.iteration, num_dimensions)
        else:
            print(f"Iteration {args.iteration} not found. Available: {sorted(trajectories.keys())}")
    
    elif args.evolution:
        # Plot evolution with likelihood
        plot_bpd_and_likelihood_evolution(trajectories, num_dimensions)
    
    else:
        # Default: run analysis and show key information
        analyze_likelihood_convergence(trajectories, num_dimensions)
        
        # Show evolution plot if multiple trajectories
        if len(trajectories) > 1:
            plot_bpd_and_likelihood_evolution(trajectories, num_dimensions)
        
        # Show a sample trajectory with likelihood
        sample_iter = sorted(trajectories.keys())[-1]  # Show latest
        plot_single_trajectory_with_likelihood(trajectories[sample_iter], sample_iter, num_dimensions)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())