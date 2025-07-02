import torch
import numpy as np
import math


def bpd_to_log_likelihood(bpd, num_dimensions):
    """
    Convert Bits per Dimension (BPD) to log-likelihood.
    
    Args:
        bpd: Bits per dimension value (scalar or array)
        num_dimensions: Total number of dimensions in the data
        
    Returns:
        log_likelihood: Log-likelihood value(s)
    """
    return -bpd * num_dimensions * math.log(2)


def log_likelihood_to_bpd(log_likelihood, num_dimensions):
    """
    Convert log-likelihood to Bits per Dimension (BPD).
    
    Args:
        log_likelihood: Log-likelihood value (scalar or array)
        num_dimensions: Total number of dimensions in the data
        
    Returns:
        bpd: Bits per dimension value(s)
    """
    return -log_likelihood / (num_dimensions * math.log(2))


def bpd_to_likelihood(bpd, num_dimensions):
    """
    Convert BPD to likelihood (with numerical stability).
    
    Args:
        bpd: Bits per dimension value
        num_dimensions: Total number of dimensions in the data
        
    Returns:
        likelihood: Likelihood value (can be very small)
    """
    log_likelihood = bpd_to_log_likelihood(bpd, num_dimensions)
    # Clip to prevent overflow/underflow
    if isinstance(log_likelihood, np.ndarray):
        return np.exp(np.clip(log_likelihood, -700, 700))
    else:
        return math.exp(min(max(log_likelihood, -700), 700))


def get_data_dimensions(dataset, input_channels=None, input_height=None):
    """
    Get the number of dimensions for common datasets.
    
    Args:
        dataset: Dataset name ('mnist', 'cifar', etc.)
        input_channels: Number of channels (optional override)
        input_height: Image height (optional override)
        
    Returns:
        num_dimensions: Number of dimensions per sample
    """
    if input_channels is not None and input_height is not None:
        return input_channels * input_height * input_height
    
    dataset = dataset.lower()
    if 'mnist' in dataset:
        return 1 * 28 * 28  # 784
    elif 'cifar' in dataset:
        return 3 * 32 * 32  # 3072
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Please specify input_channels and input_height.")


def evaluate_with_likelihood_metrics(sde, testloader, args, logit=None, cuda=False):
    """
    Enhanced evaluation function that returns both BPD and likelihood metrics.
    
    This is a drop-in replacement for the original evaluate() function.
    
    Args:
        sde: The SDE model
        testloader: Test data loader
        args: Arguments containing configuration
        logit: Logit transform (optional)
        cuda: Whether to use CUDA
        
    Returns:
        Dictionary with all metrics
    """
    test_bpd = []
    test_log_likelihood = []
    sde.eval()
    
    # Get dataset dimensions
    num_dimensions = get_data_dimensions(args.dataset)
    
    with torch.no_grad():
        for x_test, _ in testloader:
            if cuda:
                x_test = x_test.cuda()
            
            # Dequantization (same as original code)
            x_test = x_test * 255 / 256 + torch.rand_like(x_test) / 256
            
            if args.real and logit is not None:
                x_test, ldj = logit.forward_transform(x_test, 0)
                elbo_test = sde.elbo_random_t_slice(x_test)
                elbo_test += ldj
            else:
                elbo_test = sde.elbo_random_t_slice(x_test)
            
            # ELBO is already log-likelihood
            log_likelihood_batch = elbo_test.data.cpu().numpy()
            test_log_likelihood.extend(log_likelihood_batch)
            
            # Convert to BPD (note: +8 for dequantization)
            bpd_batch = -log_likelihood_batch / (num_dimensions * np.log(2)) + 8
            test_bpd.extend(bpd_batch)
    
    sde.train()
    
    # Calculate statistics
    test_bpd = np.array(test_bpd)
    test_log_likelihood = np.array(test_log_likelihood)
    
    # Return all metrics
    return {
        'bpd_mean': test_bpd.mean(),
        'bpd_std_err': test_bpd.std() / np.sqrt(len(test_bpd)),
        'log_likelihood_mean': test_log_likelihood.mean(),
        'log_likelihood_std': test_log_likelihood.std(),
        'num_dimensions': num_dimensions,
        'num_samples': len(test_bpd),
        
        # For backward compatibility
        'test_bpd_mean': test_bpd.mean(),
        'test_bpd_std_err': test_bpd.std() / np.sqrt(len(test_bpd))
    }


def convert_bpd_trajectory_to_likelihood(bpd_trajectory, num_dimensions):
    """
    Convert a BPD trajectory to likelihood metrics.
    
    Args:
        bpd_trajectory: Array of BPD values at each diffusion step
        num_dimensions: Number of dimensions in the data
        
    Returns:
        Dictionary with likelihood trajectory and statistics
    """
    # Handle NaN values
    valid_mask = ~np.isnan(bpd_trajectory)
    
    # Convert to log-likelihood
    log_likelihood_trajectory = np.full_like(bpd_trajectory, np.nan)
    log_likelihood_trajectory[valid_mask] = bpd_to_log_likelihood(
        bpd_trajectory[valid_mask], num_dimensions
    )
    
    # Convert to likelihood (with numerical stability)
    likelihood_trajectory = np.full_like(bpd_trajectory, np.nan)
    valid_log_lik = log_likelihood_trajectory[valid_mask]
    likelihood_trajectory[valid_mask] = np.exp(np.clip(valid_log_lik, -700, 700))
    
    # Calculate statistics
    if np.any(valid_mask):
        valid_log_lik = log_likelihood_trajectory[valid_mask]
        return {
            'log_likelihood_trajectory': log_likelihood_trajectory,
            'likelihood_trajectory': likelihood_trajectory,
            'log_likelihood_initial': valid_log_lik[0],
            'log_likelihood_final': valid_log_lik[-1],
            'log_likelihood_change': valid_log_lik[-1] - valid_log_lik[0],
            'log_likelihood_mean': valid_log_lik.mean(),
            'log_likelihood_std': valid_log_lik.std(),
            'num_valid_steps': len(valid_log_lik),
            'num_dimensions': num_dimensions
        }
    else:
        return {
            'log_likelihood_trajectory': log_likelihood_trajectory,
            'likelihood_trajectory': likelihood_trajectory,
            'num_valid_steps': 0,
            'num_dimensions': num_dimensions
        }


def log_likelihood_metrics_to_tensorboard(writer, metrics, iteration_count):
    """
    Log likelihood metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary from evaluate_with_likelihood_metrics()
        iteration_count: Current training iteration
    """
    # Log basic metrics
    writer.add_scalar('log_likelihood_mean', metrics['log_likelihood_mean'], iteration_count)
    writer.add_scalar('log_likelihood_std', metrics['log_likelihood_std'], iteration_count)
    
    # Log the relationship between BPD and log-likelihood
    writer.add_scalar('bpd_vs_log_likelihood_ratio', 
                     -metrics['bpd_mean'] / metrics['log_likelihood_mean'], 
                     iteration_count)


def log_trajectory_likelihood_to_tensorboard(writer, bpd_trajectory, num_dimensions, iteration_count):
    """
    Log trajectory likelihood metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        bpd_trajectory: Array of BPD values
        num_dimensions: Number of data dimensions
        iteration_count: Current training iteration
    """
    if bpd_trajectory is None or len(bpd_trajectory) == 0:
        return
    
    # Convert to likelihood metrics
    likelihood_metrics = convert_bpd_trajectory_to_likelihood(bpd_trajectory, num_dimensions)
    
    if likelihood_metrics['num_valid_steps'] > 0:
        # Log trajectory statistics
        writer.add_scalar('trajectory_log_likelihood_initial', 
                         likelihood_metrics['log_likelihood_initial'], iteration_count)
        writer.add_scalar('trajectory_log_likelihood_final', 
                         likelihood_metrics['log_likelihood_final'], iteration_count)
        writer.add_scalar('trajectory_log_likelihood_change', 
                         likelihood_metrics['log_likelihood_change'], iteration_count)
        writer.add_scalar('trajectory_log_likelihood_mean', 
                         likelihood_metrics['log_likelihood_mean'], iteration_count)
        
        # Log individual steps (if not too many)
        if len(bpd_trajectory) <= 100:  # Avoid cluttering TensorBoard
            log_lik_traj = likelihood_metrics['log_likelihood_trajectory']
            for step, log_lik in enumerate(log_lik_traj):
                if not np.isnan(log_lik):
                    writer.add_scalar(f'trajectory_log_likelihood_step_{step}', log_lik, iteration_count)


# Example usage in train_img.py:
"""
# Add this import at the top of train_img.py:
from lib.likelihood_utils import (
    evaluate_with_likelihood_metrics,
    log_likelihood_metrics_to_tensorboard,
    log_trajectory_likelihood_to_tensorboard,
    get_data_dimensions
)

# Replace the evaluate() function call with:
@torch.no_grad()
def evaluate():
    return evaluate_with_likelihood_metrics(gen_sde, testloader, args, logit, cuda)

# In the training loop, replace the evaluation logging with:
if count == 1 or count % args.print_every == 0:
    writer.add_scalar('loss', loss.item(), count)
    writer.add_scalar('T', gen_sde.T.item(), count)

    # Enhanced evaluation with likelihood metrics
    eval_metrics = evaluate()
    
    # Log all metrics
    writer.add_scalar('bpd', eval_metrics['bpd_mean'], count)
    writer.add_scalar('bpd_std_err', eval_metrics['bpd_std_err'], count)
    log_likelihood_metrics_to_tensorboard(writer, eval_metrics, count)
    
    # Handle BPD trajectory if available
    if sampling_bpd_trajectory is not None:
        num_dimensions = eval_metrics['num_dimensions']
        log_trajectory_likelihood_to_tensorboard(writer, sampling_bpd_trajectory, num_dimensions, count)
    
    print_(f'Iteration {count} \\tBPD {eval_metrics["bpd_mean"]:.4f} \\tLog-Likelihood {eval_metrics["log_likelihood_mean"]:.1f}')
"""