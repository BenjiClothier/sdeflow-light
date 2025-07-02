import torch
import numpy as np


def get_grid(sde, input_channels, input_height, n=4, num_steps=20, transform=None, 
             mean=0, std=1, clip=True, return_bpd_trajectory=False):
    """
    Generate a grid of samples with optional BPD tracking at each diffusion step.
    
    Args:
        sde: The SDE model
        input_channels: Number of input channels
        input_height: Height of input images
        n: Grid size (n x n samples)
        num_steps: Number of diffusion steps
        transform: Optional transform to apply to final samples
        mean: Mean for initial noise
        std: Standard deviation for initial noise
        clip: Whether to clip final samples to [0,1]
        return_bpd_trajectory: Whether to return BPD at each step
        
    Returns:
        If return_bpd_trajectory=False: final grid as numpy array
        If return_bpd_trajectory=True: (final_grid, bpd_trajectory)
    """
    num_samples = n ** 2
    delta = sde.T / num_steps
    y0 = torch.randn(num_samples, input_channels, input_height, input_height).to(sde.T)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(num_samples, 1, 1, 1).to(y0)
    
    bpd_trajectory = []
    dimx = input_channels * input_height * input_height

    with torch.no_grad():
        for i in range(num_steps):
            # Calculate BPD at current step if requested
            if return_bpd_trajectory:
                try:
                    # Calculate ELBO for current state
                    y_current = y0.clone().requires_grad_(True)
                    elbo_current = sde.elbo_random_t_slice(y_current)
                    bpd_current = -(elbo_current.mean().item() / dimx) / np.log(2) + 8
                    bpd_trajectory.append(bpd_current)
                except Exception as e:
                    # If ELBO calculation fails, append NaN
                    bpd_trajectory.append(float('nan'))
            
            # Perform diffusion step
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    # Calculate final BPD if requested
    if return_bpd_trajectory:
        try:
            y_final = y0.clone().requires_grad_(True)
            elbo_final = sde.elbo_random_t_slice(y_final)
            bpd_final = -(elbo_final.mean().item() / dimx) / np.log(2) + 8
            bpd_trajectory.append(bpd_final)
        except Exception as e:
            bpd_trajectory.append(float('nan'))

    if transform is not None:
        y0 = transform(y0)

    if clip:
        y0 = torch.clip(y0, 0, 1)

    # Reshape to grid format
    y0 = y0.view(
        n, n, input_channels, input_height, input_height).permute(
        2, 0, 3, 1, 4).contiguous().view(input_channels, n * input_height, n * input_height)

    y0 = y0.data.cpu().numpy()
    
    if return_bpd_trajectory:
        return y0, np.array(bpd_trajectory)
    else:
        return y0


def sample_with_bpd_tracking(sde, shape, num_steps=20, transform=None, mean=0, std=1):
    """
    Generate samples while tracking BPD at each diffusion step.
    
    Args:
        sde: The SDE model
        shape: Shape of samples to generate (batch_size, channels, height, width)
        num_steps: Number of diffusion steps
        transform: Optional transform to apply to final samples
        mean: Mean for initial noise
        std: Standard deviation for initial noise
        
    Returns:
        samples: Final generated samples
        bpd_trajectory: BPD values at each step
        step_samples: Intermediate samples at each step
    """
    batch_size, input_channels, input_height, input_width = shape
    delta = sde.T / num_steps
    y0 = torch.randn(batch_size, input_channels, input_height, input_width).to(sde.T.device)
    y0 = y0 * std + mean
    ts = torch.linspace(0, 1, num_steps + 1).to(y0) * sde.T
    ones = torch.ones(batch_size, 1, 1, 1).to(y0)
    
    bpd_trajectory = []
    step_samples = []
    dimx = input_channels * input_height * input_width

    with torch.no_grad():
        for i in range(num_steps):
            # Store current sample
            step_samples.append(y0.clone().cpu())
            
            # Calculate BPD at current step
            try:
                y_current = y0.clone().requires_grad_(True)
                elbo_current = sde.elbo_random_t_slice(y_current)
                bpd_current = -(elbo_current.mean().item() / dimx) / np.log(2) + 8
                bpd_trajectory.append(bpd_current)
            except Exception as e:
                bpd_trajectory.append(float('nan'))
            
            # Perform diffusion step
            mu = sde.mu(ones * ts[i], y0)
            sigma = sde.sigma(ones * ts[i], y0)
            y0 = y0 + delta * mu + delta ** 0.5 * sigma * torch.randn_like(y0)

    # Store final sample and calculate final BPD
    step_samples.append(y0.clone().cpu())
    try:
        y_final = y0.clone().requires_grad_(True)
        elbo_final = sde.elbo_random_t_slice(y_final)
        bpd_final = -(elbo_final.mean().item() / dimx) / np.log(2) + 8
        bpd_trajectory.append(bpd_final)
    except Exception as e:
        bpd_trajectory.append(float('nan'))

    if transform is not None:
        y0 = transform(y0)

    return y0, np.array(bpd_trajectory), step_samples
