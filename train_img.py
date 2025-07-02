import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from lib.sdes import VariancePreservingSDE, PluginReverseSDE
from lib.plotting import get_grid, sample_with_bpd_tracking
from lib.flows.elemwise import LogitTransform
from lib.models.unet import UNet
from lib.helpers import logging, create
from tensorboardX import SummaryWriter
import json
import matplotlib.pyplot as plt


_folder_name_keys = ['dataset', 'real', 'debias', 'batch_size', 'lr', 'num_iterations']


def get_args():
    parser = argparse.ArgumentParser()

    # i/o
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar'], default='mnist')
    parser.add_argument('--dataroot', type=str, default='./datasets')
    parser.add_argument('--saveroot', type=str, default='./saved')
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='number of integration steps for sampling')

    # optimization
    parser.add_argument('--T0', type=float, default=1.0,
                        help='integration time')
    parser.add_argument('--vtype', type=str, choices=['rademacher', 'gaussian'], default='rademacher',
                        help='random vector for the Hutchinson trace estimator')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000)

    # model
    parser.add_argument('--real', type=eval, choices=[True, False], default=True,
                        help='transforming the data from [0,1] to the real space using the logit function')
    parser.add_argument('--debias', type=eval, choices=[True, False], default=False,
                        help='using non-uniform sampling to debias the denoising score matching loss')
    
    # BPD tracking options
    parser.add_argument('--track_bpd_steps', type=eval, choices=[True, False], default=False,
                        help='track BPD at each diffusion step during sampling')
    parser.add_argument('--save_bpd_trajectory', type=eval, choices=[True, False], default=False,
                        help='save BPD trajectory to file')
    parser.add_argument('--bpd_sample_freq', type=int, default=5,
                        help='frequency of BPD trajectory sampling (every N evaluations)')

    return parser.parse_args()


def plot_bpd_trajectory(bpd_trajectory, save_path, iteration):
    """Plot and save BPD trajectory over diffusion steps."""
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(bpd_trajectory))
    valid_mask = ~np.isnan(bpd_trajectory)
    
    if np.any(valid_mask):
        plt.plot(steps[valid_mask], bpd_trajectory[valid_mask], 'b-', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Diffusion Step')
        plt.ylabel('Bits per Dimension')
        plt.title(f'BPD Trajectory During Sampling (Iteration {iteration})')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        valid_bpd = bpd_trajectory[valid_mask]
        stats_text = f'Mean: {valid_bpd.mean():.3f}\nStd: {valid_bpd.std():.3f}\nMin: {valid_bpd.min():.3f}\nMax: {valid_bpd.max():.3f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_bpd_data(bpd_trajectory, save_path, iteration):
    """Save BPD trajectory data to numpy file."""
    data_path = os.path.join(save_path, f'bpd_trajectory_iter_{iteration}.npy')
    np.save(data_path, bpd_trajectory)
    return data_path


args = get_args()
folder_tag = 'sde-flow'
folder_name = '-'.join([str(getattr(args, k)) for k in _folder_name_keys])
create(args.saveroot, folder_tag, args.expname, folder_name)
folder_path = os.path.join(args.saveroot, folder_tag, args.expname, folder_name)

# Create subdirectory for BPD data if tracking is enabled
if args.track_bpd_steps or args.save_bpd_trajectory:
    bpd_data_path = os.path.join(folder_path, 'bpd_trajectories')
    create(bpd_data_path)
else:
    bpd_data_path = None

print_ = lambda s: logging(s, folder_path)
print_(f'folder path: {folder_path}')
print_(str(args))
with open(os.path.join(folder_path, 'args.txt'), 'w') as out:
    out.write(json.dumps(args.__dict__, indent=4))
writer = SummaryWriter(folder_path)


if args.dataset == 'mnist':
    input_channels = 1
    input_height = 28
    dimx = input_channels * input_height ** 2

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args.dataroot, train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=args.dataroot, train=False,
                                         download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=True, num_workers=2)

    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=32,
        ch_mult=(1, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )

elif args.dataset == 'cifar':
    input_channels = 3
    input_height = 32
    dimx = input_channels * input_height ** 2

    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.dataroot, 'cifar10'), train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=os.path.join(args.dataroot, 'cifar10'), train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=True, num_workers=2)

    drift_q = UNet(
        input_channels=input_channels,
        input_height=input_height,
        ch=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        resamp_with_conv=True,
    )
else:
    raise NotImplementedError


T = torch.nn.Parameter(torch.FloatTensor([args.T0]), requires_grad=False)

inf_sde = VariancePreservingSDE(beta_min=0.1, beta_max=20.0, T=T)
gen_sde = PluginReverseSDE(inf_sde, drift_q, T, vtype=args.vtype, debias=args.debias)


cuda = torch.cuda.is_available()
if cuda:
    gen_sde.cuda()

optim = torch.optim.Adam(gen_sde.parameters(), lr=args.lr)

logit = LogitTransform(alpha=0.05)
if args.real:
    reverse = logit.reverse
else:
    reverse = None

# Counter for BPD trajectory sampling
bpd_eval_counter = 0


@torch.no_grad()
def evaluate():
    """Enhanced evaluation function that optionally returns BPD trajectory."""
    global bpd_eval_counter
    
    test_bpd = list()
    gen_sde.eval()
    
    for x_test, _ in testloader:
        if cuda:
            x_test = x_test.cuda()
        x_test = x_test * 255 / 256 + torch.rand_like(x_test) / 256
        if args.real:
            x_test, ldj = logit.forward_transform(x_test, 0)
            elbo_test = gen_sde.elbo_random_t_slice(x_test)
            elbo_test += ldj
        else:
            elbo_test = gen_sde.elbo_random_t_slice(x_test)

        test_bpd.extend(- (elbo_test.data.cpu().numpy() / dimx) / np.log(2) + 8)
    
    test_bpd = np.array(test_bpd)
    test_bpd_mean = test_bpd.mean()
    test_bpd_std_err = test_bpd.std() / len(testloader.dataset.data) ** 0.5
    
    # Generate BPD trajectory if requested and it's time to sample
    sampling_bpd_trajectory = None
    if args.track_bpd_steps and bpd_eval_counter % args.bpd_sample_freq == 0:
        try:
            shape = (4, input_channels, input_height, input_height)  # Small batch for efficiency
            _, sampling_bpd_trajectory, _ = sample_with_bpd_tracking(
                gen_sde, shape, num_steps=args.num_steps, transform=reverse
            )
        except Exception as e:
            print_(f"Warning: Could not generate sampling BPD trajectory: {e}")
            sampling_bpd_trajectory = None
    
    bpd_eval_counter += 1
    gen_sde.train()
    
    return test_bpd_mean, test_bpd_std_err, sampling_bpd_trajectory


if os.path.exists(os.path.join(folder_path, 'checkpoint.pt')):
    gen_sde, optim, not_finished, count = torch.load(os.path.join(folder_path, 'checkpoint.pt'))
else:
    not_finished = True
    count = 0
    writer.add_scalar('T', gen_sde.T.item(), count)
    
    # Initial sampling with optional BPD tracking
    if args.track_bpd_steps:
        initial_grid, initial_bpd_trajectory = get_grid(
            gen_sde, input_channels, input_height, n=4,
            num_steps=args.num_steps, transform=reverse, 
            return_bpd_trajectory=True
        )
        writer.add_image('samples', initial_grid, 0)
        
        # Log initial BPD trajectory
        if initial_bpd_trajectory is not None:
            valid_mask = ~np.isnan(initial_bpd_trajectory)
            for step, bpd_val in enumerate(initial_bpd_trajectory):
                if not np.isnan(bpd_val):
                    writer.add_scalar('bpd_per_step', bpd_val, step)
        
        if args.save_bpd_trajectory and bpd_data_path is not None:
            save_bpd_data(initial_bpd_trajectory, bpd_data_path, 0)
            plot_bpd_trajectory(initial_bpd_trajectory, 
                              os.path.join(bpd_data_path, 'bpd_trajectory_iter_0.png'), 0)
    else:
        writer.add_image('samples',
                         get_grid(gen_sde, input_channels, input_height, n=4,
                                  num_steps=args.num_steps, transform=reverse),
                         0)

# Store BPD trajectories for analysis
all_bpd_trajectories = []

while not_finished:
    for x, _ in trainloader:
        if cuda:
            x = x.cuda()
        x = x * 255 / 256 + torch.rand_like(x) / 256
        if args.real:
            x, _ = logit.forward_transform(x, 0)

        loss = gen_sde.dsm(x).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        count += 1
        if count == 1 or count % args.print_every == 0:
            writer.add_scalar('loss', loss.item(), count)
            writer.add_scalar('T', gen_sde.T.item(), count)

            # Enhanced evaluation
            eval_result = evaluate()
            bpd, std_err, sampling_bpd_trajectory = eval_result
            
            # Log standard metrics
            writer.add_scalar('bpd', bpd, count)
            writer.add_scalar('bpd_std_err', std_err, count)
            
            # Handle BPD trajectory if available
            if sampling_bpd_trajectory is not None:
                all_bpd_trajectories.append((count, sampling_bpd_trajectory))
                
                # Log each step's BPD to TensorBoard
                valid_mask = ~np.isnan(sampling_bpd_trajectory)
                for step, bpd_val in enumerate(sampling_bpd_trajectory):
                    if not np.isnan(bpd_val):
                        writer.add_scalar(f'sampling_bpd_step_{step}', bpd_val, count)
                
                # Log summary statistics of the trajectory
                if np.any(valid_mask):
                    valid_bpd = sampling_bpd_trajectory[valid_mask]
                    writer.add_scalar('sampling_bpd_trajectory_mean', valid_bpd.mean(), count)
                    writer.add_scalar('sampling_bpd_trajectory_std', valid_bpd.std(), count)
                    writer.add_scalar('sampling_bpd_trajectory_min', valid_bpd.min(), count)
                    writer.add_scalar('sampling_bpd_trajectory_max', valid_bpd.max(), count)
                
                # Save trajectory data
                if args.save_bpd_trajectory and bpd_data_path is not None:
                    save_bpd_data(sampling_bpd_trajectory, bpd_data_path, count)
                    if count % (args.print_every * 5) == 0:  # Plot less frequently
                        plot_path = os.path.join(bpd_data_path, f'bpd_trajectory_iter_{count}.png')
                        plot_bpd_trajectory(sampling_bpd_trajectory, plot_path, count)
                
                print_(f'Iteration {count} \tBPD {bpd:.4f} \tBPD Trajectory: {len(sampling_bpd_trajectory)} steps')
            else:
                print_(f'Iteration {count} \tBPD {bpd:.4f}')

        if count >= args.num_iterations:
            not_finished = False
            print_('Finished training')
            break

        if count % args.sample_every == 0:
            gen_sde.eval()
            
            if args.track_bpd_steps:
                sample_grid, sample_bpd_trajectory = get_grid(
                    gen_sde, input_channels, input_height, n=4,
                    num_steps=args.num_steps, transform=reverse,
                    return_bpd_trajectory=True
                )
                writer.add_image('samples', sample_grid, count)
                
                # Save sample BPD trajectory
                if args.save_bpd_trajectory and bpd_data_path is not None and sample_bpd_trajectory is not None:
                    save_bpd_data(sample_bpd_trajectory, bpd_data_path, f"{count}_sample")
            else:
                writer.add_image('samples',
                                 get_grid(gen_sde, input_channels, input_height, n=4,
                                          num_steps=args.num_steps, transform=reverse),
                                 count)
            gen_sde.train()

        if count % args.checkpoint_every == 0:
            torch.save([gen_sde, optim, not_finished, count], os.path.join(folder_path, 'checkpoint.pt'))

# Save final summary of all BPD trajectories
if args.save_bpd_trajectory and all_bpd_trajectories and bpd_data_path is not None:
    final_summary = {
        'iterations': [item[0] for item in all_bpd_trajectories],
        'trajectories': [item[1] for item in all_bpd_trajectories],
        'num_steps': args.num_steps,
        'dataset': args.dataset
    }
    
    summary_path = os.path.join(bpd_data_path, 'bpd_trajectories_summary.npz')
    np.savez(summary_path, **final_summary)
    print_(f'Saved BPD trajectories summary to {summary_path}')

print_('Training completed.')