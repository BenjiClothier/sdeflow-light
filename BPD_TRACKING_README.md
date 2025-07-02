# BPD Tracking Enhancement for sdeflow-light

This enhanced version of sdeflow-light includes the ability to track Bits per Dimension (BPD) at each diffusion step during the sampling process.

## New Features

### 1. BPD Tracking During Sampling
- Track BPD at each step of the reverse diffusion process
- Log BPD trajectories to TensorBoard
- Save BPD data for later analysis

### 2. Enhanced Command Line Arguments

The training script now includes additional options:

```bash
--track_bpd_steps=True/False     # Enable BPD tracking (default: False)
--save_bpd_trajectory=True/False # Save BPD data to files (default: False)  
--bpd_sample_freq=N             # Sample BPD every N evaluations (default: 5)
```

### 3. Analysis Tools

New analysis script `analyze_bpd.py` for comprehensive BPD trajectory analysis.

## Usage

### Basic Training with BPD Tracking

```bash
# MNIST with BPD tracking
python train_img.py --dataset=mnist --track_bpd_steps=True --save_bpd_trajectory=True

# CIFAR-10 with BPD tracking  
python train_img.py --dataset=cifar --track_bpd_steps=True --save_bpd_trajectory=True --print_every=1000
```

### Advanced Options

```bash
# Full training with all BPD features
python train_img.py \
    --dataset=cifar \
    --track_bpd_steps=True \
    --save_bpd_trajectory=True \
    --bpd_sample_freq=3 \
    --print_every=500 \
    --sample_every=1000 \
    --num_iterations=50000
```

### Analyzing Results

After training, analyze the BPD trajectories:

```bash
# Generate comprehensive analysis report
python analyze_bpd.py /path/to/experiment/bpd_trajectories --report

# Quick analysis with plots
python analyze_bpd.py /path/to/experiment/bpd_trajectories

# Plot specific iteration
python analyze_bpd.py /path/to/experiment/bpd_trajectories --iteration 10000

# Just evolution plot
python analyze_bpd.py /path/to/experiment/bpd_trajectories --evolution
```

## File Structure

When BPD tracking is enabled, the following structure is created:

```
experiment_folder/
├── tensorboard_logs/
├── bpd_trajectories/
│   ├── bpd_trajectory_iter_1000.npy
│   ├── bpd_trajectory_iter_2000.npy
│   ├── bpd_trajectory_iter_1000.png
│   ├── bpd_trajectories_summary.npz
│   └── analysis_report/
│       ├── trajectory_evolution.png
│       ├── trajectory_statistics.png
│       └── trajectory_iter_*.png
├── samples/
└── checkpoint.pt
```

## Understanding BPD Trajectories

### What is BPD?
Bits per Dimension (BPD) is a measure of how many bits are needed to encode each dimension of the data. Lower BPD indicates better model performance.

### Interpreting Trajectories
- **Decreasing BPD**: The model is improving sample quality during the diffusion process
- **Stable BPD**: The model maintains consistent quality throughout sampling  
- **Increasing BPD**: May indicate issues with the reverse process

### Key Metrics
- **Initial BPD**: Quality at the start of reverse diffusion (from noise)
- **Final BPD**: Quality of the final generated sample
- **Trajectory slope**: Rate of change in BPD during sampling
- **Variability**: How much BPD fluctuates during the process

## TensorBoard Integration

BPD tracking data is automatically logged to TensorBoard:

- `bpd`: Standard evaluation BPD on test set
- `sampling_bpd_step_N`: BPD at diffusion step N during sampling
- `sampling_bpd_trajectory_mean/std/min/max`: Statistics of the BPD trajectory

View with:
```bash
tensorboard --logdir=/path/to/experiment/folder
```

## Performance Considerations

- BPD tracking adds computational overhead during evaluation
- Use `--bpd_sample_freq` to control how often trajectories are computed
- For long training runs, consider setting `bpd_sample_freq=10` or higher
- BPD calculation requires gradient computation, so it's only done during evaluation

## Backward Compatibility

The enhanced code is fully backward compatible:
- Default behavior unchanged when `--track_bpd_steps=False`
- All original functionality preserved
- New features are opt-in

## Example Analysis Workflow

1. **Train model with BPD tracking**:
   ```bash
   python train_img.py --dataset=cifar --track_bpd_steps=True --save_bpd_trajectory=True --num_iterations=20000
   ```

2. **Monitor training**:
   ```bash
   tensorboard --logdir=~/.saved/sde-flow/default/
   ```

3. **Analyze results**:
   ```bash
   python analyze_bpd.py ~/.saved/sde-flow/default/experiment_name/bpd_trajectories --report
   ```

4. **Examine specific patterns**:
   ```bash
   python analyze_bpd.py ~/.saved/sde-flow/default/experiment_name/bpd_trajectories --iteration 15000
   ```

This enhancement provides detailed insights into how the model's generation quality evolves during the sampling process, enabling better understanding and debugging of diffusion models.