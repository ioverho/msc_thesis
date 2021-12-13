# ==============================================================================
# Package import
# ==============================================================================
import os
import yaml
import argparse
import warnings
from shutil import copyfile

# 3rd Party
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import higher

# User-defined
from maml_test.sine import SampleSinuisoid
from maml_test.regressor import SineRegressor

from utils.experiment import find_version, set_seed, set_deterministic, Timer

CHECKPOINT_DIR = "./checkpoints"

def train(args):
    """Train loop for MAML on toy sinusoid-regression task.

    Args:
        args ([type]): [description]
    """

    timer = Timer()

    #*#################
    #* Config reading #
    #*#################

    with open(args.config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    print(50 * "+")
    print("HYPER-PARAMETERS")
    print(yaml.dump(config))
    print(50 * "+")

    #*#####################
    #* Experiment Set-up ##
    #*#####################
    print("\nEXPERIMENT SET-UP")

    # == Version
    # ==== ./checkpoints/data_version/version_number
    full_version, experiment_dir, version_dir = find_version(config['run']['experiment_name'],
                                                             CHECKPOINT_DIR,
                                                             debug=config['run']['debug'])

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)

    copyfile(args.config_file_path,
             f"{CHECKPOINT_DIR}/{full_version}/{os.path.split(args.config_file_path)[-1]}")

    # == Device
    use_cuda = config['run']['gpu'] or config['run']['gpu'] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on {device}" + f"- {torch.cuda.get_device_name(0)}" if use_cuda else "")

    # == Logging
    writer = SummaryWriter(log_dir=f"{CHECKPOINT_DIR}/{full_version}/tensorboard")

    print(f"Saving to {CHECKPOINT_DIR}/{full_version}")

    # == Reproducibility
    set_seed(config['run']['seed'])
    if config['run']['deterministic']:
        set_deterministic()

    #*#########################
    #* Training Initalization #
    #*#########################
    print("\nDATA, MODEL, TRAINING")

    dataset = SampleSinuisoid(device)

    model = SineRegressor(hidden_dim=40).to(device)

    meta_opt = optim.Adam(model.parameters(), lr=config['train']['meta_lr'])
    adapt_opt = optim.SGD(model.parameters(), lr=config['train']['adapt_lr'])

    #*#########################
    #* Training Loop #
    #*#########################
    # Ideally this all gets abstracted away using
    # PyTorch Lightning or my own API

    model.train()
    model.thaw()

    for epoch in range(config['train']['epochs']):
        # OUTER
        # This is a single episode
        # i.e. a set of tasks

        s_losses, q_losses = [], []
        for i in range(config['train']['meta_batches']):
            # INNER LOOP
            # This is a single task
            dataset.sample_task()

            # Aggregate the meta loss
            loss_meta = torch.tensor([0.0], dtype=torch.float, device=device)
            with higher.innerloop_ctx(model,
                                    adapt_opt,
                                    copy_initial_weights=False,
                                    track_higher_grads=config['train']['first_order_train'])\
            as (model_episode, opt_episode):
                # Adaptation loop
                # Allowed to train on support for a single task
                for ii in range(config['train']['adaptation_steps']):

                    # Sample a single (support) batch
                    x_s, y_s = dataset.sample_batch(
                        config['train']['K_support'])
                    y_s_hat = model_episode(x_s)

                    loss_s = F.mse_loss(y_s_hat, y_s, reduction='none')

                    # Optimize on batch loss
                    opt_episode.step(torch.mean(loss_s))

                    if ii == 0:
                        # Record loss prior to adaption
                        rmse_s = torch.mean(torch.sqrt(loss_s.detach().cpu()))
                        s_losses.append(rmse_s)

                # Sample a single (query) batch
                x_q, y_q = dataset.sample_batch(config['train']['K_query'])
                y_q_hat = model_episode(x_q)

                loss_q = F.mse_loss(y_q_hat, y_q, reduction='none')

                # Add adaptation validation loss to meta loss
                loss_meta += torch.mean(loss_q)

                # Record adaptation validation loss post adaption
                # for comparison to pre-adaption
                rmse_q = torch.mean(torch.sqrt(loss_q.detach().cpu()))
                q_losses.append(rmse_q)

        # Average the meta loss over the number tasks
        loss_meta = loss_meta / config['train']['meta_batches']

        loss_meta.backward()

        # Optimize the meta-model
        meta_opt.step()
        meta_opt.zero_grad()

        # Boilerplate logging code
        s_losses, q_losses = torch.stack(s_losses), torch.stack(q_losses)
        mean_loss_s = torch.mean(s_losses).item()
        mean_loss_q = torch.mean(q_losses).item()
        mean_loss_diff = torch.mean(q_losses - s_losses).item()
        mean_loss_rdiff = (torch.mean((q_losses - s_losses) / s_losses)).item()
        s_losses, q_losses = [], []

        if epoch % config['train']['verbosity'] == 0 or epoch == config['train']['epochs'] - 1:
            print(f"{timer.time()} | {epoch:04d}.{i:>02d} | RMSE: pre-adapt: {mean_loss_s:6.2e},"+\
                f"adapt: {mean_loss_q:6.2e}, diff: {mean_loss_diff:+6.2e}, rdiff: {mean_loss_rdiff:+3.2f}")

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'meta_opt': meta_opt.state_dict(),
                'mean_loss_rdiff': mean_loss_rdiff,
                },
                f"{CHECKPOINT_DIR}/{full_version}/checkpoint.pt"
            )

        writer.add_scalar(
            tag='Loss/Mean Support Loss (pre-adapt)',
            scalar_value=mean_loss_s,
            global_step=epoch
        )

        writer.add_scalar(
            tag='Loss/Mean Query Loss (post-adapt)',
            scalar_value=mean_loss_q,
            global_step=epoch
        )

        writer.add_scalar(
            tag='Diff/Diff. S-Q Loss',
            scalar_value=mean_loss_diff,
            global_step=epoch
        )

        writer.add_scalar(
            tag='Diff/Rel. Diff. S-Q Loss',
            scalar_value=mean_loss_rdiff,
            global_step=epoch
        )

        writer.flush()

    timer.end()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model hyperparameters
    parser.add_argument(
        '--config_file_path', default='./config/toy_maml.yaml', type=str)

    args = parser.parse_args()

    #* WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings('ignore', message=r'.*Named tensors.*')
    warnings.filterwarnings(
        'ignore', message=r'.*does not have many workers which may be a bottleneck.*')
    warnings.filterwarnings(
        'ignore', message=r'.*GPU available but not used .*')
    warnings.filterwarnings('ignore', message=r'.*shuffle=True')

    train(args)
