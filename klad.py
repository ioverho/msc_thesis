
import torch
import torch.nn.functional as F
import torch.optim as optim

import higher


from maml_test.sine import SampleSinuisoid
from maml_test.regressor  import SineRegressor

device = torch.device('cuda')

dataset = SampleSinuisoid(device)

model = SineRegressor(hidden_dim=40).to(device)

# ==============================================================================
# Hyperparameters
# ==============================================================================
hidden_dim = 40
epochs = 10000
meta_batches = 25
adaptation_steps = 1
K_support = 10
K_query = 10
K_test = 5
meta_lr = 0.001           # beta
adapt_lr = 0.01           # alpha
first_order_train = True
first_order_test = True
verbosity = 500

# ==============================================================================
# Init code
# ==============================================================================
device = torch.device('cuda')

dataset = SampleSinuisoid(device)

model = SineRegressor(hidden_dim=40).to(device)

meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
adapt_opt = optim.SGD(model.parameters(), lr=adapt_lr)

# ==============================================================================
# Train loop
# ==============================================================================
model.train()
model.thaw()

for epoch in range(epochs):
    # OUTER
    # This is a single episode
    # i.e. a set of tasks

    s_losses, q_losses = [], []
    for i in range(meta_batches):
        # INNER LOOP
        # This is a single task
        dataset.sample_task()

        # Aggregate the meta loss
        loss_meta = torch.tensor([0.0], dtype=torch.float, device=device)
        with higher.innerloop_ctx(model,
                                  adapt_opt,
                                  copy_initial_weights=False,
                                  track_higher_grads=first_order_train) as (model_episode, opt_episode):
            # Adaptation loop
            # Allowed to train on support for a single task
            for ii in range(adaptation_steps):

                # Sample a single (support) batch
                x_s, y_s = dataset.sample_batch(K_support)
                y_s_hat = model_episode(x_s)

                loss_s = F.mse_loss(y_s_hat, y_s, reduction='none')

                # Optimize on batch loss
                opt_episode.step(torch.mean(loss_s))

                if ii == 0:
                    # Record loss prior to adaption
                    rmse_s = torch.mean(torch.sqrt(loss_s.detach().cpu()))
                    s_losses.append(rmse_s)

            # Sample a single (query) batch
            x_q, y_q = dataset.sample_batch(K_query)
            y_q_hat = model_episode(x_q)

            loss_q = F.mse_loss(y_q_hat, y_q, reduction='none')

            # Add adaptation validation loss to meta loss
            loss_meta += torch.mean(loss_q)

            # Record adaptation validation loss post adaption
            # for comparison to pre-adaption
            rmse_q = torch.mean(torch.sqrt(loss_q.detach().cpu()))
            q_losses.append(rmse_q)

    # Average the meta loss over the number tasks
    loss_meta = loss_meta / meta_batches

    loss_meta.backward()

    # Optimize the meta-model
    meta_opt.step()
    meta_opt.zero_grad()

    # Boilerplate printing code
    if epoch % verbosity == 0:
        s_losses, q_losses = torch.stack(s_losses), torch.stack(q_losses)
        mean_loss_s, mean_loss_q = torch.mean(
            s_losses).item(), torch.mean(q_losses).item()
        mean_loss_diff = torch.mean(q_losses - s_losses).item()
        mean_loss_rdiff = (torch.mean((q_losses - s_losses) / s_losses)).item()
        s_losses, q_losses = [], []

        print(f"{epoch:04d}.{i:>02d} | RMSE: pre-adapt: {mean_loss_s:6.2e}, adapt: {mean_loss_q:6.2e}, diff: {mean_loss_diff:+6.2e}, rdiff: {mean_loss_rdiff:+3.2f}")

# %%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import higher

from maml_test.sine import SampleSinuisoid
from maml_test.regressor  import SineRegressor

n_test_tasks = 10000
verbosity = 250
K_support_test = 5
K_query_test = 5
adaptation_steps = 1
first_order_test = True

device = torch.device('cuda')

cpt = torch.load("./checkpoints/MAML_toy/version_3/checkpoint.pt", device)

dataset = SampleSinuisoid(device)

model = SineRegressor(hidden_dim=40).to(device)
model.load_state_dict(cpt['model'])
model.train()
model.thaw()

adapt_lr = 1e-2
adapt_opt = optim.SGD(model.parameters(), lr=adapt_lr)

s_losses, q_losses = [], []
for i in range(n_test_tasks):
    # INNER LOOP
    # This is a single task
    dataset.sample_task()

    model.train()

    # Aggregate the meta loss
    loss_meta = torch.tensor([0.0], dtype=torch.float, device=device)
    with higher.innerloop_ctx(model,
                              adapt_opt,
                              copy_initial_weights=False,
                              track_higher_grads=first_order_test) as (model_episode, opt_episode):
        # Adaptation loop
        # Allowed to train on support for a single task
        for ii in range(adaptation_steps):

            # Sample a single (support) batch
            x_s, y_s = dataset.sample_batch(K_support_test)
            y_s_hat = model_episode(x_s)

            loss_s = F.mse_loss(y_s_hat, y_s, reduction='none')

            # Optimize on batch loss
            opt_episode.step(torch.mean(loss_s))

            if ii == 0:
                # Record loss prior to adaption
                rmse_s = torch.mean(torch.sqrt(loss_s.detach().cpu()))
                s_losses.append(rmse_s)

        # Sample a single (query) batch
        x_q, y_q = dataset.sample_batch(K_query_test)
        y_q_hat = model_episode(x_q)

        loss_q = F.mse_loss(y_q_hat, y_q, reduction='none')

        # Add adaptation validation loss to meta loss
        loss_meta += torch.mean(loss_q)

        # Record adaptation validation loss post adaption
        # for comparison to pre-adaption
        rmse_q = torch.mean(torch.sqrt(loss_q.detach().cpu()))
        q_losses.append(rmse_q)

    if i % verbosity == 0 or i == n_test_tasks - 1:
        s_losses_, q_losses_ = torch.stack(s_losses), torch.stack(q_losses)
        mean_loss_s = torch.mean(s_losses_).item()
        mean_loss_q = torch.mean(q_losses_).item()
        mean_loss_diff = torch.mean(q_losses_ - s_losses_).item()
        mean_loss_rdiff = (torch.mean((q_losses_ - s_losses_) / s_losses_)).item()

        print(f"{i:04d}.{1:>02d} | RMSE: pre-adapt: {mean_loss_s:6.2e}, adapt: {mean_loss_q:6.2e}, diff: {mean_loss_diff:+6.2e}, rdiff: {mean_loss_rdiff:+3.2f}")

# %%
import os

import matplotlib
import matplotlib.pyplot as plt

max_updates = 25
n_save = 10

cmap = matplotlib.cm.get_cmap('viridis')

os.makedirs(os.path.split("./checkpoints/MAML_toy/version_3/checkpoint.pt")[0] +
            f"/figures", exist_ok=True)

for i_save in range(n_save):
    dataset.sample_task()

    x, y = dataset.sample_batch(K_support_test)

    fig_x = torch.linspace(-5, 5, 100, device=device)
    fig_y = dataset.transform(fig_x)
    plt.plot(fig_x.detach().cpu(), fig_y.detach().cpu().squeeze(0), alpha=1.0, c='k')
    plt.scatter(x.detach().cpu(), y.detach().cpu(), c='k')

    with torch.no_grad():
        y_hat = model(x)
        model_init_curve = model(fig_x.unsqueeze(1))

    plt.plot(fig_x.detach().cpu(), model_init_curve.detach().cpu(),
            alpha=0.8, c='k', ls='-.')

    with higher.innerloop_ctx(model,
                            adapt_opt,
                            copy_initial_weights=False,
                            track_higher_grads=first_order_test) as (model_episode, opt_episode):
        for i in range(max_updates):
            y_hat = model_episode(x)

            loss = F.mse_loss(y_hat, y)

            opt_episode.step(loss)

            if i % 5 == 0:
                with torch.no_grad():
                    model_episode_curve = model_episode(fig_x.unsqueeze(1))
                    y_hat_update = model_episode(x)

                plt.plot(fig_x.detach().cpu(), model_episode_curve.detach().cpu(),
                        c=cmap(5/(max_updates+5) + i/(max_updates+5)), alpha=0.8)

    plt.ylim(-5, 5)

    plt.savefig(os.path.split("./checkpoints/MAML_toy/version_3/checkpoint.pt")[0] +
                f"/figures/maml_regression__plot_{i_save}.png")
    plt.close()

# %%
