# %%
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

import matplotlib
import matplotlib.pyplot as plt

import higher


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

dataset = SampleSinuisoid()
model = SinusoidRegressor(hidden_dim)

meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
adapt_opt = optim.SGD(model.parameters(), lr=adapt_lr)

# %%
import seaborn as sns

dataset = SampleSinuisoid(amp_params=(6,2), phase_params=(0.3,0.4))

params = []
for i in range(10000):
    dataset.sample_task()
    params.append([dataset.cur_amplitude.item(), dataset.cur_phase.item()])

params = np.stack(params)

sns.jointplot(x=params[:, 0], y=params[:, 1], kind='hex', bins=250)

plt.xlim(*dataset.amplitude_range)
plt.ylim(*dataset.phase_range)

# %%

init_params = []
adapt_params = []
actual_params = []
for i in range(meta_batches):
    # INNER LOOP
    # This is a single task
    dataset.sample_task()

    # Aggregate the meta loss
    loss_meta = torch.FloatTensor([0.0])
    with higher.innerloop_ctx(model,
                                adapt_opt,
                                copy_initial_weights=False,
                                track_higher_grads=first_order_train) as (model_episode, opt_episode):
        # Adaptation loop
        # Allowed to train on support for a single task
        for ii in range(adaptation_steps):

            x_s, y_s = dataset.sample_batch(K_support)
            y_s_hat, amp_hat, phase_hat = model_episode(x_s, dataset)

            loss_s = F.mse_loss(y_s_hat, y_s, reduction='none')

            opt_episode.step(torch.mean(loss_s))

            init_params.append([amp_hat.detach(), phase_hat.detach()])

            with torch.no_grad():
                adapt_params.append([*model_episode._get_params(x_s, dataset)])

            actual_params.append([dataset.cur_amplitude, dataset.cur_phase])

# %%

def llt_to_array(llt):
    return np.vectorize(lambda x: x.item())(np.stack(llt))

init_arr = llt_to_array(init_params)[:10]
adapt_arr = llt_to_array(adapt_params)[:10]
actual_arr = llt_to_array(actual_params)[:10]

plt.scatter(init_arr[:,0], init_arr[:,1])
plt.scatter(adapt_arr[:, 0], adapt_arr[:, 1])
plt.scatter(actual_arr[:, 0], actual_arr[:, 1])

for adapt_pt, actual_pt in zip(adapt_arr, actual_arr):
    plt.plot(
        [adapt_pt[0], actual_pt[0]],
        [adapt_pt[1], actual_pt[1]],
        c='k',
        zorder=0
        )

plt.xlim(*dataset.amplitude_range)
plt.ylim(*dataset.phase_range)

plt.show()

# %%

model.train()

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
        loss_meta = torch.FloatTensor([0.0])
        with higher.innerloop_ctx(model,
                                  adapt_opt,
                                  copy_initial_weights=False,
                                  track_higher_grads=first_order_train) as (model_episode, opt_episode):
            # Adaptation loop
            # Allowed to train on support for a single task
            for ii in range(adaptation_steps):

                x_s, y_s = dataset.sample_batch(K_support)
                y_s_hat = model_episode(x_s, dataset)

                loss_s = F.mse_loss(y_s_hat, y_s, reduction='none')

                opt_episode.step(torch.mean(loss_s))

                if ii == 0:
                    s_losses.append(torch.mean(torch.sqrt(loss_s.detach())))

            x_q, y_q = dataset.sample_batch(K_query)
            y_q_hat = model_episode(x_q, dataset)

            loss_q = F.mse_loss(y_q_hat, y_q, reduction='none')

            loss_meta += torch.mean(loss_q)

            q_losses.append(torch.mean(torch.sqrt(loss_q.detach())))

    loss_meta = loss_meta / meta_batches

    loss_meta.backward()

    meta_opt.step()
    meta_opt.zero_grad()

    if epoch % verbosity == 0:
        s_losses, q_losses = torch.stack(s_losses), torch.stack(q_losses)
        mean_loss_s, mean_loss_q = torch.mean(
            s_losses).item(), torch.mean(q_losses).item()
        mean_loss_diff = torch.mean(q_losses - s_losses).item()
        mean_loss_rdiff = (torch.mean((q_losses - s_losses) / s_losses)).item()
        s_losses, q_losses = [], []

        print(f"{epoch:04d}.{i:>02d} | RMSE: pre-adapt: {mean_loss_s:6.2e}, adapt: {mean_loss_q:6.2e}, diff: {mean_loss_diff:+6.2e}, rdiff: {mean_loss_rdiff:+3.2f}")

# %%



#%%

model.train()

meta_opt = optim.Adam(model.parameters(), lr=meta_lr)
adapt_opt = optim.SGD(model.parameters(), lr=adapt_lr)

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
        loss_meta = torch.FloatTensor([0.0])
        with higher.innerloop_ctx(model,
                                  adapt_opt,
                                  copy_initial_weights=False,
                                  track_higher_grads=first_order_train) as (model_episode, opt_episode):
            # Adaptation loop
            # Allowed to train on support for a single task
            for ii in range(adaptation_steps):

                x_s, y_s = dataset.sample_batch(K_support)
                y_s_hat = model_episode(x_s)

                loss_s = F.mse_loss(y_s_hat, y_s, reduction='none')

                opt_episode.step(torch.mean(loss_s))

                if ii == 0:
                    s_losses.append(torch.mean(torch.sqrt(loss_s.detach())))

            x_q, y_q = dataset.sample_batch(K_query)
            y_q_hat = model_episode(x_q)

            loss_q = F.mse_loss(y_q_hat, y_q, reduction='none')

            loss_meta += torch.mean(loss_q)

            q_losses.append(torch.mean(torch.sqrt(loss_q.detach())))

    loss_meta = loss_meta / meta_batches

    loss_meta.backward()

    meta_opt.step()
    meta_opt.zero_grad()

    if epoch % verbosity == 0:
        s_losses, q_losses = torch.stack(s_losses), torch.stack(q_losses)
        mean_loss_s, mean_loss_q = torch.mean(s_losses).item(), torch.mean(q_losses).item()
        mean_loss_diff = torch.mean(q_losses - s_losses).item()
        mean_loss_rdiff = (torch.mean((q_losses - s_losses) / s_losses)).item()
        s_losses, q_losses = [], []

        print(f"{epoch:04d}.{i:>02d} | RMSE: pre-adapt: {mean_loss_s:6.2e}, adapt: {mean_loss_q:6.2e}, diff: {mean_loss_diff:+6.2e}, rdiff: {mean_loss_rdiff:+3.2f}")

# %%

max_updates = 25

cmap = matplotlib.cm.get_cmap('Oranges')

dataset.sample_task()

x, y = dataset.sample_batch(K_test)

fig_x = torch.linspace(-5, 5, 100)
fig_y = dataset.transform(fig_x)
plt.plot(fig_x, fig_y, c='k')
plt.scatter(x, y, c='k')

with torch.no_grad():
    y_hat = model(x, dataset)
    model_init_curve = model(fig_x.unsqueeze(1), dataset)

plt.plot(fig_x, model_init_curve, c=cmap(1/max_updates), alpha=0.8, linewidth=0.5)

with higher.innerloop_ctx(model,
                          adapt_opt,
                          copy_initial_weights=False,
                          track_higher_grads=first_order_test) as (model_episode, opt_episode):
    for i in range(max_updates):
        y_hat = model_episode(x, dataset)

        loss = F.mse_loss(y_hat, y)

        opt_episode.step(loss)

        if i % 5 == 0:
            with torch.no_grad():
                model_episode_curve = model_episode(fig_x.unsqueeze(1), dataset)
                y_hat_update = model_episode(x, dataset)

            plt.plot(fig_x, model_episode_curve, c=cmap(2/max_updates + i/max_updates), alpha=0.5)

plt.ylim(-5,5)

# %%
x, y = dataset.sample_batch(K_test)

with torch.no_grad():
        h = model.encode_net(x)

        h = torch.mean(h, dim=0).unsqueeze(0)

        p = model.params_net(h)

        amp_hat, phase_hat = p[:, 0], p[:, 1]

        amp_hat, phase_hat = dataset._scale_params(amp_hat, phase_hat)

print(amp_hat, dataset.cur_amplitude, phase_hat, dataset.cur_phase)

with higher.innerloop_ctx(model,
                          adapt_opt,
                          copy_initial_weights=False,
                          track_higher_grads=first_order_test) as (model_episode, opt_episode):
    for i in range(max_updates):
        y_hat = model_episode(x, dataset)

        loss = F.mse_loss(y_hat, y)

        opt_episode.step(loss)

        with torch.no_grad():
            h = model_episode.encode_net(x)

            h = torch.mean(h, dim=0).unsqueeze(0)

            p = model_episode.params_net(h)

            amp_hat, phase_hat = p[:, 0], p[:, 1]

            amp_hat, phase_hat = dataset._scale_params(amp_hat, phase_hat)

        print(f"{loss.item():.2f}", amp_hat, dataset.cur_amplitude, phase_hat, dataset.cur_phase)

# %%
