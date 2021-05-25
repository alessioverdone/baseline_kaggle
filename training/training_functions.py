from random import shuffle
from utils.utils_functions import augment, get_rewards, get_advantages_and_returns, compute_losses, get_features
from model.blocks import RLAgent
from tqdm.notebook import tqdm
import numpy as np
from kaggle_environments import evaluate
from dataset.dataset_class import GeeseDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def rollout(player, env, players, buffers, gammas, lambdas):
    rewards = {str(i + 1): [] for i in range(2)}
    values = {str(i + 1): [] for i in range(2)}
    # shuffle players indices
    shuffle(players)
    trainer = env.train(players)
    observation = trainer.reset()
    prev_obs = observation
    done = False
    prev_heads = [None for _ in range(4)]
    # start rollout
    while not done:
        # cache previous state
        for i, g in enumerate(observation['geese']):
            if len(g) > 0:
                prev_heads[i] = prev_obs['geese'][i][0]
        prev_obs = observation
        # transform observation to state
        state = get_features(observation, env.configuration, prev_heads)
        # make a move
        action, logp, v = player.raw_outputs(state)
        # observe
        observation, reward, done, _ = trainer.step(['NORTH', 'EAST', 'SOUTH', 'WEST'][action])
        # data -> buffers
        buffers['states'].append(state)
        buffers['actions'].append(action)
        buffers['log-p'].append(logp.cpu().detach())
        # save rewards and values
        r = get_rewards(reward, observation, prev_obs, done)
        for i in range(2):
            rewards[str(i + 1)].append(r[i])
            values[str(i + 1)].append(v[i])
    # save advantages and returns
    for key in ['1', '2']:
        advs, rets = get_advantages_and_returns(rewards[key], values[key], gammas[key], lambdas[key])
        # add them to buffer
        buffers['adv-' + key] += advs
        buffers['ret-' + key] += rets


def runner(net, env, samples_threshold, gammas, lambdas, progress_bar=False):
    data_buffers = {'states': [], 'actions': [], 'log-p': [],
                    'adv-1': [], 'ret-1': [],
                    'adv-2': [], 'ret-2': []}
    samples_collected = 0
    if progress_bar:
        samples_bar = tqdm(total=samples_threshold, desc='Collecting Samples', leave=False)
    player = RLAgent(net, stochastic=True)
    opponents = [RLAgent(net, stochastic=False) for _ in range(3)]
    while True:
        rollout(player, env, players=[None] + opponents, buffers=data_buffers,
                gammas=gammas, lambdas=lambdas)
        if progress_bar:
            # update progress bar
            samples_bar.update(len(data_buffers['states']) - samples_collected)
        samples_collected = len(data_buffers['states'])
        if samples_collected >= samples_threshold:
            if progress_bar:
                samples_bar.close()
            return data_buffers


def train(net, optimizer, env,
          n_episodes=25,
          batch_size=256,
          samples_threshold=10000,
          n_ppo_epochs=25):
    losses_hist = {'clip': [], 'value-1': [], 'value-2': [], 'ent': [], 'lr': []}
    win_rates = {'Score': [], 'Rank': []}
    gammas = {'1': 0.8, '2': 0.8}
    lambdas = {'1': 0.7, '2': 0.7}
    print('-Start Training')
    for episode in tqdm(range(n_episodes), desc='Episode', leave=False):
        # update statistics
        net.eval()
        player = RLAgent(net, stochastic=False)
        scores = evaluate("hungry_geese", [player] + ['greedy'] * 3, num_episodes=30)
        win_rates['Score'].append(np.mean([r[0] for r in scores]))
        win_rates['Rank'].append(np.mean([sum(r[0] <= r_ for r_ in r if r_ is not None) for r in scores]))
        # collect data
        buffers = runner(net, env, samples_threshold, gammas=gammas, lambdas=lambdas)
        # perform training
        net.train()
        dataset = GeeseDataset(buffers)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        for epoch in range(n_ppo_epochs):
            for batch in dataloader:
                losses = compute_losses(net, augment(batch),
                                        c1=1, c2=1, c_ent=0.01)
                loss = losses['actor']
                losses_hist['clip'].append(losses['actor'].item())
                for i in range(2):
                    loss += losses['critic'][i]
                    losses_hist[f'value-{i + 1}'].append(losses['critic'][i].item())
                loss += losses['entropy']
                losses_hist['ent'].append(losses['entropy'].item())
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                optimizer.step()
                optimizer.zero_grad()

        # !mkdir checkpoint
        torch.save(net.state_dict(), 'checkpoint/g.net')

    return win_rates
