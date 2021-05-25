import itertools
import numpy as np
import torch
from torch.distributions import Categorical


def augment(batch):
    # random horizontal flip
    flip_mask = np.random.rand(len(batch['states'])) < 0.5
    batch['states'][flip_mask] = batch['states'][flip_mask].flip(-1)
    batch['actions'][flip_mask] = torch.where(batch['actions'][flip_mask] > 0, 4 - batch['actions'][flip_mask],
                                              0)  # 1 -> 3, 3 -> 1

    # random vertical flip (and also diagonal)
    flip_mask = np.random.rand(len(batch['states'])) < 0.5
    batch['states'][flip_mask] = batch['states'][flip_mask].flip(-2)
    batch['actions'][flip_mask] = torch.where(batch['actions'][flip_mask] < 3, 2 - batch['actions'][flip_mask],
                                              3)  # 0 -> 2, 2 -> 0

    # shuffle opponents channels
    permuted_axs = list(itertools.permutations([0, 1, 2]))
    permutations = [torch.tensor(permuted_axs[i]) for i in np.random.randint(6, size=len(batch['states']))]
    for i, p in enumerate(permutations):
        shuffled_channels = torch.zeros(3, batch['states'].shape[2], batch['states'].shape[3])
        shuffled_channels[p] = batch['states'][i, 1:4]
        batch['states'][:, 1:4] = shuffled_channels
    return batch


def get_rank(obs, prev_obs):
    geese = obs['geese']
    index = obs['index']
    player_len = len(geese[index])
    survivors = [i for i in range(len(geese)) if len(geese[i]) > 0]
    if index in survivors:  # if our player survived in the end, its rank is given by its length in the last state
        return sum(len(x) >= player_len for x in geese)  # 1 is the best, 4 is the worst
    # if our player is dead, consider lengths in penultimate state
    geese = prev_obs['geese']
    index = prev_obs['index']
    player_len = len(geese[index])
    rank_among_lost = sum(len(x) >= player_len for i, x in enumerate(geese) if i not in survivors)
    return rank_among_lost + len(survivors)


def get_rewards(env_reward, obs, prev_obs, done):
    geese = prev_obs['geese']
    index = prev_obs['index']
    step = prev_obs['step']
    if done:
        rank = get_rank(obs, prev_obs)
        r1 = (1, -0.25, -0.75, -1)[rank - 1]
        died_from_hunger = ((step + 1) % 40 == 0) and (len(geese[index]) == 1)
        r2 = -1 if died_from_hunger else 0  # int(rank == 1) # huge penalty for dying from hunger and huge award for the win
    else:
        if step == 0:
            env_reward -= 1  # somehow initial step is a special case
        r1 = 0
        r2 = max(0.1 * (env_reward - 1), 0)  # food reward
    return r1, r2


def inv_discount_cumsum(array, discount_factor):
    res = [array[-1]]
    for x in torch.flip(array, dims=[0])[1:]:
        res.append(discount_factor * res[-1] + x)
    return torch.flip(torch.stack(res), dims=[0])


def get_advantages_and_returns(rewards, values, gamma, lam):
    # lists -> tensors
    rewards = torch.tensor(rewards, dtype=torch.float)
    values = torch.tensor(values + [0.])
    # calculate deltas, A and R
    deltas = rewards + gamma * values[1:] - values[:-1]
    advs = inv_discount_cumsum(deltas, gamma * lam).cpu().detach().tolist()
    rets = inv_discount_cumsum(rewards, gamma).cpu().detach().tolist()
    return advs, rets


def compute_losses(net, data, c1, c2, c_ent, clip_ratio=0.2):
    # move data to GPU
    states = data['states'].cuda()
    actions = data['actions'].cuda()
    logp_old = data['log-p'].cuda()
    returns = [data[f'ret-{i}'].float().cuda() for i in range(1, 3)]
    advs = data['adv-1'].float().cuda()
    advs += data['adv-2'].float().cuda()
    # get network outputs
    logp_dist, (values_1, values_2) = net(states)
    logp = torch.stack([lp[a] for lp, a in zip(logp_dist, actions)])
    # compute actor loss
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advs
    actor_loss = -(torch.min(ratio * advs, clip_adv)).mean()
    # critic losses
    critic_loss_1 = ((values_1 - returns[0]) ** 2).mean()
    critic_loss_2 = ((values_2 - returns[1]) ** 2).mean()
    # entropy loss
    entropy = Categorical(probs=torch.exp(logp_dist)).entropy()
    entropy[entropy != entropy] = torch.tensor(0.).cuda()  # remove NaNs if any
    entropy_loss = -entropy.mean()
    return {'actor': actor_loss,
            'critic': (c1 * critic_loss_1,
                       c2 * critic_loss_2),
            'entropy': c_ent * entropy_loss}


def get_position_from_index(index, columns):
    row = index // columns
    col = index % columns
    return row, col


def get_index_from_position(row, col, columns):
    return row * columns + col


def find_new_head_position(head_row, head_col, action, rows, columns):
    if action == 0: # north
        new_row, new_col = (head_row + rows - 1) % rows, head_col
    elif action == 1: # east
        new_row, new_col = head_row, (head_col + 1) % columns
    elif action == 2: # south
        new_row, new_col = (head_row + 1) % rows, head_col
    else: # west
        new_row, new_col = head_row, (head_col + columns - 1) % columns
    return new_row, new_col


def shift_head(head_id, action, rows=7, columns=11):
    head_row, head_col = get_position_from_index(head_id, columns)
    new_row, new_col = find_new_head_position(head_row, head_col, action, rows, columns)
    new_head_id = get_index_from_position(new_row, new_col, columns)
    return new_head_id


def get_previous_head(ids, last_action, rows, columns):
    if len(ids) > 1:
        return ids[1]
    return shift_head(ids[0], (last_action + 2) % 4, rows, columns)


def ids2locations(ids, prev_head, step, rows, columns):
    state = np.zeros((4, rows * columns))
    if len(ids) == 0:
        return state
    state[0, ids[0]] = 1 # goose head
    if len(ids) > 1:
        state[1, ids[1:-1]] = 1 # goose body
        state[2, ids[-1]] = 1 # goose tail
    if step != 0:
        state[3, prev_head] = 1 # goose head one step before
    return state


def get_features(observation, config, prev_heads):
    rows, columns = config['rows'], config['columns']
    geese = observation['geese']
    index = observation['index']
    step = observation['step']
    # convert indices to locations
    locations = np.zeros((len(geese), 4, rows * columns))
    for i, g in enumerate(geese):
        locations[i] = ids2locations(g, prev_heads[i], step, rows, columns)
    if index != 0: # swap rows for player locations to be in first channel
        locations[[0, index]] = locations[[index, 0]]
    # put locations into features
    features = np.zeros((12, rows * columns))
    for k in range(4):
        features[k] = np.sum(locations[k][:3], 0)
        features[k + 4] = np.sum(locations[:, k], 0)
    features[-4, observation['food']] = 1 # food channel
    features[-3, :] = (step % config['hunger_rate']) / config['hunger_rate'] # hunger danger channel
    features[-2, :] = step / config['episodeSteps'] # timesteps channel
    features[-1, :] = float((step + 1) % config['hunger_rate'] == 0) # hunger milestone indicator
    features = torch.Tensor(features).reshape(-1, rows, columns)
    # roll
    head_id = geese[index][0]
    head_row = head_id // columns
    head_col = head_id % columns
    features = torch.roll(features, ((rows // 2) - head_row, (columns // 2) - head_col), dims=(-2, -1))
    return features

