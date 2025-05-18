import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pickle
from tqdm import trange, tqdm
from ndlinear import NdLinear

TensorBatch = List[torch.Tensor]

REF_MIN_SCORE = {
    'halfcheetah': -79.20,
    'hopper': 395.64,
    'walker2d': 0.20,
}

REF_MAX_SCORE = {
    'halfcheetah': 16584.93,
    'hopper': 4376.33,
    'walker2d': 6972.80,
}

@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "BC-Minari-0.1"
    # wandb run name
    name: str = "BC_0.1"
    # evaluation environment
    env: str = "halfcheetah"
    # dataset name
    dataset: str = "expert"
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # what top fraction of the dataset (sorted by return) to use
    frac: float = 0.1
    # maximum possible trajectory length
    max_traj_len: int = 1000
    # whether to normalize states
    normalize: bool = True
    # discount factor
    discount: float = 0.99
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 42
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{self.dataset}--ndlinear"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def get_normalised_returns(return_score, min_score, max_score):
    # Normalize the returns to be between 0 and 1
    norm_return = (return_score - min_score) / (max_score - min_score)
    return max(0,norm_return * 100)  # Scale to 100

def get_normalized_score(env_name: str, score: float) -> float:
    if env_name not in REF_MIN_SCORE or env_name not in REF_MAX_SCORE:
        raise ValueError(f"Environment {env_name} not supported for normalization.")
    min_score = REF_MIN_SCORE[env_name]
    max_score = REF_MAX_SCORE[env_name]
    return get_normalised_returns(score, min_score, max_score)
  
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def get_normalised_returns(return_score, min_score, max_score):
    # Normalize the returns to be between 0 and 1
    norm_return = (return_score - min_score) / (max_score - min_score)
    return max(0,norm_return * 100)  # Scale to 100

def get_normalized_score(env_name: str, score: float) -> float:
    if env_name not in REF_MIN_SCORE or env_name not in REF_MAX_SCORE:
        raise ValueError(f"Environment {env_name} not supported for normalization.")
    min_score = REF_MIN_SCORE[env_name]
    max_score = REF_MAX_SCORE[env_name]
    return get_normalised_returns(score, min_score, max_score)

def flatten_minari_dataset(minari_dataset: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenates Minari episodes into flat arrays (like D4RL format)."""
    flat_dataset = {}
    keys = ['observations', 'actions', 'rewards', 'terminations', 'truncations']
    for key in keys:
        flat_dataset[key] = np.concatenate([ep[key] for ep in minari_dataset], axis=0)

    # Compute next_observations (shifted observations)
    flat_dataset['next_observations'] = np.concatenate(
        [np.vstack([ep['observations'][1:], ep['observations'][-1:]]) for ep in minari_dataset],
        axis=0
    )

    return flat_dataset

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward / reward_scale

    env = gym.wrappers.TransformObservation(env, normalize_state, observation_space=env.observation_space)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_minari_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        
        # Combine terminations and truncations to create the "done" flag
        # An episode is considered done if it's either terminated or truncated
        dones = np.logical_or(data["terminations"], data["truncations"])
        self._dones[:n_transitions] = self._to_tensor(dones[..., None])
        
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        # In gymnasium, use reset with seed parameter instead of env.seed()
        env.reset(seed=seed)
        # And seed the action space directly
        env.action_space.seed(seed)
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.reset(seed=seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        # Gymnasium reset returns (observation, info)
        state, _ = env.reset()
        episode_reward = 0.0
        terminated, truncated = False, False
        
        # Episode ends when either terminated or truncated is True
        while not (terminated or truncated):
            action = actor.act(state, device)
            # Gymnasium step returns (observation, reward, terminated, truncated, info)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

def keep_best_trajectories_minari(
    dataset: Dict[str, np.ndarray],
    frac: float,
    discount: float,
    max_episode_steps: int = 1000,
):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0

    # "Done" = either terminated or truncated
    done_flags = np.logical_or(dataset["terminations"], dataset["truncations"])

    for i, (reward, done) in enumerate(zip(dataset["rewards"], done_flags)):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0

    # Rank trajectories by return (descending)
    sort_ord = np.argsort(returns)[::-1]
    top_trajs = sort_ord[: max(1, int(frac * len(sort_ord)))]

    # Flatten selected indices
    order = []
    for i in top_trajs:
        order += ids_by_trajectories[i]
    order = np.array(order)

    # Apply ordering to all keys
    keys = ['observations', 'actions', 'rewards', 'terminations', 'truncations', 'next_observations']
    for key in keys:
        dataset[key] = dataset[key][order]


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            # NdLinear(input_dims=(state_dim,), hidden_size=(hidden_dim,))
            NdLinear(input_dims=(state_dim,), hidden_size=(256,)),
            nn.ReLU(),
            NdLinear(input_dims=(256,), hidden_size=(256,)),
            nn.ReLU(),
            NdLinear(input_dims=(256,), hidden_size=(action_dim,)),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class BC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cuda",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]

@pyrallis.wrap()
def train(config: TrainConfig):
    if config.env == "halfcheetah":
        env = gym.make("HalfCheetah-v5")
        scale = 1000
    
    elif config.env == "walker2d":
        env = gym.make("Walker2d-v5")
        scale = 1000
    
    elif config.env == "hopper":
        env = gym.make("Hopper-v5")
        scale = 1000
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    BASE_DIR = os.path.join(os.path.dirname(__file__), 'data')
    if not config.env.startswith("mujoco_"):
        dataset_env_name = f"mujoco_{config.env}"
    else:
        dataset_env_name = config.env
        
    print(f"Loading dataset for {dataset_env_name} with dataset {config.dataset}")
    dataset_path = f"{BASE_DIR}/{dataset_env_name}_{config.dataset}-v0.pkl"
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    
    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)
    
    print('=' * 50)
    print(f'Starting new experiment: {config.env} {config.dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    basic_env_name = config.env.replace('mujoco_', '')
    
    if basic_env_name in REF_MIN_SCORE:
        norm_returns = get_normalised_returns(np.mean(returns), REF_MIN_SCORE[basic_env_name], REF_MAX_SCORE[basic_env_name])
        print(f'Normalised return: {norm_returns:.2f}')
        print(f'Normalised max return: {get_normalised_returns(np.max(returns), REF_MIN_SCORE[basic_env_name], REF_MAX_SCORE[basic_env_name]):.2f}')
        print(f'Normalised min return: {get_normalised_returns(np.min(returns), REF_MIN_SCORE[basic_env_name], REF_MAX_SCORE[basic_env_name]):.2f}')
        print(f'Normalised std: {get_normalised_returns(np.std(returns), REF_MIN_SCORE[basic_env_name], REF_MAX_SCORE[basic_env_name]):.2f}')
        print('=' * 50)
    else:
        print('No reference scores found for this environment.')
        print('=' * 50)

    # Get the flattened dataset
    dataset = flatten_minari_dataset(trajectories)
    
    keep_best_trajectories_minari(dataset, config.frac, config.discount)
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    
    replay_buffer.load_minari_dataset(dataset)
    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    max_action = float(env.action_space.high[0])
    
    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    
    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    
    # Initialize policy
    trainer = BC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))
    
    evaluations = []
    pbar = trange(int(config.max_timesteps), desc="Training", ncols=80)
    for t in pbar:
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)

        if (t + 1) % config.eval_freq == 0:
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = get_normalized_score(config.env, eval_score) * 100.0
            evaluations.append(normalized_eval_score)

            tqdm.write("---------------------------------------")
            tqdm.write(
                f"Time steps: {t + 1}\n"
                f"Evaluation over {config.n_episodes} episodes: "
                f"mean: {eval_score:.2f}, std: {eval_scores.std():.2f}"
            )
            tqdm.write("---------------------------------------")

            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log({"Minari_normalized_score": normalized_eval_score}, step=trainer.total_it)

    pbar.close()


if __name__ == "__main__":
    train()
