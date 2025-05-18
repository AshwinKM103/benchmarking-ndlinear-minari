import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

from ndlinear import NdLinear
import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional
import wandb
from tqdm import tqdm

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
    group: str = "AWAC-Minari"
    # wandb run name
    name: str = "AWAC"
    # evaluation environment
    env: str = "halfcheetah"
    # dataset name
    dataset: str = "medium"
    # actor and critic hidden dim
    hidden_dim: int = 256
    # actor and critic learning rate
    learning_rate: float = 3e-4
    # discount factor
    gamma: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 5e-3
    # awac actor loss temperature, controlling balance
    # between behaviour cloning and Q-value maximization
    awac_lambda: float = 1.0
    # total number of gradient updated during training
    num_train_ops: int = 1_000_000
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    # evaluation frequency, will evaluate every eval_frequency
    # training steps
    eval_frequency: int = 50000
    # number of episodes to run during evaluation
    n_test_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    seed: int = 42
    # evaluation random seed
    test_seed: int = 69
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}--{self.dataset}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

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
        device: str = "cpu",
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

class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        min_action: float = -1.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self._mlp = nn.Sequential(
            NdLinear(input_dims=(state_dim,), hidden_size=(hidden_dim,)),
            nn.ReLU(),
            NdLinear(input_dims=(hidden_dim,), hidden_size=(hidden_dim,)),
            nn.ReLU(),
            NdLinear(input_dims=(hidden_dim,), hidden_size=(hidden_dim,)),
            nn.ReLU(),
            NdLinear(input_dims=(hidden_dim,), hidden_size=(action_dim,)),
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

    def _get_policy(self, state: torch.Tensor) -> torch.distributions.Distribution:
        mean = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        action = action_t[0].cpu().numpy()
        return action

class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        
        self._mlp = nn.Sequential(
            NdLinear(input_dims=(state_dim + action_dim,), hidden_size=(hidden_dim,)),
            nn.ReLU(),
            NdLinear(input_dims=(hidden_dim,), hidden_size=(hidden_dim,)),
            nn.ReLU(),
            NdLinear(input_dims=(hidden_dim,), hidden_size=(hidden_dim,)),
            nn.ReLU(),
            NdLinear(input_dims=(hidden_dim,), hidden_size=(1,)),  # Output scalar
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, action], dim=-1))
        return q_value


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class AdvantageWeightedActorCritic:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 5e-3,  # parameter for the soft target update,
        awac_lambda: float = 1.0,
        exp_adv_max: float = 100.0,
    ):
        self._actor = actor
        self._actor_optimizer = actor_optimizer

        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)

        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)

        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_max

    def _actor_loss(self, states, actions):
        with torch.no_grad():
            pi_action, _ = self._actor(states)
            v = torch.min(
                self._critic_1(states, pi_action), self._critic_2(states, pi_action)
            )

            q = torch.min(
                self._critic_1(states, actions), self._critic_2(states, actions)
            )
            adv = q - v
            weights = torch.clamp_max(
                torch.exp(adv / self._awac_lambda), self._exp_adv_max
            )

        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _critic_loss(self, states, actions, rewards, dones, next_states):
        with torch.no_grad():
            next_actions, _ = self._actor(next_states)

            q_next = torch.min(
                self._target_critic_1(next_states, next_actions),
                self._target_critic_2(next_states, next_actions),
            )
            q_target = rewards + self._gamma * (1.0 - dones) * q_next

        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)

        q1_loss = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = q1_loss + q2_loss
        return loss

    def _update_critic(self, states, actions, rewards, dones, next_states):
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)

        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)

        result = {"critic_loss": critic_loss, "actor_loss": actor_loss}
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_2": self._critic_2.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict["actor"])
        self._critic_1.load_state_dict(state_dict["critic_1"])
        self._critic_2.load_state_dict(state_dict["critic_2"])

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

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

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
        
        print(f"Episode reward: {episode_reward}")
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

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
    
    state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
        
    # Normalize the dataset
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

    actor_critic_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim": config.hidden_dim,
    }

    actor = Actor(**actor_critic_kwargs)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(config.device)
    critic_2.to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.learning_rate)

    awac = AdvantageWeightedActorCritic(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=critic_1,
        critic_1_optimizer=critic_1_optimizer,
        critic_2=critic_2,
        critic_2_optimizer=critic_2_optimizer,
        gamma=config.gamma,
        tau=config.tau,
        awac_lambda=config.awac_lambda,
    )
    wandb_init(asdict(config))
    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    
    evaluations = []
    pbar = tqdm(total=config.num_train_ops, ncols=80, dynamic_ncols=True, leave=True)
    for t in range(config.num_train_ops):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        update_result = awac.update(batch)
        # wandb.log(update_result, step=t)

        if (t + 1) % config.eval_frequency == 0:
            print(f"\nTime steps: {t + 1}")   # <-- Use print() not tqdm.write()
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_test_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = get_normalized_score(config.env, eval_score) * 100.0
            evaluations.append(normalized_eval_score)

            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_test_episodes} episodes: "
                f"{eval_score:.3f} , " f"Std Deviation: {eval_scores.std():.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    awac.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log({"eval_score": eval_scores.mean()}, step=t)

        pbar.update(1)

    pbar.close()
if __name__ == "__main__":
    train()