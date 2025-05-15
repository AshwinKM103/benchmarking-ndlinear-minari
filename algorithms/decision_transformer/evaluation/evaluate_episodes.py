import numpy as np
import torch
import imageio


def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
        render=False,
        frame_collector=None
):

    model.eval()
    model.to(device=device)
    

    state_mean_tensor = torch.from_numpy(state_mean).to(device=device)
    state_std_tensor = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return_tensor = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
    
    k_buffer_size = min(20, max_ep_len)
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        states_input = states[-k_buffer_size:] if states.shape[0] >= k_buffer_size else states
        actions_input = actions[-k_buffer_size:] if actions.shape[0] >= k_buffer_size else actions
        rewards_input = rewards[-k_buffer_size:] if rewards.shape[0] >= k_buffer_size else rewards
        target_return_input = target_return_tensor[:, -k_buffer_size:] if target_return_tensor.shape[1] >= k_buffer_size else target_return_tensor


        action = model.get_action(
            (states_input.to(dtype=torch.float32) - state_mean_tensor) / state_std_tensor,
            actions_input.to(dtype=torch.float32),
            rewards_input.to(dtype=torch.float32),
            target_return_input.to(dtype=torch.float32), 
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combine termination and truncation flags
        
        if render and frame_collector is not None:
            frame = env.render()
            frame_collector.append(frame)


        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward


        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        render=False,
        frame_collector=None
):
    model.eval()
    model.to(device=device)

    state_mean_tensor = torch.from_numpy(state_mean).to(device=device)
    state_std_tensor = torch.from_numpy(state_std).to(device=device)

    state, _ = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # Capture the first frame if rendering
    if render and frame_collector is not None:
        frame = env.render()
        frame_collector.append(frame)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    # Add a small epsilon to ensure the agent doesn't stop prematurely
    ep_return = target_return + 0.05  # Slightly higher target to avoid early stopping
    target_return_tensor = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    
    # Maintain a buffer of recent states for more consistent action prediction
    # This helps stabilize the transformer's predictions
    k_buffer_size = min(20, max_ep_len)  # Match the context length K used in training
    
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        # Only use the last k_buffer_size states/actions/rewards for prediction
        # This better matches the training context window
        states_input = states[-k_buffer_size:] if states.shape[0] >= k_buffer_size else states
        actions_input = actions[-k_buffer_size:] if actions.shape[0] >= k_buffer_size else actions
        rewards_input = rewards[-k_buffer_size:] if rewards.shape[0] >= k_buffer_size else rewards
        
        # Replicate target_return_tensor and timesteps for consistency
        # These should have same sequence length as the states
        target_return_input = target_return_tensor[:, -k_buffer_size:] if target_return_tensor.shape[1] >= k_buffer_size else target_return_tensor
        timesteps_input = timesteps[:, -k_buffer_size:] if timesteps.shape[1] >= k_buffer_size else timesteps

        action = model.get_action(
            (states_input.to(dtype=torch.float32) - state_mean_tensor) / state_std_tensor,
            actions_input.to(dtype=torch.float32),
            rewards_input.to(dtype=torch.float32),
            target_return_input.to(dtype=torch.float32),
            timesteps_input.to(dtype=torch.long),
        )
        
        actions[-1] = action
        action_np = action.detach().cpu().numpy()

        state, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated
        
        # Capture frame after taking action if rendering
        if render and frame_collector is not None:
            frame = env.render()
            frame_collector.append(frame)
        

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward


        # More stable return-to-go updates
        if mode != 'delayed':
            pred_return = target_return_tensor[0, -1] - (reward / scale) + 0.001
            pred_return = torch.clamp(pred_return, min=0.0)
        else:
            pred_return = target_return_tensor[0, -1]

        target_return_tensor = torch.cat(
            [target_return_tensor, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        # Additional check to avoid early termination on stalled progress
        if done:
            break

    return episode_return, episode_length
