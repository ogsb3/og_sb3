import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import yaml
import argparse
from typing import Dict, Any, List
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch.multiprocessing as mp
from datetime import datetime
import torch
import logging
from pathlib import Path


def make_env(env_id: str, rank: int, seed: int = 0, monitor_dir: str = None, atari: bool = False):
    def _init():
        env = AtariWrapper(gym.make(env_id)) if atari else gym.make(env_id)
        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        env.reset(seed=seed+rank)
        return env
    return _init

def setup_env(config: Dict[str, Any], seed: int, monitor_dir: str = None):
    env_id = config['env']['id']
    n_envs = config['env']['n_envs']
    atari = config['env']['atari']
    
    env = DummyVecEnv([
        make_env(env_id, i, seed, monitor_dir, atari) 
        for i in range(n_envs)
    ])

    stack = config['env']['n_stack']
    if stack != 0 :
        env = VecFrameStack(env, n_stack=stack)
    
    return env

def setup_logger(log_dir: str):
    """Setup logging to both file and stdout"""
    log_file = os.path.join(log_dir, 'train_log.txt')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def train_single_run(
    config_name: str,
    config: Dict[str, Any],
    gpu_id: int,
    seed: int,
    run_dir: str
):
    # Set the torch device according to gpu_id
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    print(f"Starting training for {config_name} on GPU {device} with seed {seed}")

    # Setup environment
    train_monitor_dir = os.path.join(run_dir, 'monitor', 'train')
    eval_monitor_dir = os.path.join(run_dir, 'monitor', 'eval')
    os.makedirs(train_monitor_dir, exist_ok=True)
    os.makedirs(eval_monitor_dir, exist_ok=True)

    env = setup_env(config, seed, train_monitor_dir)
    eval_env = setup_env(config, seed + 1000, eval_monitor_dir)
    eval_env = VecTransposeImage(eval_env) if config['env']['atari'] else eval_env
    

    actual_eval_freq = config['logging']['eval_freq'] // config['env'].get('n_envs', 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=actual_eval_freq,
        n_eval_episodes=config['logging']['n_eval_episodes'],
        deterministic=True,
        verbose=config['logging'].get('verbose', 0),
        #tag=f"{config['logging'].get('run_tag', 'default')}_{config_name}_{seed}"
    )

    if config['logging']['model_save_freq'] != 0:
        actual_model_save_freq = config['logging']['model_save_freq'] // config['env'].get('n_envs', 1) 
        checkpoint_callback = CheckpointCallback(
            save_freq=actual_model_save_freq,
            save_replay_buffer=config['logging']['save_buffer'],
            save_path=run_dir,
            name_prefix="check_dip",
        )

        callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    else: callbacks = eval_callback

    
    # Configure SB3 logger with specified formats
    new_logger = configure(
        folder=run_dir,
        format_strings=config['logging']['format']
    )
    
    # Initialize algorithm
    algo_type = config[config_name]['algorithm']
    algo_class = {"dqn": DQN}[algo_type]

    tag = f"{config['logging'].get('run_tag', 'default')}_{config_name}_{seed}"

    model_params = config[config_name]['params']

    # Only add EvalCallback if the algorithm is not DGPI
    # callbacks = [] #metrics_callback
    # if algo_type != "dgpi":
    #     actual_eval_freq = config['logging']['eval_freq'] // config['env'].get('n_envs', 1)
    #     eval_callback = EvalCallback(
    #         eval_env,
    #         best_model_save_path=run_dir,
    #         log_path=run_dir,
    #         eval_freq=actual_eval_freq,
    #         n_eval_episodes=config['logging']['n_eval_episodes'],
    #         deterministic=True,
    #         verbose=config['logging'].get('verbose', 0),
    #         #tag=tag
    #     )
    #     callbacks.append(eval_callback)
    # else:
    #     model_params['tag'] = tag
    #     model_params['eval_env'] = eval_env


    model = algo_class(
        policy="CnnPolicy" if config['env']['atari'] else "MlpPolicy",
        env=env,
        verbose=config['logging'].get('verbose', 0),
        seed=seed,
        device=device,
        **config[config_name]['params']
    )
    model.set_logger(new_logger)
    
    logger = logging.getLogger()
 

 
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            progress_bar=config['logging'].get('progress_bar', False)
        )
        
        # Save final model
        #final_model_path = os.path.join(run_dir, "final_model")
        #model.save(final_model_path)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    else:
        logger.info(f"Training completed successfully for {config_name} seed {seed} on GPU {device}")
    
    finally:
        env.close()
        eval_env.close()

def get_gpu_allocation(n_gpus: int, config_names: List[str]) -> List[int]:
    """Distribute configurations across available GPUs in an alternating fashion."""
    if n_gpus == 0:
        return [-1] * len(config_names)  # -1 indicates CPU
    
    if n_gpus == 1:
        return [0] * len(config_names)
    
    # Alternate between GPUs for better distribution
    gpu_assignments = []
    for i in range(len(config_names)):
        gpu_id = i % n_gpus
        gpu_assignments.append(gpu_id)
    
    return gpu_assignments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging directory and logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = config['logging'].get('run_tag', 'default')
    log_dir = os.path.join(
        config['logging']['log_dir'], 
        f"{run_tag}_run_{timestamp}"
    )
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)
    
    # GPU setup
    n_gpus = min(config['hardware']['n_gpus'], torch.cuda.device_count()) if torch.cuda.is_available() else 0
    devices = get_gpu_allocation(n_gpus, config['training']['configurations'])
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup multiprocessing
    mp.set_start_method('spawn')
    max_processes = config['hardware']['max_processes']

    # Generate all training configurations
    training_configs = []
    common_settings = config.get('common_settings', {})
    
    # First, create all configurations without GPU assignments
    all_configs = []
    for config_name in config['training']['configurations']:
        local_config = config.get(config_name, {}).copy()
        
        # Update local config with common settings if not specified
        for key, value in common_settings.items():
            if key not in local_config['params']:
                local_config['params'][key] = value
        
        # Update the global config with the modified local config
        config[config_name] = local_config
        
        for seed in config['training']['seeds']:
            all_configs.append((config_name, seed))

    # Get GPU assignments for all runs
    devices = get_gpu_allocation(n_gpus, all_configs)
    
    # Create final training configurations with GPU assignments
    for idx, ((config_name, seed), device) in enumerate(zip(all_configs, devices)):
        run_dir = os.path.join(log_dir, f"{config_name}_run_{idx}")
        os.makedirs(run_dir, exist_ok=True)
        training_configs.append((config_name, config, device, seed, run_dir))
    
    # Run training in batches
    total_runs = len(training_configs)
    completed = 0
    print([x[2] for x in training_configs])
    
    while completed < total_runs:
        processes = []
        # Start up to max_processes trainings
        for cfg in training_configs[completed:completed + max_processes]:
            p = mp.Process(
                target=train_single_run,
                args=cfg
            )
            p.start()
            processes.append(p)
            logger.info(f"Started process for {cfg[0]} with seed {cfg[3]}")
        
        # Wait for all processes in this batch to complete
        for p in processes:
            p.join()
        
        completed += len(processes)
        logger.info(f"Completed {completed}/{total_runs} training runs")
    
    logger.info("All training runs completed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user. Exiting gracefully.")
        sys.exit(0)
