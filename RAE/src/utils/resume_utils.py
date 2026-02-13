import os
import sys
from omegaconf import OmegaConf
import logging
from typing import Optional, Tuple
import shutil
from .wandb_utils import initialize, create_logger
import logging

def configure_experiment_dirs(args, full_cfg, rank) -> Tuple[str, str, logging.Logger]:
    experiment_name = f"{full_cfg.stage_2.model}__{full_cfg.stage_2.exp_name}__g{full_cfg.training.extract_layer}__kernel{args.kernel_size}__view{args.max_views}__{args.precision}__ep{full_cfg.training.epochs}__bs-{full_cfg.training.global_batch_size}__lr-{full_cfg.training.optimizer.lr:.0e}__ema{full_cfg.training.ema_decay}"
    experiment_dir = os.path.join(args.results_dir, full_cfg.stage_2.save_folder, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints") 
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, 'rae')
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None, 'rae')
    return experiment_dir, checkpoint_dir, logger

def find_resume_checkpoint(resume_dir) -> Optional[str]:
    """
    Find the latest checkpoint file in the resume directory.
    Args:
        resume_dir (str): Path to the resume directory.
    Returns:
        str: Path to the latest checkpoint file.
    """
    if not os.path.exists(resume_dir):
        raise ValueError(f"Resume directory {resume_dir} does not exist.")
    checkpoint_dir = os.path.join(resume_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") or f.endswith(".ckpt") or f.endswith(".safetensor")
    ]
    if len(checkpoints) == 0:
        return None
    # sort via epoch, saved as 'ep-{epoch:07d}.pt'
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(os.path.basename(x).split("-")[1].split(".")[0]),
    )
    return checkpoints[-1]

def save_worktree(
    path: str,
    config: OmegaConf,  
) -> None:
    OmegaConf.save(config, os.path.join(path, "config.yaml"))
    worktree_path = os.path.join(os.getcwd(), "RAE", "src")
    shutil.copytree(worktree_path, os.path.join(path, "RAE", "src/"), dirs_exist_ok=True)
    print(f'Worktree {worktree_path} saved to {os.path.join(path, "RAE", "src/")}')

if __name__ == "__main__":
    experiment_dir = sys.argv[1]
    ckpt_path = find_resume_checkpoint(experiment_dir)
    print(f"Latest checkpoint found at: {ckpt_path}")