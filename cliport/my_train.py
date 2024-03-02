"""Main training script."""

import os
from pathlib import Path

import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset, RavenMultiTaskDatasetBalance, MyCustomDataset

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import IPython
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import datetime
import time
import random


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="./cfg", config_name='train', version_base="1.2")
def main(cfg):
    import torch.cuda
    # Logger
    set_seed_everywhere(1)
    wandb_logger = None

    if cfg['train']['log']:
        try:
            wandb_logger = WandbLogger(name=cfg['tag'])
        except:
            pass

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['data_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = [ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=os.path.join(checkpoint_path, 'best_{epoch:02d}'),
        mode='min',
        save_top_k=3,
        every_n_epochs=1,
        save_last=True,
        # every_n_train_steps=1    
        )]

    # Trainer
    max_epochs = cfg['train']['n_steps'] * cfg['train']['batch_size'] // cfg['train']['n_demos']
    
    if cfg['train']['training_step_scale'] > 0:
        # scale training time depending on the tasks to ensure coverage.
        max_epochs = cfg['train']['training_step_scale']  # // cfg['train']['batch_size']
    use_cuda = torch.cuda.is_available()
    device = "gpu" if use_cuda else "cpu"
    trainer = Trainer(
        accelerator=device,
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        callbacks=checkpoint_callback,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        # resume_from_checkpoint=last_checkpoint,
        sync_batchnorm=True,
        log_every_n_steps=1,  
        enable_checkpointing=True      
    )

    print(f"max epochs: {max_epochs}!")
    
    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    
    if agent_type == 'mdetr':
        print('======import torch.multiprocessing to avioid shared memory issue======')
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

    # n_demos = cfg['train']['n_demos']
    # n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)
            
    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', 
                    n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    elif 'weighted' in dataset_type:
        train_ds = RavenMultiTaskDatasetBalance(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavenMultiTaskDatasetBalance(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    elif 'my' in dataset_type:
        data_dir = cfg['train']['train_dir']
        # tsk_name = 'data'
        # list_IDs = range(1000)
        # training_sets = []
        # training_sets.append(Dataset_Custom(data_path,tsk_name,range(74)))
        # import torch
        # train_dev_sets = torch.utils.data.ConcatDataset(training_sets)

        ds = MyCustomDataset(data_dir, 'data', cfg, 
                    n_demos=n_demos, augment=False)
        # val_ds = MyCustomDataset(data_dir, 'data', cfg, 
        #             n_demos=n_val, augment=False)

        train_size = int(0.9 * len(ds))
        val_size = len(ds) - train_size
        print("TRAIN/VAL SIZE: ", train_size, val_size)
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])


    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)


    
    # Parameters
    # params = {'batch_size': 1,  #256
    #         'shuffle': True,
    #         'num_workers': 0}
    # if 'my' in dataset_type:
    #     train_loader = torch.utils.data.DataLoader(train_dev_sets, **params)
    #     test_loader = torch.utils.data.DataLoader(train_dev_sets, **params)
    # else:
    # Initialize agent
    train_loader = DataLoader(train_ds, shuffle=True,
                    pin_memory=True,
                    batch_size=cfg['train']['batch_size'],
                    num_workers=11 )
    test_loader = DataLoader(val_ds, shuffle=False,
                num_workers=11,
                batch_size=cfg['train']['batch_size'],
                pin_memory=True)

    agent = agents.names[agent_type](name, cfg, train_loader, test_loader)
    agent = agent.to(dtype=torch.float)
    dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    print("current time:", dt_string)
    
    start_time = time.time()
    # Main training loop
    trainer.fit(agent, ckpt_path=last_checkpoint)
    trainer.save_checkpoint(last_checkpoint_path)
    
    print("current time:", time.time() - start_time)

if __name__ == '__main__':
    main()
