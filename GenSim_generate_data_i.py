import os
import sys
import hydra
import numpy as np
import random
import time

from torch.nn import functional as F
from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
import IPython
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from cliport.dataset import RavensDataset
import copy

from cliport.utils import utils
from cliport.tasks import cameras

os.environ['TOKENIZERS_PARALLELISM']= 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
pix_size = 0.003125
cam_config = cameras.RealSenseD415.CONFIG

def main():
    pass

if __name__ == "__main__":
    args = sys.argv[1:]
    index_ = int(args[0])
    
    cfg = {}
    cfg['n'] = 1000
    cfg['task'] = 'build-car'
    cfg['mode'] = 'test'
    cfg['save_data'] = True
    cfg['assets_root'] = '/GenSim/cliport/environments/assets'
    cfg['data_dir'] = '/GenSim/data'
    cfg['disp'] = False
    cfg['shared_memory'] = False
    cfg['record'] = {}
    cfg['record']['save_video'] = True
    cfg['record']['save_video_path'] = '/GenSim/videos'
    cfg['record']['add_text'] = False
    cfg['record']['add_task_text'] = True
    cfg['record']['fps'] = 20
    cfg['record']['video_height'] = 640
    cfg['record']['video_width'] = 720
    
    cfg['dataset'] = {}
    cfg['dataset']['type'] = 'single' # 'single' or 'multi'
    cfg['dataset']['images'] = True
    cfg['dataset']['cache'] = True # load episodes to memory instead of reading from disk
    cfg['dataset']['augment'] = {}
    cfg['dataset']['augment']['theta_sigma'] = 60

    
    data_path = f"/GenSim/data/GenSim_Data_Train/{cfg['task']}/eval"
    if not os.path.exists(data_path): 
        os.makedirs(data_path)
    tasks_ = np.unique(['_'.join(i.split('_')[:-1]) for i in os.listdir(data_path)])
    task_dataset = {}
    for task in tasks_:
        task_dataset[task] = [i for i in os.listdir(data_path) if '_'.join(i.split('_')[:-1])==task]
    task_dataset_len = {}
    task_names_1000 = []
    for task in tasks_:
        task_dataset_len[task] = len([i for i in os.listdir(data_path) if '_'.join(i.split('_')[:-1])==task])
        if task_dataset_len[task]>1000:
            task_names_1000.append(task)

    
    num_demos = 10000
    
    env = Environment(
                cfg['assets_root'],
                disp=cfg['disp'],
                shared_memory=cfg['shared_memory'],
                hz=480,
                record_cfg=cfg['record']
    )

    traj_len = 25
    
    
    for task_name in ['build-car']: # list(tasks.names.keys())[index_*10:(index_+1)*10]:

        if np.any([True if task_name in i else False for i in task_names_1000]):
            print('################################################')
            print('SKIP TASK: ',task_name)
            print('################################################')
            continue
        
        cfg['task'] = task_name.replace("_", "-")
        task = tasks.names[cfg['task']]()
        task.mode = cfg['mode']
        record = cfg['record']['save_video']
        save_data = cfg['save_data']
        agent = task.oracle(env)
        
        record = False
        save_data = True
        suc_ep = 0
        dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
        for seed in range(0,num_demos*2):
        
            np.random.seed(seed)
            random.seed(seed)
            
            env.set_task(task)
            obs = env.reset()
            cmap, hmap = utils.get_fused_heightmap(obs, cam_config, bounds, pix_size)
            info = env.info
            reward = 0
            episode, total_reward = [], 0
            failed_action_episode = False
            
            if record:
                env.start_rec(f'{dataset.n_episodes+1:06d}')
            
            # Rollout expert policy
            for _ in range(task.max_steps):
                act = agent.act(obs, info)

                if (act is not None):
                    if (act['pose0'] is not None):
                        if (act['pose0'][0] is not None):
                            if (act['pose0'][0][0]<0.25 or act['pose0'][0][0]>0.75) or (act['pose0'][0][1]<-0.5 or act['pose0'][0][1]>0.5) or (act['pose0'][0][2]<0.0 or act['pose0'][0][2]>0.28):
                                failed_action_episode = True
                                break
            
                            if (act['pose1'][0][0]<0.25 or act['pose1'][0][0]>0.75) or (act['pose1'][0][1]<-0.5 or act['pose1'][0][1]>0.5) or (act['pose1'][0][2]<0.0 or act['pose1'][0][2]>0.28):
                                failed_action_episode = True
                                break

                episode.append((cmap, hmap, act, reward, info['lang_goal']))
                lang_goal = info['lang_goal']
                obs, reward, done, info = env.step(act)
                cmap, hmap = utils.get_fused_heightmap(obs, cam_config, bounds, pix_size)
                total_reward += reward
                print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
                if done:
                    break
            if record:
                env.end_rec()
        
            episode.append((cmap, hmap, None, reward, info['lang_goal']))
    
            if (failed_action_episode==False) and len(episode)<=25 and save_data and (total_reward > 0.99):
                
                ep_15 = [[],[],[],[],[]]
                img = torch.stack([torch.from_numpy(i[0]) for i in episode])
                ep_15[0] = torch.zeros(traj_len,320,160,3).to(torch.uint8)
                ep_15[0][:len(episode)] = img.to(torch.uint8)

                depth = torch.stack([torch.from_numpy(i[1]) for i in episode])
                ep_15[1] = torch.zeros(traj_len,320,160).to(torch.uint8)
                ep_15[1][:len(episode)] = depth.to(torch.uint8)

                ep_15[2] = {}
                ep_15[2]['pose0'] = [torch.ones(traj_len,3)*torch.from_numpy(np.array([0.25,-0.5,0.])),torch.ones(traj_len,4)*(-1)]
                pose00 = torch.stack([torch.from_numpy(i[2]['pose0'][0]) for i in episode[:-1]])
                pose01 = torch.stack([torch.from_numpy(i[2]['pose0'][1]) for i in episode[:-1]])
                ep_15[2]['pose0'][0][:len(episode)-1] = pose00
                ep_15[2]['pose0'][1][:len(episode)-1] = pose01
                
                pose10 = torch.stack([torch.from_numpy(i[2]['pose1'][0]) for i in episode[:-1]])
                pose11 = torch.stack([torch.from_numpy(i[2]['pose1'][1]) for i in episode[:-1]])
                ep_15[2]['pose1'] = [torch.ones(traj_len,3)*torch.from_numpy(np.array([0.25,-0.5,0.])),torch.ones(traj_len,4)*(-1)]
                ep_15[2]['pose1'][0][:len(episode)-1] = pose10
                ep_15[2]['pose1'][1][:len(episode)-1] = pose11
                
                ep_15[3] = torch.zeros(traj_len,1)
                ep_15[3][:len(episode)] = torch.stack([torch.from_numpy(np.array([i[3]])) for i in episode])        
                ep_15[4] = ['' for i in range(traj_len)]
                ep_15[4][:len(episode)] = [i[4] for i in episode]
                
                torch.save(ep_15, data_path+'/'+'data_{}_{}.db'.format(task_name,str(100000+suc_ep)[1:]))
                
                suc_ep+=1
        
                if suc_ep>1000:
                    break
    
    
    

