import torch

class Dataset_Custom(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, task_name, list_IDs):
        'Initialization'
        self.data_path = data_path
        self.list_IDs = list_IDs
        self.task_name = task_name

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load(self.data_path+'/'+self.task_name+'-{}.db'.format(str(10000+ID)[1:]))

        return X


data_path = '/home/cosmos/VScode Projects/coglab/GenSim/data_1k'
tsk_name = 'data'
list_IDs = range(1000)

training_sets = []
training_sets.append(Dataset_Custom(data_path,tsk_name,range(74)))

train_dev_sets = torch.utils.data.ConcatDataset(training_sets)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 1,  #256
          'shuffle': True,
          'num_workers': 0}

training_generator = torch.utils.data.DataLoader(train_dev_sets, **params)


for local_batch in training_generator:
        tr = local_batch
        
        states = tr[0].to(device);
        actions = {k:[v[0].to(device),v[1].to(device)] for k,v in tr[1].items()}
        rewards = tr[2].to(device);
        lang_goals = tr[3]

        x, poses, losses = model.forward_df(states, lang_goals, actions, targets=True)