import torch
import torch.nn.functional as F
import GameTransformer
import scipy.io as matIo
import numpy as np
import os
import random
from pathlib import Path
import time
from torch import distributions as D

def getBatch(batch_size, max_grains, files, device):
    #Get dataset for training
    actDict = {'nothing':0,'ori':1,'OptGrain':2,'Undo':3,'anneal':4}
    
    batch_timesteps = 4
    state, actions, returns, pos_encode, timesteps, obsMask, actMask, retMask = [], [], [], [], [], [], [], []
    allGrains = []
    for i in range(batch_size):
        validFile = True
        f, d, quats, nTimeSteps = [],[],[],[]
        while validFile:

            f = random.randint(0, len(files)-1)
            d = matIo.loadmat(f'{Path(__file__).parent}/PlayerDataSmallOnly/'+files[f],squeeze_me=True)

            quats = np.stack(d['tHist'][:,1],dtype=np.float32) # These are the quaternions of all grains at each time step
            nTimeSteps = quats.shape[0]
            if nTimeSteps > 4:
                validFile = False

        Deff = d['tHist'][:,2] # This is the effective diffusivity at each time step
        Action = d['tHist'][:,0] # This is the action taken at each time step
        ActGrain = d['tHist'][:,3] # This is the grain the action was taken on
        EigVec = np.stack(d['tHist'][:,4],dtype=np.float32) # These are the eigenvectors of the Dual network
        MaxScore = d['maxScore']
        dual = d['structureUsed']['Dual'] # This is the dual of the GBN / the position encoding of the action space
        nGrains = quats[0].shape[0]
        
        assert nTimeSteps > 4, f"File {files[f]} does not have enough timesteps"
        
        sample = random.randint(0, nTimeSteps-batch_timesteps-1) # Pick a set of batch_timesteps from file to sample
        #State requires [Batch x Timestep x Grain x Quaternion]
        state.append(quats[sample:sample+batch_timesteps,:,:])

        #Action requires [Batch x Timestep x Action]
        parsedAct = [actDict[x] for x in Action]
        act = np.zeros([nTimeSteps, nGrains],dtype=int)
        for i in range(nTimeSteps):
            act[i,ActGrain[i]-1] = parsedAct[i]
        # Get the next action done for the given state
        actions.append(act[sample+1:sample+1+batch_timesteps,:])

        #Return requires [Batch x Timestep] contains [return]
        # Like action get the expected score from the next action
        returns.append(np.expand_dims(np.stack(Deff[sample+1:sample+1+batch_timesteps] / MaxScore,dtype=np.float32), axis=1))

        #Timestep requires [Batch x Total Tokens in Timestep] contains [Timestep]
        timesteps.append(np.repeat(np.arange(sample,sample+batch_timesteps),max_grains*2+1))
        
        #Position encodings
        nEigs = 4
        pos_encode.append(EigVec[sample:sample+batch_timesteps,:,:4])

        #Pad the outputs (prepend) and generate the mask (not the attention mask)
        pad_size = max_grains-nGrains
        state[-1] = np.concatenate([np.zeros([state[-1].shape[0], pad_size, 4]), state[-1]],axis=1)
        actions[-1] = np.concatenate([np.zeros([actions[-1].shape[0], pad_size]), actions[-1]], axis=1)
        pos_encode[-1] = np.concatenate([np.zeros([state[-1].shape[0], pad_size, 4]), pos_encode[-1]],axis=1)
        obsMask.append(np.concatenate([np.zeros([state[-1].shape[0], pad_size], dtype=bool), np.ones([batch_timesteps, nGrains], dtype=bool)],axis=1))
        actMask.append(np.concatenate([np.zeros([state[-1].shape[0], pad_size], dtype=bool), np.ones([batch_timesteps, nGrains], dtype=bool)],axis=1))
        retMask.append(np.ones([batch_timesteps, 1], dtype=bool))
        allGrains.append(nGrains)

    state = torch.from_numpy(np.stack(state, axis=0)).to(dtype=torch.float32, device=device)
    actions = torch.from_numpy(np.stack(actions, axis=0)).to(dtype=torch.int, device=device)
    returns = torch.from_numpy(np.stack(returns, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).to(dtype=torch.int, device=device)
    pos_encode = torch.from_numpy(np.stack(pos_encode, axis=0)).to(dtype=torch.float32, device=device)

    mask = [np.stack(obsMask, axis=0), np.stack(retMask, axis=0), np.stack(actMask, axis=0)]
    mask = np.concatenate(mask, axis=-1)
    mask = np.reshape(mask, [batch_size, (max_grains * 2 +1) * batch_timesteps])
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    return state, actions, returns, pos_encode, timesteps, mask, allGrains

#Model Parameters
max_grains = 50
hidden_size = 1024
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
#device = "cpu"
print(device)
model = GameTransformer.GameTransformer(max_grains=max_grains,act_dim=5,hidden_size=hidden_size,max_ep_length=810,n_eigs=4).to(device)
#training
model.train()
#Hyperparameters
batch_size = 16
learnRate = 1e-7
decay= 1e-6
warmup_steps = 1000
max_steps = 10000
optimizer = torch.optim.AdamW(model.parameters(), lr=learnRate, weight_decay=decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda steps: min((steps+1)/warmup_steps,1))
#Loss and accuracy functions are part of the model already
#Get batch
files = os.listdir(f"{Path(__file__).parent}/PlayerDataSmallOnly")
#Set up tracking
train_losses = []
train_acc = []
#training loop
train_start = time.time()

for i in range(max_steps):
    #training step
    state, actions, returns, pos_encode, timesteps, mask, allGrains = getBatch(batch_size=batch_size, max_grains=max_grains, files=files, device=device)
    optimizer.zero_grad()
    state_preds, action_preds, return_preds, loss, acc = model.forward(state, actions, returns, pos_encode, timesteps, mask, attention_mask=None, n_g=allGrains)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.detach().cpu().item())
    train_acc.append(acc.detach().cpu().item())
    scheduler.step()
    if i % 1000 == 0:
        torch.save(model.state_dict(), f'./Checkpoints/chkpt{i}.pth')
        f = open(f'./Checkpoints/chkpt{i}.txt','w')
        f.write(str(train_losses))
        f.write("\n")
        f.write(str(train_acc))
        f.close()
        print(f"Checkpoint {i} time: {time.time() - train_start}")

print(f"Training time: {time.time() - train_start}")
torch.save(model.state_dict(), "trainedModelChangedSmallOnly.pth")
f = open(f'LossAndAccChangedSmallOnly.txt','w')
f.write(str(train_losses))
f.write("\n")
f.write(str(train_acc))
f.close()