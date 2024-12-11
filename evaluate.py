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
import GameSimulation

#evaluation

checkpointSteps = 20
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
model.load_state_dict(torch.load('trainedModelChangedSmallOnly.pth', map_location=device))
model.eval()
max_eval_steps = 810

def sampleActions(logits, temp):
    """     percentile = torch.quantile(logits, 0.5, dim=-1)
    logits = torch.where(logits > percentile[..., None], logits, -np.inf)
    chosenAction = D.Categorical(logits=torch.flatten(logits[:,1:])*temp).sample()
    action = torch.zeros((logits.size(0)))
    action[chosenAction//4] = (chosenAction % 4) + 1 """
    grain = torch.max(torch.max(logits[:,1:],1).values,0)
    act = torch.max(logits[:,1:],1).indices[grain.indices]
    action = torch.zeros((logits.size(0)))
    action[grain.indices] = act+1
    return action

# Get sets of game simulations and initial states
files = os.listdir(f"{Path(__file__).parent}/EvalDataBRK")
envs = []
returns = []
grains = []
batch_timesteps = 4
batch_size = len(files)
state, actions, returns, pos_encode, timesteps, obsMask, actMask, retMask = [], [], [], [], [], [], [], []
with torch.no_grad():
    for i in range(len(files)):
        env = GameSimulation.GameSimulation(files[i])
        envs.append(env)
        state.append(np.stack([env.prevQ[0] for j in range(4)], axis=0))
        returns.append(np.expand_dims([env.evalGBN(env.prevQ[0])/env.MaxScore for j in range(4)], axis=1)) # Get starting property
        actions.append(np.stack([np.zeros((env.nGrains)) for j in range(4)], axis=0))
        grains.append(env.nGrains)
        pos_encode.append([env.getPosEncoding(env.prevQ[0]) for j in range(4)])

        #Pad the outputs (prepend) and generate the mask (not the attention mask)
        pad_size = max_grains-envs[i].nGrains
        state[-1] = np.concatenate([np.zeros([batch_timesteps, pad_size, 4]), state[-1]],axis=1)
        actions[-1] = np.concatenate([np.zeros([batch_timesteps,pad_size]), actions[-1]], axis=1)
        pos_encode[-1] = np.concatenate([np.zeros([batch_timesteps,pad_size, 4]), pos_encode[-1]],axis=1)
        obsMask.append(np.concatenate([np.zeros([batch_timesteps, pad_size], dtype=bool), np.ones([batch_timesteps, envs[i].nGrains], dtype=bool)],axis=1))

        # Mask away the actions so that the model predicts instead of learns 
        actMask.append(np.concatenate([np.zeros([batch_timesteps, pad_size], dtype=bool), np.ones([batch_timesteps, envs[i].nGrains], dtype=bool)],axis=1))
        retMask.append(np.ones([batch_timesteps, 1], dtype=bool))
        timesteps.append(np.repeat(np.arange(0,0+batch_timesteps),max_grains*2+1))

    states = torch.from_numpy(np.stack(state, axis=0)).to(dtype=torch.float32, device=device)
    actions = torch.from_numpy(np.stack(actions, axis=0)).to(dtype=torch.int, device=device)
    returns = torch.from_numpy(np.stack(returns, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).to(dtype=torch.int, device=device)
    pos_encode = torch.from_numpy(np.stack(pos_encode, axis=0)).to(dtype=torch.float32, device=device)
    mask = [np.stack(obsMask, axis=0), np.stack(retMask, axis=0), np.stack(actMask, axis=0)]
    mask = np.concatenate(mask, axis=-1)
    mask = np.reshape(mask, [batch_size, (max_grains * 2 +1) * batch_timesteps])
    mask = torch.tensor(mask, dtype=torch.bool, device=device)

    #Evaluate models
    return_hist = []
    action_hist = []
    state_hist = []
    attn_hist = []
    allDone = np.zeros(len(envs), dtype=bool)
    for i in range(max_eval_steps):
        
        # Sample a higher next return, CAN BE IMPROVED
        state_preds, action_preds, return_preds, seq_loss, seq_acc = model.forward(states, actions, returns, pos_encode, timesteps, mask, attention_mask=None, n_g=grains)
        if i < 3:
            returns[:,i,:] = return_preds[:,i,:]+0.1
        else:
            returns[:,-1,:] = return_preds[:,-1,:]+0.1
        # Get action logits
        state_preds, action_preds, return_preds, seq_loss, seq_acc = model.forward(states, actions, returns, pos_encode, timesteps, mask, attention_mask=None, n_g=grains)

        # Take next step
        done, Returns, States, posEncodings, acts = [],[],[],[],[]
        for j, env in enumerate(envs):
            if i<3:
                acts.append(torch.cat((torch.zeros((max_grains-env.nGrains), device=device),sampleActions(action_preds[j,i,-env.nGrains:], 1).to(dtype=torch.int, device=device))))
            else:
                acts.append(torch.cat((torch.zeros((max_grains-env.nGrains), device=device),sampleActions(action_preds[j,-1,-env.nGrains:], 1).to(dtype=torch.int, device=device))))
            isdone, Deff, state, posEncoding = env.step(acts[j][-env.nGrains:].detach().cpu().numpy())
            done.append(isdone)
            Returns.append(Deff/env.MaxScore) # Normalize score from max possible score
            States.append(np.concatenate([np.zeros([max_grains-env.nGrains, 4]), state],axis=0))
            posEncodings.append(np.concatenate([np.zeros([max_grains-env.nGrains, 4]), posEncoding],axis=0))
        acts = torch.stack(acts, dim=0).to(dtype=torch.int, device=device)
        States = torch.from_numpy(np.stack(States, axis=0)).to(dtype=torch.float32, device=device)
        Returns = torch.from_numpy(np.expand_dims(np.stack(Returns, axis=0), axis=1)).to(dtype=torch.float32, device=device)
        done = np.stack(done)
        allDone = np.logical_or(allDone, done)
        posEncodings = torch.from_numpy(np.stack(posEncodings, axis=0)).to(dtype=torch.float32, device=device)
        #Record data
        return_hist.append(Returns)
        action_hist.append(acts)
        #state_hist.append(States)
        #attn_hist.append(model.attention.layers[0].attn.attnScores.cpu())
        # Prepare next step, dropping the oldest
        if i<3:
            states[:,i+1,...] = States
            actions[:,i,:] = acts
            pos_encode[:,i+1,...] = posEncodings
            returns[:,i+1] = Returns
            actStart = (max_grains * 2 +1)*(i+1)
            for j, env in enumerate(envs):
                mask[j,actStart-env.nGrains:actStart] = 1 # Update mask to allow attention to previous actions
        else:
            states = torch.concat((states[:,1:,...], States[:,None,:,:]), dim=1).to(dtype=torch.float32, device=device)
            actions = torch.concat((actions[:,1:3,:], acts[:,None,:], actions[:,3:4,:]), dim=1).to(dtype=torch.int, device=device)
            pos_encode = torch.concat((pos_encode[:,1:,:], posEncodings[:,None,:]), dim=1).to(dtype=torch.float32, device=device)
            returns = torch.concat((returns[:,1:], Returns[:,None,:]), dim=1).to(dtype=torch.float32, device=device)
            timesteps = torch.from_numpy(np.tile(np.repeat(np.arange(i+2-batch_timesteps,i+2),max_grains*2+1), (len(envs),1))).to(dtype=torch.int, device=device)

        if allDone.all():
            break
        #matIo.savemat('AllHistPred10-2-24ParaPert1.mat', {'ReturnHist':np.stack([ret.cpu() for ret in return_hist]),'ActHist':np.stack([act.cpu() for act in action_hist]),'StateHist':np.stack([st.cpu() for st in state_hist]),'AttnScores':np.stack([at.cpu() for at in attn_hist])})
        # Checkpoint status
        if i%checkpointSteps == 0:
            matIo.savemat('AllHistPred11-13-24SmallOnlyBRK.mat', {'ReturnHist':np.stack([ret.cpu() for ret in return_hist]),'ActHist':np.stack([act.cpu() for act in action_hist])})
    # Save after completion
    matIo.savemat('AllHistPred11-13-24SmallOnlyBRK.mat', {'ReturnHist':np.stack([ret.cpu() for ret in return_hist]),'ActHist':np.stack([act.cpu() for act in action_hist])})
