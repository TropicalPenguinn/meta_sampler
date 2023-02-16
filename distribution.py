import pickle
import numpy as np

with open('checkpoints/sac_buffer_HalfCheetah-v2_1000','rb') as f:
    data1=pickle.load(f)
with open('checkpoints/sac_buffer_HalfCheetah-v2_2000','rb') as f:
    data2=pickle.load(f)
with open('checkpoints/sac_buffer_HalfCheetah-v2_3000','rb') as f:
    data3=pickle.load(f)
with open('checkpoints/sac_buffer_HalfCheetah-v2_4000','rb') as f:
    data4=pickle.load(f)

state=[]
action=[]
for d in data1:
    state.append(d[0])
    action.append(d[1])
for d in data2:
    state.append(d[0])
    action.append(d[1])
for d in data3:
    state.append(d[0])
    action.append(d[1])
for d in data4:
    state.append(d[0])
    action.append(d[1])

state=np.array(state)
action=np.array(action)
print("state dim: ",state.shape[1])
print("action dim: ",action.shape[1])

for i in range(state.shape[1]):
    print("state dim_{}".format(i),np.mean(state[:,i]),np.std(state[:,i]),np.min(state[:,i]),np.max(state[:,i]))

for i in range(action.shape[1]):
    print("action dim_{}".format(i), np.mean(action[:, i]), np.std(action[:, i]))