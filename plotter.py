

#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import argparse
import string

parser = argparse.ArgumentParser(description='Plot Generator')
parser.add_argument('--task')
parser.add_argument('--algo',type=str)

args=parser.parse_args()
algo=args.algo.split(',')
csv_algo_lst = list()
csv_env_step = list()
csv_reward = list()

for a in algo:
    df1 = pd.read_csv('./run/{}/{}/test_reward_seeds.csv'.format(args.task,a))
    df1 = df1.values.tolist()
    for lst in df1:
        for i in lst[3:-1]:
            csv_algo_lst.append(a)
            csv_env_step.append(lst[0])
            csv_reward.append(i)

my_df=pd.DataFrame({"Model":csv_algo_lst,"timesteps":csv_env_step,"Accumulated Reward":csv_reward})
sns.lineplot(x="timesteps",y="Accumulated Reward",hue="Model",data=my_df)
plt.title(args.task)
i=0
while True:
    if os.path.exists('{}_{}.png'.format(args.task,i)):
        i += 1
    else:
        break
plt.savefig('{}_{}.png'.format(args.task,i))