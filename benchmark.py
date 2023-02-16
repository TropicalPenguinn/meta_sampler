import pandas as pd
import argparse
import string
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='benchmark1')
parser.add_argument('--task')
parser.add_argument('--algo',type=str)
args=parser.parse_args()
task=args.task.split(',')
algo=args.algo.split(',')
#benchmark2={"InvertedPendulum-v2":50,"Reacher-v2":100,"Swimmer-v2":50}

def mean(lst):
    return sum(lst)/len(lst)
def std(lst):
    lst=np.array(lst)
    return np.std(lst)


for t in task:
    episodic_mean = []
    std_lst = []
    #benchmark2 = {"InvertedPendulum-v2": 50, "Reacher-v2": 100, "Swimmer-v2": 50}
    target_score = []
    #N_f = benchmark2[t]
    for a in algo:
        df1 = pd.read_csv('./run/{}/{}/test_reward_seeds.csv'.format(t,a))
        df2=df1
        df1 = df1.values.tolist()

        for lst in df1:
            l=lst[3:-1]
            episodic_mean.append(mean(l))
            #if len(episodic_mean)>=N_f:
            #    target_score.append(mean(l))
            std_lst.append(std(l))
        print(args.algo)
        print("R: {:0.3f}".format(max(episodic_mean)))
        print("sigma:{:0.3f}".format(mean(std_lst)))
        #print("target score: {:0.3f}".format(mean(target_score)))
        #df2 = df2.to_numpy()
        for lst in df1:
            l=lst[3:-1]
            if max(episodic_mean)==mean(l):
                break



        #index=df2[:,0].tolist()
        #target_reach=[]
        #for i in range(3,df2.shape[1]):
        #    d=df2[:,i].tolist()
        #    for j in range(df2.shape[0]):
        #        if d[j]>mean(target_score):
        #            target_reach.append(index[j])
        #            break

        #print("Reach mean: {:0.3f} Reach Std: {:0.3f}".format(mean(target_reach),std(target_reach)))
        #print('\n')






