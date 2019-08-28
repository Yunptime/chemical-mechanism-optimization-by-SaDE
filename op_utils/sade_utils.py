import numpy as np
import deap
import random
from deap import base
toolbox = base.Toolbox()
def generate_reactionset(ngene,reaction,reacline):
    with open('./prepare_setting/reaction_setting.txt', 'w', encoding='utf-8')as f:
        for i in range(ngene):
            f.write('count[%u]=' % (reacline[i]))
            f.write(r'"')
            f.write(str(reaction[i]))
            f.write(r'\n')
            f.write(r'"')
            f.write('%')
            f.write('ind[%u]' % i)
            f.write('\n')


def generate_process(numpro,parameter):
    f=open('./prepare_setting/process_setting.txt','w',encoding='utf-8')
    i=0
    while i < numpro:
        f.write('        a%u = p.apply_async(pre_process, [process_parameter[%u]])'%(i+1,i))
        f.write('\n')
        i+=1
    list=range(2)
    result=[]
    for i in list:
        result.append(i)
    f.write('        result=')
    f.write(str(result))
    f.write('\n')
    i=0
    while i<numpro:
        f.write('        result[%u]=a%u.get()'%(i,i+1))
        f.write('\n')
        i+=1
    f.close()

def generate_tools(num_gene,value,low,high):
    f=open('./prepare_setting/setting.txt','w',encoding='utf-8')
    i=0
    temp=[]
    while i < num_gene:
        f.write('toolbox.register("attr_int%u", random.uniform, %f, %f)'%(i+1,value[i]*low,value[i]*high))
        f.write('\n')
        temp.append('toolbox.attr_int%u'%(i+1))
        i+=1
    a=','.join(temp)
    f.write('toolbox.register("individual", tools.initCycle, creator.Individual,(%s),n=1)'%str(a))
    f.close()
def relative_min(num_conditions,sim,exp,index):
    sim=np.array(sim)
    exp=np.array(exp)
    fit=((sim-exp)/exp)**index
    sum=0
    for i in fit:
        sum+=abs(i)
    return sum/num_conditions,

class Sade():
    def __init__(self,fmin,fmax):
        self.fmin=fmin
        self.fmax=fmax

    def rand_1_bin(self,ind, dim, wf, cr, pop):  # 定义了两种选择策略
        p1 = ind
        while (p1 == ind):
            p1 = random.choice(pop)
        p2 = ind
        while (p2 == ind or p2 == p1):
            p2 = random.choice(pop)
        p3 = ind
        while (p3 == ind or p3 == p1 or p3 == p2):
            p3 = random.choice(pop)  ###随机选择种群中的三个个体出来，三个个体为P1,P2,P3
        fmin=self.fmin
        fmax=self.fmax

        cutpoint = random.randint(0, dim - 1)  # 分割点是维度减一，随机生成的一个数字
        # 候选策略池子，其实是一个新个体的一整条基因啊！
        candidateSol = toolbox.clone(p1)
        for i in range(dim):
            if (i == cutpoint or random.uniform(0, 1) < cr):
                # candidateSol.append(p3[i]+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))i
                candidateSol[i] = p3[i] + wf * (p1[i] - p2[i])
            else:
                # candidateSol.append(ind[i])
                candidateSol[i] = ind[i]
            while candidateSol[i] < fmin[i] or candidateSol[i] > fmax[i]:
                candidateSol[i] = ind[i]
        return candidateSol  # 返回的是一个新个体，通过策略生成的新个体！


    def randToBest_2_bin(self,ind, best, dim, wf, cr, pop):
        p1 = ind
        while (p1 == ind):
            p1 = random.choice(pop)
        p2 = ind  # 再随机挑选两个，这就是rand to best bin 2
        while (p2 == ind or p2 == p1):
            p2 = random.choice(pop)
        p3 = ind
        while (p3 == ind or p3 == p1 or p3 == p2):
            p3 = random.choice(pop)
        p4 = ind
        while (p4 == ind or p4 == p1 or p4 == p2 or p4 == p3):
            p4 = random.choice(pop)
        cutpoint = random.randint(0, dim - 1)
        candidateSol = toolbox.clone(p1)
        fmin=self.fmin
        fmax=self.fmax
        for i in range(dim):
            if (i == cutpoint or random.uniform(0, 1) < cr):
                # candidateSol.append(ind[i]+wf*(best[i]-ind[i])+wf*(p1[i]-p2[i])) # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))
                candidateSol[i] = ind[i] + wf * (best[i] - ind[i]) + wf * (p1[i] - p2[i]) + wf * (p3[i] - p4[i])
            else:
                # candidateSol.append(ind[i])
                candidateSol[i] = ind[i]
            while candidateSol[i] < fmin[i] or candidateSol[i] > fmax[i]:
                candidateSol[i] = ind[i]
                # print('candidateSol: %s' % str(candidateSol))
            # input('...')
            # print('\n\n')
        return candidateSol


    def rand_2_bin(self, ind, dim, wf, cr, pop):
        p1 = ind
        while (p1 == ind):
            p1 = random.choice(pop)
        p2 = ind
        while (p2 == ind or p2 == p1):
            p2 = random.choice(pop)
        p3 = ind
        while (p3 == ind or p3 == p1 or p3 == p2):
            p3 = random.choice(pop)
        p4 = ind
        while (p4 == ind or p4 == p1 or p4 == p2 or p4 == p3):
            p4 = random.choice(pop)
        p5 = ind
        while (p5 == ind or p5 == p1 or p5 == p2 or p5 == p3 or p5 == p4):
            p5 = random.choice(pop)
        fmin=self.fmin
        fmax=self.fmax
        cutpoint = random.randint(0, dim - 1)  # 分割点是维度减一，随机生成的一个数字
        candidateSol = toolbox.clone(p1)

        for i in range(dim):
            if (i == cutpoint or random.uniform(0, 1) < cr):
                # -> rand(p3) , vetor diferença (wf*(p1[i]-p2[i]))i
                candidateSol[i] = p1[i] + wf * (p2[i] - p3[i]) + wf * (p4[i] - p5[i])
            else:
                candidateSol[i] = ind[i]
            while candidateSol[i] < fmin[i] or candidateSol[i] > fmax[i]:
                candidateSol[i] = ind[i]
        return candidateSol


    def currentTorand_1(self, ind, dim, wf, cr, pop):
        p1 = ind
        while (p1 == ind):
            p1 = random.choice(pop)
        p2 = ind
        while (p2 == ind or p2 == p1):
            p2 = random.choice(pop)
        p3 = ind
        while (p3 == ind or p3 == p1 or p3 == p2):
            p3 = random.choice(pop)
        cutpoint = random.randint(0, dim - 1)  # 分割点是维度减一，随机生成的一个数字
        candidateSol = toolbox.clone(p1)
        fmin=self.fmin
        fmax=self.fmax
        K = 0.4  # 来源于文献Solving Rotated Multi-objective Optimization Problems UsingDi erentialEvolution
        for i in range(dim):
            candidateSol[i] = ind[i] + K * (p1[i] - ind[i]) + wf * (p2[i] - p3[i])

            while candidateSol[i] < fmin[i] or candidateSol[i] > fmax[i]:
                candidateSol[i] = ind[i]
        return candidateSol
