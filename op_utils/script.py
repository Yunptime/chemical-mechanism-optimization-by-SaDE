from op_utils.chemical_utils import cal_ignition,pre_process,cal_fspeed
from op_utils.sade_utils import relative_min,generate_tools,Sade
import cantera as ct
from multiprocessing import Pool,Process,Manager
import collections
from get_flags import Flags
import numpy as np
import time
import random
import array
from deap import base,benchmarks,creator,tools
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()




toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("select", tools.selRandom, k=3)

toolbox.register("evaluate",relative_min)
def getBestSolution(maximize,fpop,pop):#找到目前最好的个体和适应度的值，不作为输出，主要用于策略2的选择任务。
    fbest = fpop[0]
    best = [values for values in pop[0]]
    for ind in range(1,len(pop)):
        if maximize == True:
            if fpop[ind] >= fbest:
                fbest = float(fpop[ind])
                best = [values for values in pop[ind]]
        else:
            if fpop[ind] <= fbest:
                fbest = fpop[ind]
                best = [values for values in pop[ind]]

    return fbest,best
def main():

    fl=Flags()
    fmin = fl.fmin
    fmax = fl.fmax
    fmin = np.array(fmin)
    fmax = np.array(fmax)

    ssade = Sade(fmin, fmax)  # 定义Sade类

    process_parameter=fl.pro_parameter

    # print('start of the SaDE')
    random.seed(fl.rseed) #固定随机数方便复现
    MU = fl.MU #个体数量

    NGEN = fl.NGEN    #代数  要更新的 维度dim，种群个体数MU，三个周期，基因边界范围
    dim=fl.dim
    learningPeriod=fl.lp  #学习周期,p\cr的更新周期
    npro=fl.num_pro
    fitness_index=fl.fitness_index
    num_conditions=fl.num_condi
    experi_data=fl.exp
    file=fl.file

    p1,p2,p3,p4,crm1,crm2,crm3,crm4=0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5
    ns1,ns2,ns3,ns4,nf1,nf2,nf3,nf4=[],[],[],[],[],[],[],[]
    maximize = False

    pop = toolbox.population(n=MU);
    hof = tools.HallOfFame(1)  # 精英策略、名人堂,保留最优解
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # 随时观察元素状态，用于输出
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    fitnesses = []
    for index,ind in enumerate(pop):
        f=open(file,'r',encoding='UTF-8')
        count = f.readlines()
        f.close()
        #生成反应信息




        f=open(file,'w',encoding='UTF-8')
        f.writelines(count)
        f.close()
        p=Pool(npro)
    # 加入多进程







        p.close()
        p.join()
        newresult=[]
        for i in result:
            for value in i:
                newresult.append(value)
        fitnesses.append(toolbox.evaluate(num_conditions,newresult,experi_data,fitness_index))
    shuzhifitness=[]
    for ness in fitnesses:
        shuzhifitness.append(str(ness)[1:-2])
    logbook = tools.Logbook()#日志记录的工具
    logbook.header = "gen", "invals", "std", "min", "avg", "max"#标头
    t0=time.time()
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    t1=time.time()
    # print('initial fitnesses time for %f individuals'%MU,'compute time:{:3.2f}s'.format(t1-t0))
    # print('initial definition time for %f individuals'%MU,'compute time:{:3.2f}s'.format(t0-t8))

    best=[]
    fbest=0.00
    fbest,best = getBestSolution(maximize, shuzhifitness,pop)
    nowtime = time.strftime('%Y%m%d', time.localtime(time.time()))
    crossover_rate1 = [random.gauss(crm1, 0.1) for i in range(MU)] ##pop_size手动修改吗？
    crossover_rate2 = [random.gauss(crm2, 0.1) for i in range(MU)] ##pop_size手动修改吗？
    crossover_rate3 = [random.gauss(crm3, 0.1) for i in range(MU)] ##pop_size手动修改吗？
    crossover_rate4 = [random.gauss(crm4, 0.1) for i in range(MU)] ##pop_size手动修改吗？
    cr1_list = []
    cr2_list = []
    cr3_list = []
    cr4_list = []
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    f4=open('output/log_%s.txt'%(str(file)),'a+',encoding='UTF-8')
    #都是保存信息的意思**record是fitness的数值属性
    # print(logbook.stream)
    f4.write(str(logbook.stream))
    f4.write('\n')
    f4.close()
    for g in range(1, NGEN):
        list1=['p1','p2','p3','p4']
        def weight_choice(weight):
            t=random.uniform(0,1)
            for i,val in enumerate(weight):
                t-=val
                if t<0:
                    return i
        t2=time.time()
        strategy = 0
        f1=open('output/forrestart_%s.txt'%(str(file)),'a+',encoding='UTF-8')
        f2=open('output/population_%s.txt'%(str(file)),'a+',encoding='UTF-8')
        f2.write(str(g))
        f2.write('sade\n')
        f2.write('\n')

        for k in range(0, len(pop)):
            weight_factor = random.gauss(0.5, 0.3)
            a = list1[weight_choice([p1, p2, p3, p4])]
            ind=pop[k]
            if a == 'p1':
                candSol = ssade.rand_1_bin(pop[k], dim, weight_factor, crossover_rate1[k], pop)
                strategy = 1
            if a == 'p2':
                candSol = ssade.randToBest_2_bin(pop[k], best, dim, weight_factor, crossover_rate2[k], pop)
                strategy = 2
            if a == 'p3':
                candSol = ssade.rand_2_bin(pop[k], dim, weight_factor, crossover_rate3[k], pop)
                strategy = 3
            if a == 'p4':
                candSol = ssade.currentTorand_1(pop[k], dim, weight_factor, crossover_rate4[k], pop)
                strategy = 4
            f2.write(str(a))
            f2.write('..')
            f2.write('\n')
            f=open(file,'r',encoding='UTF-8')
            count = f.readlines()
            f.close()
            # 加入反应信息






            f=open(file,'w',encoding='UTF-8')
            f.writelines(count)
            f.close()
            p=Pool(npro)
        # 加入多进程







            p.close()
            p.join()
            newresult = []
            for i in result:
                for value in i:
                    newresult.append(value)
            candSol.fitness.values = toolbox.evaluate(num_conditions,newresult,experi_data,fitness_index)


            fcandSol=candSol.fitness.values
            if candSol.fitness > pop[k].fitness:
                pop[k]=candSol
                shuzhifitness[k]=str(fcandSol)[1:-2]
                if strategy == 1:
                    ns1.append(1)
                    ns2.append(0)
                    ns3.append(0)
                    ns4.append(0)
                    nf1.append(0) #不写的话若成功f组不会添加，要保证每一代f\s都加mu个
                    nf2.append(0)
                    nf3.append(0)
                    nf4.append(0)
                    cr1_list.append(crossover_rate1[k])
                if strategy == 2:
                    ns1.append(0)
                    ns2.append(1)
                    ns3.append(0)
                    ns4.append(0)
                    nf1.append(0) #不写的话若成功f组不会添加，要保证每一代f\s都加mu个
                    nf2.append(0)
                    nf3.append(0)
                    nf4.append(0)
                    cr2_list.append(crossover_rate2[k])
                if strategy == 3:
                    ns1.append(0)
                    ns2.append(0)
                    ns3.append(1)
                    ns4.append(0)
                    nf1.append(0) #不写的话若成功f组不会添加，要保证每一代f\s都加mu个
                    nf2.append(0)
                    nf3.append(0)
                    nf4.append(0)
                    cr3_list.append(crossover_rate3[k])
                if strategy == 4:
                    ns1.append(0)
                    ns2.append(0)
                    ns3.append(0)
                    ns4.append(1)
                    nf1.append(0) #不写的话若成功f组不会添加，要保证每一代f\s都加mu个
                    nf2.append(0)
                    nf3.append(0)
                    nf4.append(0)
                    cr4_list.append(crossover_rate4[k])
            else:
               # print('s5')
                if strategy == 1:
                    nf1.append(1)
                    nf2.append(0)
                    nf3.append(0)
                    nf4.append(0)
                    ns1.append(0)
                    ns2.append(0)
                    ns3.append(0)
                    ns4.append(0)
                if strategy == 2:
                    nf1.append(0)
                    nf2.append(1)
                    nf3.append(0)
                    nf4.append(0)
                    ns1.append(0)
                    ns2.append(0)
                    ns3.append(0)
                    ns4.append(0)
                if strategy == 3:
                    nf1.append(0)
                    nf2.append(0)
                    nf3.append(1)
                    nf4.append(0)
                    ns1.append(0)
                    ns2.append(0)
                    ns3.append(0)
                    ns4.append(0)
                if strategy == 4:
                    nf1.append(0)
                    nf2.append(0)
                    nf3.append(0)
                    nf4.append(1)
                    ns1.append(0)
                    ns2.append(0)
                    ns3.append(0)
                    ns4.append(0)
        fbest,best = getBestSolution(maximize,shuzhifitness,pop)
        if g > learningPeriod and g != 0:
            as1 = sum(ns1[:MU])
            as2 = sum(ns2[:MU])
            as3 = sum(ns3[:MU])
            as4 = sum(ns4[:MU])
            crm1 = np.median(cr1_list)  # sum(cr_list1)/len(cr_list1)
            crm2 = np.median(cr2_list)  # sum(cr_list2)/len(cr_list2)
            crm3 = np.median(cr3_list)  # sum(cr_list3)/len(cr_list3)
            crm4 = np.median(cr4_list)  # sum(cr_list4)/len(cr_list4)
            del cr1_list[:as1], cr2_list[:as2], cr3_list[:as3], cr4_list[:as4]
            crossover_rate1 = [random.gauss(crm1, 0.1) for i in range(MU)]  # 按文献取的高斯分布更新
            crossover_rate2 = [random.gauss(crm2, 0.1) for i in range(MU)]  ##pop_size手动修改吗？
            crossover_rate3 = [random.gauss(crm3, 0.1) for i in range(MU)]  ##pop_size手动修改吗？
            crossover_rate4 = [random.gauss(crm4, 0.1) for i in range(MU)]  ##pop_size手动修改吗？
        # print('success5')
        if g > learningPeriod and g != 0:
            s1 = 0.01 + sum(ns1) / ((sum(ns1) + sum(nf1)))
            s2 = 0.01 + sum(ns2) / ((sum(ns2) + sum(nf2)))
            s3 = 0.01 + sum(ns3) / ((sum(ns3) + sum(nf3)))
            s4 = 0.01 + sum(ns4) / ((sum(ns4) + sum(nf4)))
            ssum = s1 + s2 + s3 + s4
            p1 = s1 / ssum
            p2 = s2 / ssum
            p3 = s3 / ssum
            p4 = s4 / ssum
            f1.write('\n')
            f1.write('new pro\n')
            f1.write(str(p1))
            f1.write('\n')
            f1.write(str(p2))
            f1.write('\n')
            f1.write(str(p3))
            f1.write('\n')
            f1.write(str(p4))
            f1.write('\n')
            # p1 =(ns1*(ns2+nf2))/(ns2*(ns1+nf1)+ns1*(ns2+nf2))
            # p2=1-p1
            del ns1[0:MU], ns2[0:MU], ns3[0:MU], ns4[0:MU], nf1[0:MU], nf2[0:MU], nf3[0:MU], nf4[0:MU]
        f1.write('crm\n')
        f1.write(str(crm1))
        f1.write('\n')
        f1.write(str(crm2))
        f1.write('\n')
        f1.write(str(crm3))
        f1.write('\n')
        f1.write(str(crm4))
        f1.write('\n')

        f1.write('ns1 nf1\n')
        f1.write(str(ns1))
        f1.write('\n')
        f1.write(str(nf1))
        f1.write('\n')
        f1.write('ns2 nf2\n')
        f1.write(str(ns2))
        f1.write('\n')
        f1.write(str(nf2))
        f1.write('\n')
        f1.write('ns3 nf3\n')
        f1.write(str(ns3))
        f1.write('\n')
        f1.write(str(nf3))
        f1.write('\n')
        f1.write('ns4 nf4\n')
        f1.write(str(ns4))
        f1.write('\n')
        f1.write(str(nf4))
        f1.write('\n')
        f1.write('\n')
        f1.write('crossover_rate1\n')
        f1.write(str(cr1_list))
        f1.write('\n')
        f1.write('crossover_rate2\n')
        f1.write(str(cr2_list))
        f1.write('\n')
        f1.write('crossover_rate3\n')
        f1.write(str(cr3_list))
        f1.write('\n')
        f1.write('crossover_rate4\n')
        f1.write(str(cr4_list))
        f1.write('\n')
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        t3 = time.time()
        f2.write('\n')
        f2.write(str(g))
        f2.write('best individual\n')
        f2.write(str(hof[0]))
        f2.write(str(hof[0].fitness.values))
        f2.write('\n')
        f2.write('pop\n')
        f2.write(str(pop))
        f2.write('\n')
        f1.write('\n')
        f1.write('fitness\n')
        f1.write(str(shuzhifitness))
        f1.write('\n')
        # print(logbook.stream, 'compute time for every cycle{:3.2f}s'.format(t3 - t2))
        f4 = open('output/log_%s.txt'%(str(file)), 'a+', encoding='UTF-8')
        f4.write(str(logbook.stream))
        f4.write('  best individual')
        f4.write(str(hof[0]))
        f4.write(str( hof[0].fitness.values[0]))
        f4.write('\n')
        f4.close()
        nowtime = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time()))
        # print(nowtime)
        f1.write('\n')
        f1.write(str(nowtime))
        f1.write('\n')
        f2.write('\n')
        f2.write(str(nowtime))
        f2.write('\n')
        f1.close()
        f2.close()
    f4 = open('output/log_%s.txt'%(str(file)), 'a+', encoding='UTF-8')
    f4.write('  best individual')
    f4.write(str(hof[0]))
    f4.write(str(hof[0].fitness.values[0]))
    f4.write('\n')
    f4.close()
    # print("Best individual is ", hof[0], hof[0].fitness.values[0])
    # print(pop)
    t21=time.time()
    # print('total compute time{:3.2f}s'.format(t21-t8))

if __name__ == "__main__":
    main()