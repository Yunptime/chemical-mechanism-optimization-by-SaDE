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


toolbox.register("attr_int1", random.uniform, 0.078667, 0.314667)
toolbox.register("attr_int2", random.uniform, 1.857354, 7.429416)
toolbox.register("attr_int3", random.uniform, 0.062200, 0.248800)
toolbox.register("attr_int4", random.uniform, 20.440617, 81.762470)
toolbox.register("attr_int5", random.uniform, 0.027490, 0.109958)
toolbox.register("attr_int6", random.uniform, 21.847914, 87.391656)
toolbox.register("attr_int7", random.uniform, 14.278171, 57.112684)
toolbox.register("attr_int8", random.uniform, 0.083800, 0.335200)
toolbox.register("attr_int9", random.uniform, 0.194133, 0.776533)
toolbox.register("attr_int10", random.uniform, 0.038833, 0.155333)
toolbox.register("attr_int11", random.uniform, 0.289167, 1.156667)
toolbox.register("attr_int12", random.uniform, 0.038033, 0.152133)
toolbox.register("attr_int13", random.uniform, 0.208333, 0.833333)
toolbox.register("attr_int14", random.uniform, 0.059900, 0.239600)
toolbox.register("attr_int15", random.uniform, 0.084000, 0.336000)
toolbox.register("attr_int16", random.uniform, 0.250000, 1.000000)
toolbox.register("attr_int17", random.uniform, 0.103200, 0.412800)
toolbox.register("attr_int18", random.uniform, 0.039333, 0.157333)
toolbox.register("attr_int19", random.uniform, 0.083333, 0.333333)
toolbox.register("attr_int20", random.uniform, 0.083333, 0.333333)
toolbox.register("attr_int21", random.uniform, 0.083333, 0.333333)
toolbox.register("attr_int22", random.uniform, 0.325533, 1.302133)
toolbox.register("attr_int23", random.uniform, 0.077933, 0.311733)
toolbox.register("attr_int24", random.uniform, 0.034367, 0.137467)
toolbox.register("attr_int25", random.uniform, 0.137300, 0.549200)
toolbox.register("attr_int26", random.uniform, 0.043667, 0.174667)
toolbox.register("attr_int27", random.uniform, 0.112333, 0.449333)
toolbox.register("attr_int28", random.uniform, 0.081667, 0.326667)
toolbox.register("attr_int29", random.uniform, 0.034333, 0.137333)
toolbox.register("attr_int30", random.uniform, 0.090667, 0.362667)
toolbox.register("attr_int31", random.uniform, 0.038133, 0.152533)
toolbox.register("attr_int32", random.uniform, 0.035667, 0.142667)
toolbox.register("attr_int33", random.uniform, 0.188333, 0.753333)
toolbox.register("attr_int34", random.uniform, 1157.760000, 4631.040000)
toolbox.register("attr_int35", random.uniform, 160.320000, 641.280000)
toolbox.register("attr_int36", random.uniform, 0.058667, 0.234667)
toolbox.register("attr_int37", random.uniform, 0.050333, 0.201333)
toolbox.register("individual", tools.initCycle, creator.Individual,(toolbox.attr_int1,toolbox.attr_int2,toolbox.attr_int3,toolbox.attr_int4,toolbox.attr_int5,toolbox.attr_int6,toolbox.attr_int7,toolbox.attr_int8,toolbox.attr_int9,toolbox.attr_int10,toolbox.attr_int11,toolbox.attr_int12,toolbox.attr_int13,toolbox.attr_int14,toolbox.attr_int15,toolbox.attr_int16,toolbox.attr_int17,toolbox.attr_int18,toolbox.attr_int19,toolbox.attr_int20,toolbox.attr_int21,toolbox.attr_int22,toolbox.attr_int23,toolbox.attr_int24,toolbox.attr_int25,toolbox.attr_int26,toolbox.attr_int27,toolbox.attr_int28,toolbox.attr_int29,toolbox.attr_int30,toolbox.attr_int31,toolbox.attr_int32,toolbox.attr_int33,toolbox.attr_int34,toolbox.attr_int35,toolbox.attr_int36,toolbox.attr_int37),n=1)

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



        count[850]="reaction('C2H5COCH3 + OH <=> C4H7O + H2O', [%fe+02, 3.15, -3050.4])\n"%ind[0]
        count[858]="reaction('C2H5COCH3 + H <=> C4H7O + H2', [%fe+06, 2.256, 4083.8])\n"%ind[1]
        count[863]="reaction('C2H5COCH3 + CH3O2 <=> C4H7O + CH3O2H', [%fe-03, 4.75, 16233.2])\n"%ind[2]
        count[870]="                 kf=[%fe+27, -3.184, 85748.1],\n"%ind[3]
        count[871]="                 kf0=[%fe+79, -17.02, 96875.4],\n"%ind[4]
        count[882]="reaction('C4H7O <=> CH3 + CH3CHCO', [%fe+20, -1.826, 49167.1])\n"%ind[5]
        count[888]="reaction('C4H7O + HO2 <=> RO + OH', [%fe+29, -4.607, 11770.5])\n"%ind[6]
        count[893]="reaction('C4H7O + CH3O2 <=> RO + CH3O', [%fe+25, -3.607, 6948.5])\n"%ind[7]
        count[898]="reaction('C4H7O + O2 <=> ROO', [%fe+60, -15.344, 17596.6],\n"%ind[8]
        count[903]="reaction('C4H7O + O2 <=> ROO', [%fe+107, -27.173, 70238.6],\n"%ind[9]
        count[909]="reaction('RO <=> CH3CO + CH3CHO', [%fe+10, 0.614, 1017.2])\n"%ind[10]
        count[913]="reaction('RO <=> CH2O + C2H5CO', [%fe+13, 0.099, 1125.8])\n"%ind[11]
        count[917]="reaction('CH2O + CH3COCH2 <=> RO', [%fe+10, 0.0, 11900.0])\n"%ind[12]
        count[923]="reaction('ROO <=> C2H3COCH3 + HO2', [%fe+54, -13.77, 37419.2])\n"%ind[13]
        count[928]="reaction('ROO <=> QOOH', [%fe+12, -0.251, 27950.0])\n"%ind[14]
        count[933]="reaction('QOOH <=> C4H6O2 + OH', [%fe+10, 0.0, 18800.0])\n"%ind[15]
        count[938]="reaction('QOOH => CH2CO + OH + CH3CHO', [%fe+18, -1.73, 26820.0])\n"%ind[16]
        count[942]="reaction('QOOH => CH3CHCO + OH + CH2O', [%fe+09, 1.2, 22700.0])\n"%ind[17]
        count[947]="reaction('C4H6O2 + OH => H2O + HCCO + CH3CHO', [%fe+12, 0.0, 0.0])\n"%ind[18]
        count[951]="reaction('C4H6O2 + OH => H2O + CH3CO + CH2CO', [%fe+12, 0.0, 0.0])\n"%ind[19]
        count[955]="reaction('C4H6O2 + OH => H2O + HCO + C2H3CHO', [%fe+12, 0.0, 0.0])\n"%ind[20]
        count[962]="                 kf=[%fe+20, -1.176, 84070.1],\n"%ind[21]
        count[963]="                 kf0=[%fe+73, -15.321, 93700.1],\n"%ind[22]
        count[970]="reaction('H + C2H3COCH3 <=> C2H4 + CH3CO', [%fe+18, -1.013, 14229.1])\n"%ind[23]
        count[983]="reaction('C2H3COCH3 + OH <=> CH3CHO + CH3CO', [%fe+24, -3.285, 13062.0])\n"%ind[24]
        count[996]="reaction('CH3CHO + H <=> CH3CO + H2', [%fe+05, 2.58, 1220.0])\n"%ind[25]
        count[999]="reaction('CH3CHO + OH <=> CH3CO + H2O', [%fe+12, 0.0, -619.0])\n"%ind[26]
        count[1003]="                 kf=[%fe+22, -1.74, 86360.0],\n"%ind[27]
        count[1004]="                 kf0=[%fe+59, -11.3, 95912.5],\n"%ind[28]
        count[1009]="                 kf=[%fe+21, -1.74, 86355.0],\n"%ind[29]
        count[1010]="                 kf0=[%fe+58, -11.3, 95912.5],\n"%ind[30]
        count[1015]="                 kf=[%fe+12, 0.63, 16900.0],\n"%ind[31]
        count[1016]="                 kf0=[%fe+18, -0.97, 14600.0],\n"%ind[32]
        count[1023]="                 kf=[%fe+10, 0.929, 78030.6],\n"%ind[33]
        count[1024]="                 kf0=[%fe+54, -9.989, 87553.3],\n"%ind[34]
        count[1030]="reaction('CH2CO + CH3 <=> CH3COCH2', [%fe+04, 2.48, 6130.0])\n"%ind[35]
        count[1034]="reaction('C2H5 + CO <=> C2H5CO', [%fe+11, 0.0, 4810.0])\n"%ind[36]

        f=open(file,'w',encoding='UTF-8')
        f.writelines(count)
        f.close()
        p=Pool(npro)
    # 加入多进程




        a1 = p.apply_async(pre_process, [process_parameter[0]])
        a2 = p.apply_async(pre_process, [process_parameter[1]])
        result=[0, 1]
        result[0]=a1.get()
        result[1]=a2.get()



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




            count[850]="reaction('C2H5COCH3 + OH <=> C4H7O + H2O', [%fe+02, 3.15, -3050.4])\n"%ind[0]
            count[858]="reaction('C2H5COCH3 + H <=> C4H7O + H2', [%fe+06, 2.256, 4083.8])\n"%ind[1]
            count[863]="reaction('C2H5COCH3 + CH3O2 <=> C4H7O + CH3O2H', [%fe-03, 4.75, 16233.2])\n"%ind[2]
            count[870]="                 kf=[%fe+27, -3.184, 85748.1],\n"%ind[3]
            count[871]="                 kf0=[%fe+79, -17.02, 96875.4],\n"%ind[4]
            count[882]="reaction('C4H7O <=> CH3 + CH3CHCO', [%fe+20, -1.826, 49167.1])\n"%ind[5]
            count[888]="reaction('C4H7O + HO2 <=> RO + OH', [%fe+29, -4.607, 11770.5])\n"%ind[6]
            count[893]="reaction('C4H7O + CH3O2 <=> RO + CH3O', [%fe+25, -3.607, 6948.5])\n"%ind[7]
            count[898]="reaction('C4H7O + O2 <=> ROO', [%fe+60, -15.344, 17596.6],\n"%ind[8]
            count[903]="reaction('C4H7O + O2 <=> ROO', [%fe+107, -27.173, 70238.6],\n"%ind[9]
            count[909]="reaction('RO <=> CH3CO + CH3CHO', [%fe+10, 0.614, 1017.2])\n"%ind[10]
            count[913]="reaction('RO <=> CH2O + C2H5CO', [%fe+13, 0.099, 1125.8])\n"%ind[11]
            count[917]="reaction('CH2O + CH3COCH2 <=> RO', [%fe+10, 0.0, 11900.0])\n"%ind[12]
            count[923]="reaction('ROO <=> C2H3COCH3 + HO2', [%fe+54, -13.77, 37419.2])\n"%ind[13]
            count[928]="reaction('ROO <=> QOOH', [%fe+12, -0.251, 27950.0])\n"%ind[14]
            count[933]="reaction('QOOH <=> C4H6O2 + OH', [%fe+10, 0.0, 18800.0])\n"%ind[15]
            count[938]="reaction('QOOH => CH2CO + OH + CH3CHO', [%fe+18, -1.73, 26820.0])\n"%ind[16]
            count[942]="reaction('QOOH => CH3CHCO + OH + CH2O', [%fe+09, 1.2, 22700.0])\n"%ind[17]
            count[947]="reaction('C4H6O2 + OH => H2O + HCCO + CH3CHO', [%fe+12, 0.0, 0.0])\n"%ind[18]
            count[951]="reaction('C4H6O2 + OH => H2O + CH3CO + CH2CO', [%fe+12, 0.0, 0.0])\n"%ind[19]
            count[955]="reaction('C4H6O2 + OH => H2O + HCO + C2H3CHO', [%fe+12, 0.0, 0.0])\n"%ind[20]
            count[962]="                 kf=[%fe+20, -1.176, 84070.1],\n"%ind[21]
            count[963]="                 kf0=[%fe+73, -15.321, 93700.1],\n"%ind[22]
            count[970]="reaction('H + C2H3COCH3 <=> C2H4 + CH3CO', [%fe+18, -1.013, 14229.1])\n"%ind[23]
            count[983]="reaction('C2H3COCH3 + OH <=> CH3CHO + CH3CO', [%fe+24, -3.285, 13062.0])\n"%ind[24]
            count[996]="reaction('CH3CHO + H <=> CH3CO + H2', [%fe+05, 2.58, 1220.0])\n"%ind[25]
            count[999]="reaction('CH3CHO + OH <=> CH3CO + H2O', [%fe+12, 0.0, -619.0])\n"%ind[26]
            count[1003]="                 kf=[%fe+22, -1.74, 86360.0],\n"%ind[27]
            count[1004]="                 kf0=[%fe+59, -11.3, 95912.5],\n"%ind[28]
            count[1009]="                 kf=[%fe+21, -1.74, 86355.0],\n"%ind[29]
            count[1010]="                 kf0=[%fe+58, -11.3, 95912.5],\n"%ind[30]
            count[1015]="                 kf=[%fe+12, 0.63, 16900.0],\n"%ind[31]
            count[1016]="                 kf0=[%fe+18, -0.97, 14600.0],\n"%ind[32]
            count[1023]="                 kf=[%fe+10, 0.929, 78030.6],\n"%ind[33]
            count[1024]="                 kf0=[%fe+54, -9.989, 87553.3],\n"%ind[34]
            count[1030]="reaction('CH2CO + CH3 <=> CH3COCH2', [%fe+04, 2.48, 6130.0])\n"%ind[35]
            count[1034]="reaction('C2H5 + CO <=> C2H5CO', [%fe+11, 0.0, 4810.0])\n"%ind[36]


            f=open(file,'w',encoding='UTF-8')
            f.writelines(count)
            f.close()
            p=Pool(npro)
        # 加入多进程





            a1 = p.apply_async(pre_process, [process_parameter[0]])
            a2 = p.apply_async(pre_process, [process_parameter[1]])
            result=[0, 1]
            result[0]=a1.get()
            result[1]=a2.get()


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