from op_utils.chemical_utils import cal_ignition,pre_process,cal_fspeed

"""Self-adaptive differential evolution algorithm for mechanism optimization
method article: A.K. Qin, V.L. Huang, P.N. Suganthan. Differential Evolution Algorithm With Strategy Adaptation for Global Numerical Optimization. IEEE Transactions on Evolutionary Computation. 13 (2009) 398-417.
"""
"""
SaDE setting 
"""
MU=5  #num of population
ngen=2 #num of generation
random_seed=0  #random seed for repeat
Dim=37         # num of genes in one individual
learnperiod=15 # learn period of SaDE


"""
conditions  setting
"""
filename='butanone.cti'

num_conditions=8
exp_data=[100,200,300,400,500,600,500]

# ignition delay time, just copy it if there are multi conditions
con_i1 = {'C2H5COCH3': 0.01, 'O2': 0.11, 'Ar': 0.88}
Ti1 = [1521.164021, 1493.506494, 1300]
ignay1 = {'pressure': 5, 'fuel':con_i1, 'temperature': Ti1, 'filename': filename}

#laminar speed
con_s1 = {'C2H5COCH3': 0.01, 'O2': 0.055, 'N2': 0.2068}
Ts1=373
dlb1 = [0.7]
lspeed1 = {'pressure': 103351.5, 'fuel': con_s1, 'temperature': Ts1, 'filename': filename, 'DLB': dlb1}

"""
multi process setting
"""
igni1 = cal_ignition(ignay1) # Initialize the condition
speed1 = cal_fspeed(lspeed1)

num_process=2 #example for two process

pro1 = [igni1,igni1]
pro2 = [speed1]
pro_parameter=[pro1,pro2]


"""
mechanism setting 
"""
variable_ratio=[1,4] #low and high ratio
# initial value of reaction rates
alunniwuzi=[0.07866666666666666,1.857354, 0.0622,20.4406175,0.0274895,21.847914 ,
            14.278171,0.0838, 0.19413333333333335, 0.03883333333333334,
             0.2891666666666667, 0.03803333333333333, 0.20833333333333334, 0.0599, 0.084, 0.25, 0.1032,
             0.03933333333333333, 0.08333333333333333, 0.08333333333333333, 0.08333333333333333,
             0.32553333333333334, 0.07793333333333334, 0.03436666666666666, 0.1373, 0.043666666666666666,
             0.11233333333333334, 0.08166666666666667, 0.034333333333333334, 0.09066666666666667,
             0.03813333333333333, 0.035666666666666666, 0.18833333333333332,1157.76,160.32 ,0.058666666666666666,
             0.050333333333333334]

fitness_index=1
#reaction information
reac_lines=[850,858,863,870,871,882,888,893,898,903,909,913,917,923,928,933,938,942,947,951,955,962,963,970,983,996,999,1003,1004,1009,1010,1015,1016,1023,1024,1030,1034]
reactions=["reaction('C2H5COCH3 + OH <=> C4H7O + H2O', [%fe+02, 3.15, -3050.4])" ,
           "reaction('C2H5COCH3 + H <=> C4H7O + H2', [%fe+06, 2.256, 4083.8])",
           "reaction('C2H5COCH3 + CH3O2 <=> C4H7O + CH3O2H', [%fe-03, 4.75, 16233.2])" ,
           "                 kf=[%fe+27, -3.184, 85748.1],",
         "                 kf0=[%fe+79, -17.02, 96875.4]," ,
        "reaction('C4H7O <=> CH3 + CH3CHCO', [%fe+20, -1.826, 49167.1])" ,
        "reaction('C4H7O + HO2 <=> RO + OH', [%fe+29, -4.607, 11770.5])" ,
        "reaction('C4H7O + CH3O2 <=> RO + CH3O', [%fe+25, -3.607, 6948.5])" ,
        "reaction('C4H7O + O2 <=> ROO', [%fe+60, -15.344, 17596.6]," ,
        "reaction('C4H7O + O2 <=> ROO', [%fe+107, -27.173, 70238.6]," ,
        "reaction('RO <=> CH3CO + CH3CHO', [%fe+10, 0.614, 1017.2])" ,
        "reaction('RO <=> CH2O + C2H5CO', [%fe+13, 0.099, 1125.8])" ,
        "reaction('CH2O + CH3COCH2 <=> RO', [%fe+10, 0.0, 11900.0])" ,
        "reaction('ROO <=> C2H3COCH3 + HO2', [%fe+54, -13.77, 37419.2])",
        "reaction('ROO <=> QOOH', [%fe+12, -0.251, 27950.0])" ,
        "reaction('QOOH <=> C4H6O2 + OH', [%fe+10, 0.0, 18800.0])" ,
        "reaction('QOOH => CH2CO + OH + CH3CHO', [%fe+18, -1.73, 26820.0])",
        "reaction('QOOH => CH3CHCO + OH + CH2O', [%fe+09, 1.2, 22700.0])" ,
        "reaction('C4H6O2 + OH => H2O + HCCO + CH3CHO', [%fe+12, 0.0, 0.0])" ,
        "reaction('C4H6O2 + OH => H2O + CH3CO + CH2CO', [%fe+12, 0.0, 0.0])" ,
        "reaction('C4H6O2 + OH => H2O + HCO + C2H3CHO', [%fe+12, 0.0, 0.0])" ,
        "                 kf=[%fe+20, -1.176, 84070.1]," ,
        "                 kf0=[%fe+73, -15.321, 93700.1]," ,
        "reaction('H + C2H3COCH3 <=> C2H4 + CH3CO', [%fe+18, -1.013, 14229.1])" ,
        "reaction('C2H3COCH3 + OH <=> CH3CHO + CH3CO', [%fe+24, -3.285, 13062.0])" ,
        "reaction('CH3CHO + H <=> CH3CO + H2', [%fe+05, 2.58, 1220.0])" ,
        "reaction('CH3CHO + OH <=> CH3CO + H2O', [%fe+12, 0.0, -619.0])" ,
        "                 kf=[%fe+22, -1.74, 86360.0]," ,
        "                 kf0=[%fe+59, -11.3, 95912.5],",
        "                 kf=[%fe+21, -1.74, 86355.0]," ,
        "                 kf0=[%fe+58, -11.3, 95912.5]," ,
        "                 kf=[%fe+12, 0.63, 16900.0]," ,
        "                 kf0=[%fe+18, -0.97, 14600.0]," ,
        "                 kf=[%fe+10, 0.929, 78030.6]," ,
        "                 kf0=[%fe+54, -9.989, 87553.3]," ,
        "reaction('CH2CO + CH3 <=> CH3COCH2', [%fe+04, 2.48, 6130.0])" ,
        "reaction('C2H5 + CO <=> C2H5CO', [%fe+11, 0.0, 4810.0])" ]


"""
Don't need to change
"""
fmin=[]
fmax=[]
for i in alunniwuzi:
    fmin.append(i*variable_ratio[0])
    fmax.append(i*variable_ratio[1])
class Flags():
    def __init__(self):
        self.MU=MU
        self.NGEN=ngen
        self.rseed=random_seed
        self.dim=Dim
        self.lp=learnperiod
        self.ratio=variable_ratio
        self.it_value=alunniwuzi  #未调前值
        self.num_pro=num_process
        self.pro_parameter=pro_parameter
        self.exp=exp_data
        self.num_condi=num_conditions
        self.fitness_index=fitness_index
        self.ignay_conditions=igni1
        self.speed_conditions=speed1
        self.fmin=fmin
        self.fmax=fmax
        self.file=filename
        self.reaction=reactions
        self.reacline=reac_lines
       