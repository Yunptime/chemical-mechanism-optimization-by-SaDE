import numpy as np
import cantera as ct

class cal_ignition(object):
    def __init__(self,conditions):
        self.p1=conditions['pressure']
        self.file=conditions['filename']
        self.tem=conditions['temperature']
        self.conditions=conditions['fuel']

    def ignitionDelay(self, states, species, dT=1):
        if dT == 1:
            i_ign = (np.argmax(np.gradient(states().P) / np.gradient(
                states().t)))
        else:
            i_ign = states(species).Y.argmax()
        return states.t[i_ign]

    def forward(self):

        estimatedIgnitionDelayTimes = 0.005
        igni = []
        for i, tem in enumerate(self.tem):

            ideal_gas = ct.Solution(self.file)  # 要不要加呢

            reactorTemperature10 = tem  # 这里temperature改成每一个温度
            ideal_gas.TP = reactorTemperature10, self.p1
            ideal_gas.X = self.conditions
            r = ct.Reactor(contents=ideal_gas)
            reactorNetwork = ct.ReactorNet([r])
            timeHistory = ct.SolutionArray(ideal_gas, extra=['t'])
            t = 0
            counter = 0
            while t < estimatedIgnitionDelayTimes:
                t = reactorNetwork.step()
                if not counter % 20:
                    timeHistory.append(r.thermo.state, t=t)
                counter += 1
            tau = self.ignitionDelay(timeHistory, 'OH')
            igni.append(tau)  # 工况1的第一个温度的着火延迟
        return igni

class cal_fspeed(object):
    def __init__(self,conditions):
        self.p1=conditions['pressure']
        self.file=conditions['filename']
        self.tem=conditions['temperature']
        self.conditions=conditions['fuel']
        self.dlb=conditions['DLB'] #传入list
    def forward(self):
        reactorPressure1 =self.p1
        Tin = self.tem
        DLB = self.dlb
        width = 0.03  # m
        speed = []


        l = list(self.conditions.keys())
        v=list(self.conditions.values())
        value=v[0]
        for i in DLB:
            conditions = self.conditions
            # print(self.conditions)
            conditions[l[0]]=value*i

            # print(conditions)
            ideal_gas = ct.Solution(self.file)  # 要不要加呢
            ideal_gas.TP = Tin, reactorPressure1
            ideal_gas.X = self.conditions

            f = ct.FreeFlame(ideal_gas, width=width)
            f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)

            f.solve(loglevel=1, auto=True)
            speed.append(f.u[0])
        return speed


def pre_process(sumclass):
    value=[]
    for i in sumclass:
        value+=i.forward()
    return value
