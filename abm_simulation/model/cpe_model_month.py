#%%
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
from mesa.space import MultiGrid
from model.agents import *
import sys
import os
import pandas as pd
from scipy.stats import qmc, expon

#%%
def getHCWInfec(model):
    """every time a HCW infects a patient"""
    return model.num_infecByHCW

def getCumul(model):
    """cumulative patients"""
    return model.cumul_patients

def getTotalInfec(model):
    """the number of infections per day"""
    return model.totalHWCinf

#%%
ICUA = ['A14','A15','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13']
ICUB = ['B14','B15','B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']

import numpy as np
path_A = np.array_split(ICUA, 5)
path_B = np.array_split(ICUB, 5)
paths = np.concatenate((path_A, path_B))

class CPE_Model_month(Model):
    """A model with some number of agents."""
    def __init__(self, 
    data_type,
    prob_new_patient, 
    prob_transmission, 
    isolation_factor, 
    cleaningDay, hcw_wash_rate, isolation_time , 
    height, width):

        self.num_HCWs = 10
        self.num_Patients = 30
        self.NumEnv = []
        self.running = True

        # 기본 parameter
        if data_type == 'A':
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..',
            'data/dataA_per_months.csv')
            df = pd.read_csv(csv_path, usecols=['PI_counts'])
            PI_counts = df['PI_counts'].astype(int).tolist()
            months = list(range(0, 30 * 19 + 1, 30)) # 1.5 yrs
            self.inflow_date = [x for x, n in zip(months, PI_counts) for _ in range(n)]
            self.hospital_period = 7 # exp(1/lambda)

        elif data_type == 'B':
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..',
            'data/dataB_per_months.csv')
            df = pd.read_csv(csv_path, usecols=['PI_counts'])
            PI_counts = df['PI_counts'].astype(int).tolist()
            months = list(range(0, 30 * 36 + 1, 30)) # 3 yrs
            self.inflow_date = [x for x, n in zip(months, PI_counts) for _ in range(n)]
            self.hospital_period = 14 # exp(1/lambda)


        elif data_type == 'B_':
            csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..',
            'data/dataB_per_months.csv')
            df = pd.read_csv(csv_path, usecols=['PI_counts'])
            PI_counts = df['PI_counts'].astype(int).tolist()
            months = list(range(0, 30 * 30 + 1, 30)) # 2.5 yrs
            self.inflow_date = [x for x, n in zip(months, PI_counts) for _ in range(n)]
            self.hospital_period = 16 # exp(1/lambda)


        self.prob_new_patient = prob_new_patient # geometric rv
        self.prob_transmission = prob_transmission
        self.isolation_factor = isolation_factor

        self.cleaningDay = cleaningDay
        self.isolation_time = isolation_time
        self.hcw_wash_rate = hcw_wash_rate

        self.grid = MultiGrid(width, height, torus =False)

        self.ticks_in_hour = 3*36 # 36 ticks to visit 3 patients, 3 cycles per hour
        self.ticks_in_day = 24 * self.ticks_in_hour
        self.day = 0

        self.schedule = BaseScheduler(self)
        self._steps = 0
        self._time = 0
        # Data collect variables 설정
        self.discharged = []
        self.current_patients = [] # List of all patients
        self.empty_beds = set() # empty set
        self.shared_beds = [] # list of all non isolated beds
        self.isol_beds = [] # list of all isolated beds

        self.summon = False
        self.summoner = 0

        self.cumul_patients = self.num_Patients # cumulative patients (*)
        self.cumul_sick_patients = 0 # cumulative sick patients (*)
        self.num_infecByHCW = 0 # HCW에 의한 감염환자 수 (*)
        self.totalHWCinf = [] # list for infected people 
        self.P_I = 0
        self.num_move2isol = 0 # 검사로 인해 Isolated bed로 옮겨지는 환자 수 (*)

        # Create Agents
        # Xray Drs 3명
        strange = XrayDr(-self.num_HCWs - 7, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-1, y = 0, workHours=24, numCare = 30, shiftsPerDay = 2)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = XrayDr(-self.num_HCWs - 8, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-2, y = 0, workHours=24, numCare = 20, shiftsPerDay = 1)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = XrayDr(-self.num_HCWs - 9, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-2, y = 1, workHours=24, numCare = 3, shiftsPerDay = 1)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        # Drs 6명
        strange = Dr(-self.num_HCWs - 6, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-1, y = height-1, workHours=24, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = Dr(-self.num_HCWs - 5, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-2, y = height-1, workHours=12, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = Dr(-self.num_HCWs - 4, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-2, y = height-2, workHours=12, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = Dr(-self.num_HCWs - 3, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-1, y = height-2, workHours=3, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = Dr(-self.num_HCWs - 2, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-4, y = height-1, workHours=1, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        strange = Dr(-self.num_HCWs - 1, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = width-4, y = height-2, workHours=1, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))


        # Create HCW agents 간호사 Num_HCWs = 10명(30명/3교대)
        for j in range(-self.num_HCWs, 0):
            b = Nurse(j, self, colonized = False, hand_wash_rate = self.hcw_wash_rate, x = -2*j-2, y = 0)#, path = ex_path)
            self.schedule.add(b)
            self.grid.place_agent(b, (b.x, b.y)) #added 1 to start near patients for route simulation (temporary solution)
            #print(model.schedule.agents)
            b.path = paths[-j-1]
            if b.path[0][0] == 'A':
                b.hall = 6 #ICUA
            else:
                b.hall = 5 #ICUB

        ## Latin hypercube for a.stay
        N = 30  # 샘플 개수
        sampler = qmc.LatinHypercube(d=1)
        lhs_samples = sampler.random(n=N)
        lambda_param = self.hospital_period  # 지수 분포의 람다(lambda) 파라미터
        exponential_dist = expon(scale=1/lambda_param)  # scale = 1 / lambda
        exponential_samples = exponential_dist.ppf(lhs_samples.flatten())
        exponential_samples = np.round(exponential_samples)
        
        # bed,patient,Goo 30개
        for i in range(30):
            # 위치 정하기
            if i in range(11):
                xpos = 2 * i + 5
                ypos = 1
            elif i in range(11, 13):
                #isolated = True
                xpos = 2 * (i-11) + 1
                ypos = 3
            elif i in range(13, 15):
                #isolated = True
                xpos = 2 * (i-13) + 27
                ypos = 3
            elif i in range(15, 17):
                #isolated = True
                xpos = 2 * (i-15) + 1
                ypos = 8
            elif i in range(17, 19):
                #isolated = True
                xpos = 2 * (i-17) + 27
                ypos = 8
            elif i in range(19, 30):
                xpos = 2 * (i-19) + 5
                ypos = 10

            #isolated status
            isolated = False
            if xpos in {1,3,5,7,23,25,27,29}: #isolation
                isolated = True

            """We first create beds so that patient.checkIsolated() works"""
            # Beds
            if isolated:
                b = IsolatedBed(i, self, colonized = False, x = xpos, y = ypos) # 0~30
                self.schedule.add(b)
                self.grid.place_agent(b, (b.x, b.y))
                self.isol_beds.append(b)
            else:
                b = Bed(i, self, colonized = False, x = xpos, y = ypos) # 0~30
                self.schedule.add(b)
                self.grid.place_agent(b, (b.x, b.y))
                self.shared_beds.append(b)

            # Patients
            a = Patient(self.num_Patients + i, self, colonized = False, x = xpos, y = ypos)#30~60
            self.schedule.add(a)
            self.grid.place_agent(a, (a.x, a.y))
            if a.colonized == True:
                self.cumul_sick_patients += 1
            
            a.stay = np.random.choice(exponential_samples, 1, replace = False) * self.ticks_in_day

            #Goo
            if ypos > 5: # top row
                d = Goo(-self.num_Patients - i, self, colonized = False, x=xpos, y=ypos-1) #-30~-60
            else: # bottom row
                d = Goo(-self.num_Patients - i, self, colonized = False, x=xpos, y=ypos+1) #-30~-60
            self.schedule.add(d)
            self.grid.place_agent(d, (d.x, d.y))

        
        self.datacollector = DataCollector(
            model_reporters={
                "HCW related infections": getTotalInfec,
                # "Cumulative Patients": getCumul
                })

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.schedule.time %= self.ticks_in_day # to keep the number from getting too large
        if self.schedule.time == 0:
            if self.day%30 == 0:
                self.totalHWCinf.append(self.num_infecByHCW)
                self.num_infecByHCW = 0
                self.day += 1
            else: 
                self.day += 1
                
            #self.totalHWCinf.append(self.num_infecByHCW) ##revised_by_Wi
            #self.num_infecByHCW = 0 ##revised_by_Wi. Reset PHAI
            
            
        # sommon Nurse
        if self.schedule.time % self.ticks_in_hour == 0:
            self.summoner = self.schedule.time // self.ticks_in_hour
            if self.summoner > 0 and self.summoner < 16: # only for patients 1~15
                self.summon = True

        # remove patient
        for ex_patient in self.discharged:
            self.grid.remove_agent(ex_patient)
            self.schedule.remove(ex_patient)
            self.discharged.remove(ex_patient)

        # Move patients to Isolated beds
        for bed in self.shared_beds: # 7-person room
            if bed.filledSick:
                cellmates = self.grid.get_cell_list_contents([bed.pos])
                for cellmate in cellmates: # in case HCW is also in the same cell
                    if not cellmate.isPatient:
                        continue # hcw or THE BED ITSELF
                    else: # cellmate = patient
                        if cellmate.move2isol: # 옮겨야 될 시간이 된 공유침대 감염환자 발견
                            sickguy = cellmate
                            for ibed in self.isol_beds:
                                if ibed.filledSick:
                                    continue
                                if ibed.filled and not ibed.filledSick: # 격리침대 환자가 아프지 않은지 확인
                                    icellmates = self.grid.get_cell_list_contents([ibed.pos])

                                    for icellmate in icellmates:
                                        if icellmate.isPatient: # 아프지 않은 환자일거임
                                            healthyguy = icellmate
                                            self.grid.move_agent(sickguy, (ibed.x, ibed.y))
                                            self.num_move2isol += 1
                                            self.grid.move_agent(healthyguy, (bed.x, bed.y))
                                            ibed.checkFilled() # to label the bed filled
                                            break # we don't need to consider other icellmates

                                    break # we don't need to consider other beds
                                else: # not filled
                                    self.grid.move_agent(sickguy, (ibed.x, ibed.y))
                                    self.num_move2isol += 1
                                    ibed.checkFilled() # to label the bed filled
                                break
            else:
                continue

# %%
