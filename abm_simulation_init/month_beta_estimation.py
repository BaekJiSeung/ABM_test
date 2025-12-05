# %%
from model.cpe_model_month import CPE_Model_month
from mesa.batchrunner import batch_run
from model.cpe_model_month import getHCWInfec
from model.cpe_model_month import getTotalInfec
from mesa.batchrunner import BatchRunner
from multiprocessing import freeze_support
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
data_type = 'A'
num_iter = 40; np.int64(num_iter)

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14

runtime = 30*19 # dont forget change A : 30 * 19, B : 30 * 36
probNewPatient = 0.003 # 0.053, Old Calibration # 1/2000, 2592 ticks per day
probTransmission = 0.00005 # calibration result
isolationFactor = 0.75 # fix
height=11
width=32


fixed_params = {
    "data_type" : data_type , 
    "prob_new_patient" : probNewPatient, 
    "prob_transmission" :probTransmission,
    "isolation_factor" : isolationFactor, 
    "cleaningDay" : cleanDay,
    "hcw_wash_rate" : washrate, 
    "isolation_time" : isolationTime, 
    "height" : height, "width" : width 
    }

variable_name = 'prob_transmission'
variable_value = [0.0786]

del fixed_params[variable_name]
variable_params = {variable_name : variable_value}

start_time = time.time()

# STEP4
model = CPE_Model_month(
    data_type=data_type,
    prob_new_patient=probNewPatient, 
    prob_transmission=probTransmission, 
    isolation_factor=isolationFactor,
    cleaningDay=cleanDay,
    hcw_wash_rate=washrate,
    isolation_time=isolationTime,
    height=height, width=width
    )
print('loading...\n\n')


batch_run = BatchRunner(
    CPE_Model_month,
    variable_parameters = variable_params,
    fixed_parameters = fixed_params,
    iterations = num_iter,
    max_steps = model.ticks_in_day * runtime,
    display_progress=True,
    model_reporters={"HCW_related_infecs" : getTotalInfec}
                    #  ,"Num_move_Patients": getNumIsol}
                    #  "Number_of_Patients_sick":getNumSick}
)
print('now run')


# for _ in range(num_iter):
batch_run.run_all()
run_data = batch_run.get_model_vars_dataframe()
df = pd.DataFrame()
for value in variable_value:
    temp = run_data.query('{}=={}'.format(variable_name,value))['HCW_related_infecs']
    df[value] = temp.reset_index(drop = True)
emulated_data = df.values[0][0]

print(emulated_data)

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    'result/emulation_beta_{}.csv'.format(data_type))
if os.path.isfile(csv_path):
    saved_df = pd.read_csv(csv_path,index_col=0)
    saved_df.columns = variable_value
    df = pd.concat([saved_df,df], ignore_index=True)
df.to_csv(csv_path)
print("done!!")


# %%
data_type = 'B'
num_iter = 50; np.int64(num_iter)

# Parameters
cleanDay = 180
washrate = 0.9
isolationTime = 14

runtime = 30*36 # dont forget change A : 30 * 19, B : 30 * 36
probNewPatient = 0.003 # 0.053, Old Calibration # 1/2000, 2592 ticks per day
probTransmission = 0.00005 # calibration result
isolationFactor = 0.75 # fix
height=11
width=32


fixed_params = {
    "data_type" : data_type , 
    "prob_new_patient" : probNewPatient, 
    "prob_transmission" :probTransmission,
    "isolation_factor" : isolationFactor, 
    "cleaningDay" : cleanDay,
    "hcw_wash_rate" : washrate, 
    "isolation_time" : isolationTime, 
    "height" : height, "width" : width 
    }

variable_name = 'prob_transmission'
variable_value = [0.081,0.083,0.087,0.089]

del fixed_params[variable_name]
variable_params = {variable_name : variable_value}

start_time = time.time()

# STEP4
model = CPE_Model_month(
    data_type=data_type,
    prob_new_patient=probNewPatient, 
    prob_transmission=probTransmission, 
    isolation_factor=isolationFactor,
    cleaningDay=cleanDay,
    hcw_wash_rate=washrate,
    isolation_time=isolationTime,
    height=height, width=width
    )
print('loading...\n\n')


batch_run = BatchRunner(
    CPE_Model_month,
    variable_parameters = variable_params,
    fixed_parameters = fixed_params,
    iterations = num_iter,
    max_steps = model.ticks_in_day * runtime,
    display_progress=True,
    model_reporters={"HCW_related_infecs" : getTotalInfec}
                    #  ,"Num_move_Patients": getNumIsol}
                    #  "Number_of_Patients_sick":getNumSick}
)
print('now run')


# for _ in range(num_iter):
batch_run.run_all()
run_data = batch_run.get_model_vars_dataframe()
df = pd.DataFrame()
for value in variable_value:
    temp = run_data.query('{}=={}'.format(variable_name,value))['HCW_related_infecs']
    df[value] = temp.reset_index(drop = True)
emulated_data = df.values[0][0]

print(emulated_data)

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    'result/emulation_beta_{}.csv'.format(data_type))
if os.path.isfile(csv_path):
    saved_df = pd.read_csv(csv_path,index_col=0)
    saved_df.columns = variable_value
    df = pd.concat([saved_df,df], ignore_index=True)
df.to_csv(csv_path)

print("done!!")
# %%
