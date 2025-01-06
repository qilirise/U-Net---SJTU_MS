#文件拆分函数
import os
import xarray as xr

def func2(var_name,coord):
    #var:变量名 coord:坐标长度
    for time_index in range(coord):
        if var_name[-5:]=='Train':
            os.makedirs(f'./DATA/{var_name}', exist_ok=True)
            data[var_name].isel(time_train=time_index,channel=0).to_netcdf(f'./DATA/{var_name}/{time_index}.nc')  
        if var_name[-3:]=='Val':
            os.makedirs(f'./DATA/{var_name}', exist_ok=True)
            data[var_name].isel(time_val=time_index,channel=0).to_netcdf(f'./DATA/{var_name}/{time_index}.nc')
        if var_name[-4:]=='Test'or var_name[-3:]=='Pre':
            os.makedirs(f'./DATA/{var_name}', exist_ok=True)
            data[var_name].isel(time_test=time_index,channel=0).to_netcdf(f'./DATA/{var_name}/{time_index}.nc')
# data
list1=['SSH_Train','SSH_Val','SSH_Test','BM_Train','BM_Val','BM_Test','UBM_Train','UBM_Val','UBM_Test']


data=xr.open_dataset('./Data_SCS.nc')
for var_name in list1:
    if var_name[-5:]=='Train':
        coord=data.time_train.size
        func2(var_name,coord)
    if var_name[-3:]=='Val':
        coord=data.time_val.size
        func2(var_name,coord)
    if var_name[-4:]=='Test'or var_name[-3:]=='Pre':
        coord=data.time_test.size
        func2(var_name,coord)