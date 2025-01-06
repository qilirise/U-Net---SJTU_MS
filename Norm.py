import os
import xarray as xr
import matplotlib.pyplot as plt
data=xr.open_dataset('./Data_SCS.nc')




def single_norm(var_name,coord_name):

    x_sum=data[var_name].isel(channel=0).mean(dim=coord_name).sum(dim=['i','j'])
    x_up=x_sum+0.01*280*242#时间点矩阵尺寸
    x_down=x_sum-0.01*280*242

    for i in range(data[var_name].sizes[coord_name]):
        os.makedirs(f'D:/DATA/test1/{var_name}', exist_ok=True)
        if (data[var_name].isel(**{coord_name:i},channel=0).sum(dim=['i','j']))>x_down and (data[var_name].isel(**{coord_name:i},channel=0).sum(dim=['i','j']))<x_up :
            print(f'{i}')
            # data[var_name].isel(**{coord_name:i},channel=0).to_netcdf(f'D:/DATA/test1/{var_name}/{i}.nc')
            data[var_name].isel(**{coord_name:i},channel=0).plot(cmap='gray',add_colorbar=False,add_labels=False).axes.axis('off')
            plt.savefig(f'D:/DATA/test1/{var_name}/{i}.png',bbox_inches='tight',pad_inches=0)
            plt.close()

            var_name1='UBM'+var_name[3:]
            print(var_name)
            data[var_name1].isel(**{coord_name:i},channel=0).plot(cmap='gray',add_colorbar=False,add_labels=False).axes.axis('off')
            plt.savefig(f'D:/DATA/test1/{var_name1}/{i}.png',bbox_inches='tight',pad_inches=0)
            plt.close()


var_tuple=zip(['SSH_Train','SSH_Test'],['time_train','time_test'])
for j,k in var_tuple:
    single_norm(var_name=j,coord_name=k)
