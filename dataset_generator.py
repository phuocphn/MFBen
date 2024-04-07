import os
import numpy as np
import h5py
from tqdm import tqdm
import pyvista as pv
import matplotlib
matplotlib.use('Agg')

def reduce(list1, factor):
    list2 = [list1[0][i * factor] for i in range(int(list1[0].shape[0] / factor))]
    return [np.array(list2)]

def reduce_points(array, factor):
    return np.array([array[i * factor] for i in range(int(array.shape[0] / factor))])

def padding(array, num_cells=86400):
    padded_array = np.pad(array, [(0, num_cells - array.shape[0])], mode='constant', constant_values=-100.0)
    return padded_array.reshape((num_cells,))


def extract_boundaries_data(simulation_path, case_name):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [] }
    for entity in ['U', 'p', 'Cx', 'Cy']:
        vtk_list = []

        #boundaries
        for patch in ['inlet', 'obstacle', 'outlet', 'wall']:
          path = os.path.join(simulation_path, case_name, 'VTK', patch,  f'case_1_385.vtk')
          mesh = pv.read(path) 
          vtk_list.append(reduce_points(mesh.cell_data[entity],1))

        if entity == 'U':
          data['Ux'].append(padding(np.concatenate(vtk_list)[:,0], num_cells=5400))
          data['Uy'].append(padding(np.concatenate(vtk_list)[:,1], num_cells=5400))

        else:
          data[entity].append(padding(np.concatenate(vtk_list), num_cells=5400))

    return data



def extract_interior_data(simulation_path, case_name):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [] }
    for entity in ['U', 'p', 'Cx', 'Cy']:
        vtk_list = []
        path = os.path.join(simulation_path, case_name, 'VTK', f'case_1_385.vtk')
        mesh = pv.read(path)#.cell_arrays

        vtk_list.append(reduce_points(mesh.cell_data[entity],1))
        #vtk_list = reduce(vtk_list,10)        

        if entity == 'U':
          data['Ux'].append(padding(np.concatenate(vtk_list)[:,0]))
          data['Uy'].append(padding(np.concatenate(vtk_list)[:,1]))

        else:
          data[entity].append(padding(np.concatenate(vtk_list)))

    return data


simdirs = ["simulation_data/case_" + str(i) for i in range(1,11)]
total_sim = len(simdirs)
total_time = 1
num_points = 86400 #num of cells.
num_fields = 5 # (Ux, Uy, p, Cx, Cy)
train_shape = boundaries_shape = (total_sim,  num_points, num_fields)

hdf5_path = 'ml_data.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('sim_data', train_shape, np.float32)

import matplotlib.pyplot as plt
for i, case in enumerate(simdirs):
    data = extract_interior_data(".", case)
    print (data)

    hdf5_file['sim_data'][i, ..., 0] = data['Ux']
    hdf5_file['sim_data'][i, ..., 1] = data['Uy']
    hdf5_file['sim_data'][i, ..., 2] = data['p']
    hdf5_file['sim_data'][i, ..., 3] = data['Cx']
    hdf5_file['sim_data'][i, ..., 4] = data['Cy']
    
    #import pdb; pdb.set_trace()
    plt.scatter(data['Cx'][0], data['Cy'][0])
    plt.axis('scaled')
    plt.savefig('plot.png')

hdf5_file.close()
print ("Done !")




