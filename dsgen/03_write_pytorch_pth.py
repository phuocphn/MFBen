import os
import numpy as np
import h5py
import glob
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import argparse
from tqdm import tqdm
import click
from typing import List, Tuple
from vtk.util.numpy_support import vtk_to_numpy
fields = ["Cx", "Cy", "Ux", "Uy", "p"]
total_cells = 5000

from typing import Any, Union

import torch
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

matplotlib.use("Agg")


def visualize_single(
    x_coord: Union[np.array, torch.tensor] = [],
    y_coord: Union[np.array, torch.tensor] = [],
    exact_u: Union[np.array, torch.tensor] = [],
    exact_v: Union[np.array, torch.tensor] = [],
    exact_p: Union[np.array, torch.tensor] = [],
    fig_title: str = "__cmp__",
    save_path: str = "fig.png",
    kwargs: dict = {},
):
    # assert x_coord.shape[-1] > 500
    # extent = -0.25, 0.65, -0.1, 0.1

    if torch.is_tensor(x_coord):
        x_coord = x_coord.cpu().detach().numpy()
    if torch.is_tensor(y_coord):
        y_coord = y_coord.cpu().detach().numpy()
    if torch.is_tensor(exact_u):
        exact_u = exact_u.cpu().detach().numpy()
    if torch.is_tensor(exact_v):
        exact_v = exact_v.cpu().detach().numpy()
    if torch.is_tensor(exact_p):
        exact_p = exact_p.cpu().detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.suptitle(fig_title)

    plt.subplot(111)
    plt.ylabel("p:exact", fontsize=15)
    plt.scatter(x_coord, y_coord, c=exact_p, cmap="jet")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def read_vtk(filename, display_name, show_info=True):
    """Read the content of VTK file and return a dict contains all
    filed data.

    :param filename: path to the VTK file
    :param display: show in the printing log, if the `show_info` enabled.
    :return: a dict with keys are physical field names, and values are physical data.
    """
    data = dict()
    mesh = pv.read(filename)

    if show_info:
        # import pudb; pu.db
        print("------------------------------")
        print("mesh name:", display_name)
        # print("num cells:", mesh.cell_data["Cx"].shape[0])
        # print("num points:", mesh.point_data["Cx"].shape[0])

    # data["num_cells"] = mesh.cell_data["Cx"].shape[0]
    data["num_points"] = mesh.GetNumberOfPoints()#mesh.point_data["Cx"].shape[0]

    # mesh.GetPointData().GetArray('Ux')
    # from vtk.util.numpy_support import vtk_to_numpy
    # GetNumberOfPoints
    # GetPoint(i)
    Px = np.zeros((data["num_points"], 1))
    Py = np.zeros((data["num_points"], 1))

    for i in range(mesh.GetNumberOfPoints()):
        x, y, _ = mesh.GetPoint(i)
        # print (mesh.GetPoint(i))
        Px[i, 0] = x
        Py[i, 0] = y


    Ux = vtk_to_numpy(mesh.GetPointData().GetArray('Ux')).reshape(data["num_points"], 1)
    Uy = vtk_to_numpy(mesh.GetPointData().GetArray('Uy')).reshape(data["num_points"], 1)
    p = vtk_to_numpy(mesh.GetPointData().GetArray('p')).reshape(data["num_points"], 1)

    data['Cx'] = Px
    data['Cy'] = Py
    data['Ux'] = Ux
    data['Uy'] = Uy
    data['p'] = p
    # import pudb ; pu.db

    for field in fields:
        if show_info:
            print(
                f"   field: {field}, min={data[field].min()},  max={data[field].max()}"
            )

    if show_info:
        print("------------------------------")

    visualize_single(Px.flatten(), Py.flatten(), Ux.flatten(), Uy.flatten(), p.flatten(), display_name, display_name+".png")
    return data


def create_matrix(interior_vtk: str, boundary_vtks: Tuple[str, str], out_dir):
    used_points = 0

    interior_data = read_vtk(interior_vtk, "interior")
    boundary_data = []
    boundary_names = []

    for boundary_vtk in boundary_vtks:
        patch_data = read_vtk(*boundary_vtk)

        numpy_d = np.concatenate([patch_data[field] for field in fields], axis=1)
        boundary_data.append(numpy_d)
        boundary_names.append(boundary_vtk[-1])

    interior_matrix: np.array = np.concatenate(
        [interior_data[field] for field in fields], axis=1
    )
    boundary_matrix: np.array = np.vstack(boundary_data)
    used_points = boundary_matrix.shape[0]

    print("interior_matrix.shape: ", interior_matrix.shape)
    print("boundary_matrix.shape: ", boundary_matrix.shape)



    collocation_tensor = torch.tensor(interior_matrix, dtype=torch.float32)
    boundary_tensor = torch.tensor(boundary_matrix, dtype=torch.float32)
    combined_tensor = torch.cat([collocation_tensor, boundary_tensor], dim=0).numpy()

    assert collocation_tensor.shape[-1] ==5
    assert combined_tensor.shape[-1] == 5, combined_tensor.shape
    # print ("combined_tensor.shape", combined_tensor.shape)
    u_min = np.min(combined_tensor[:,2])
    u_max = np.max(combined_tensor[:,2])
    v_min = np.min(combined_tensor[:,3])
    v_max = np.max(combined_tensor[:,3])
    p_min = np.min(combined_tensor[:,4])
    p_max = np.max(combined_tensor[:,4])


    normalized_collocation_tensor = collocation_tensor.clone()
    normalized_boundary_tensor = boundary_tensor.clone()

    normalized_collocation_tensor[:,2] = (normalized_collocation_tensor[:,2] - u_min)/(u_max - u_min)
    normalized_collocation_tensor[:,3] = (normalized_collocation_tensor[:,3] - v_min)/(v_max - v_min)
    normalized_collocation_tensor[:,4] = (normalized_collocation_tensor[:,4] - p_min)/(p_max - p_min)
    normalized_boundary_tensor[:,2] = (normalized_boundary_tensor[:,2] - u_min)/(u_max - u_min)
    normalized_boundary_tensor[:,3] = (normalized_boundary_tensor[:,3] - v_min)/(v_max - v_min)
    normalized_boundary_tensor[:,4] = (normalized_boundary_tensor[:,4] - p_min)/(p_max - p_min)

    # print (collocation_tensor, collocation_tensor.shape)
    # print (boundary_tensor, boundary_tensor.shape)
    # print (collocation_points)

    torch.save(collocation_tensor.T, os.path.join(out_dir, "u+collocation_points.pt"))
    torch.save(boundary_tensor.T, os.path.join(out_dir, "u+boundary_points.pt"))
    torch.save(normalized_collocation_tensor.T, os.path.join(out_dir, "n+collocation_points.pt"))
    torch.save(normalized_boundary_tensor.T, os.path.join(out_dir, "n+boundary_points.pt"))
    
    for matrix_data, bc in zip(boundary_data,boundary_names):
        boundary_tensor = torch.tensor(matrix_data, dtype=torch.float32)
        normalized_boundary_tensor = boundary_tensor.clone()

        normalized_boundary_tensor[:,2] = (normalized_boundary_tensor[:,2] - u_min)/(u_max - u_min)
        normalized_boundary_tensor[:,3] = (normalized_boundary_tensor[:,3] - v_min)/(v_max - v_min)
        normalized_boundary_tensor[:,4] = (normalized_boundary_tensor[:,4] - p_min)/(p_max - p_min)

        torch.save(boundary_tensor.T, os.path.join(out_dir, f"u+boundary_points+{bc}.pt"))
        torch.save(normalized_boundary_tensor.T, os.path.join(out_dir, f"n+boundary_points+{bc}.pt"))


    print (f"boundary points={boundary_tensor.shape[0]}, collocation_points={collocation_tensor.shape[0]}, collocation_tensor={collocation_tensor.shape[0]}")


    return interior_data, boundary_data


def create_fixed_matrix(interior_vtk: str, boundary_vtks: Tuple[str, str]):
    used_points = 0

    interior_data = read_vtk(interior_vtk, "interior")
    boundary_data = []

    for boundary_vtk in boundary_vtks:
        patch_data = read_vtk(*boundary_vtk)

        numpy_d = np.concatenate([patch_data[field] for field in fields], axis=1)
        boundary_data.append(numpy_d)

    interior_matrix: np.array = np.concatenate(
        [interior_data[field] for field in fields], axis=1
    )
    boundary_matrix: np.array = np.vstack(boundary_data)
    used_points = boundary_matrix.shape[0]

    print("interior_matrix.shape: ", interior_matrix.shape)
    print("boundary_matrix.shape: ", boundary_matrix.shape)
    remaining_points = total_cells - boundary_matrix.shape[0]

    idx = np.random.choice(
        range(interior_matrix.shape[0]), size=remaining_points, replace=False
    )
    assert len(list(set(idx))) == remaining_points
    #import pudb; pu.db
    combined_matrix: np.array = np.vstack([interior_matrix[idx, :], boundary_matrix])
    print("combined_matrix.shape: ", combined_matrix.shape)
    print("number of boundary points: ", used_points)
    return combined_matrix, used_points


@click.group(chain=True)
@click.option('--common-option1')
@click.option('--common-option2')
def main(common_option1, common_option2):
    pass


@main.command()
@click.option('--sim_dir', default='simulation_data/', prompt='OpenFOAM Simulation Dir', help='dir to OpenFOAM case')
@click.option('--out_dir', prompt='Output Dir',default="/tmp", help='output directory for saving .pt files.')
@click.option('--max_iters', prompt='Maxium Iters', default=2000, help='maximum number of iterations when run OpenFOAM simulation.')

def gen_single(sim_dir, out_dir, max_iters):
    # print (sim_dir, out_dir)

    maxIter=max([int(d) for d in os.listdir(sim_dir) if d.isnumeric()])
    if maxIter == max_iters or maxIter == 0:
        print ("ignoring: ", sim_dir)
        return 1
    
    interiror_vtk = os.path.join(
        sim_dir, "VTK", os.path.basename(sim_dir) + "_" + str(maxIter) + ".vtk"
    )

    boundary_vtks = [
        (os.path.join(sim_dir, "VTK", patch, patch + "_" + str(maxIter) + ".vtk"), patch)
        for patch in ["obstacle", "wall", "inlet", "outlet"]
    ]

    data, num_boundary_points = create_matrix(interiror_vtk, boundary_vtks, out_dir)
    
    print (f"vtkFile: {interiror_vtk} \nvtkBoundaryFiles: {boundary_vtks}")
    return 0


@main.command()
@click.option('--sim_dir', default='simulation_data/', prompt='OpenFOAM Simulation Dir', help='dir to OpenFOAM case')
@click.option('--out_dir', prompt='Output Dir',default="/tmp", help='output directory for saving .pt files.')
@click.option('--max_iters', prompt='Maxium Iters', default=2000, help='maximum number of iterations when run OpenFOAM simulation.')

def gen_multiple(sim_dir, out_dir, max_iters):
    # print (sim_dir, out_dir)i
    ignore_cases = []
    for sim_case in glob.glob(os.path.join(sim_dir, "*")):
        maxIter=max([int(d) for d in os.listdir(sim_case) if d.isnumeric()])
        if maxIter == max_iters or maxIter == 0:
            print ("ignoring: ", sim_case)
            ignore_cases.append(sim_case)
            continue
        
        interiror_vtk = os.path.join(
            sim_case, "VTK", os.path.basename(sim_case) + "_" + str(maxIter) + ".vtk"
        )

        boundary_vtks = [
            (os.path.join(sim_case, "VTK", patch, patch + "_" + str(maxIter) + ".vtk"), patch)
            for patch in ["obstacle", "wall", "inlet", "outlet"]
        ]
        
        from pathlib import Path
        save_dir = os.path.join(out_dir, os.path.basename(sim_case))
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        data, num_boundary_points = create_matrix(interiror_vtk, boundary_vtks, save_dir)
        
        print (f"vtkFile: {interiror_vtk} \nvtkBoundaryFiles: {boundary_vtks}")
    print ("Ignore cases:", ignore_cases)

@main.command()
@click.option('--sim_dir', default='simulation_data/', prompt='OpenFOAM Simulation Dir', help='dir to OpenFOAM case')
@click.option('--out_dir', prompt='Output Dir',default="/tmp", help='output directory for saving .pt files.')
@click.option('--max_iters', prompt='Maxium Iters', default=2000, help='maximum number of iterations when run OpenFOAM simulation.')

def get_bcinfo(sim_dir, out_dir, max_iters):
    # print (sim_dir, out_dir)i
    ignore_cases = []
    for sim_case in glob.glob(os.path.join(sim_dir, "*")):
        maxIter=max([int(d) for d in os.listdir(sim_case) if d.isnumeric()])
        if maxIter == max_iters or maxIter == 0:
            print ("ignoring: ", sim_case)
            ignore_cases.append(sim_case)
            continue
        
        interiror_vtk = os.path.join(
            sim_case, "VTK", os.path.basename(sim_case) + "_" + str(maxIter) + ".vtk"
        )

        boundary_vtks = [
            (os.path.join(sim_case, "VTK", patch, patch + "_" + str(maxIter) + ".vtk"), patch)
            for patch in ["obstacle", "wall", "inlet", "outlet"]
        ]
        print (f"========={sim_case}=========")
        infox = {}
        for boundary_vtk in boundary_vtks:
            patch_data = read_vtk(*boundary_vtk, show_info=False)

            numpy_d = np.concatenate([patch_data[field] for field in fields], axis=1)
            infox[boundary_vtk[-1]] = numpy_d.shape[0]
        print (infox)

        


if __name__ == '__main__':
    main()

'''
simdirs = glob.glob(f"{args.simulation_data}/*")
# sanity check
for i, case in enumerate(tqdm(simdirs)):
    case_name = case
    maxIter=max([int(d) for d in os.listdir(case) if d.isnumeric()])
    if maxIter == args.max_iters or maxIter == 0:
        print ("Case: ", case, " not converged!")
        print ("To avoid potential training errors, consider removing the case")
        exit()

print ("sanity check done!")
total_sim = len(simdirs)
num_fields = 5 # (Cx, Cy, Ux, Uy, p)
train_shape = boundaries_shape = (total_sim,  args.num_cells, num_fields)

hdf5_path = args.save_path
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('interior_data', train_shape, np.float32)
hdf5_file.create_dataset('boundary_data', train_shape, np.float32)

#raw_values = {'Ux': [], 'Uy': [], 'p': []}
normalization_info = {
        'u_max': -float("inf"), 'u_min': float("inf"),
        'v_max': -float("inf"), 'v_min': float("inf"),
        'p_max': -float("inf"), 'p_min': float("inf"),
}
pbar = tqdm(simdirs)
for i, case in enumerate(pbar):
    case_name = case
    pbar.set_description("working on:  " + case)
    maxIter=max([int(d) for d in os.listdir(case) if d.isnumeric()])
    if maxIter == args.max_iters or maxIter == 0:
        print ("ignoring: ", case)
        continue
    
    vtkFile = os.path.join(case, "VTK", os.path.basename(case) + "_" + str(maxIter) + ".vtk")
    interior_data, raw_data = extract_interior_data(vtkFile)

    #patches=['inlet', 'obstacle', 'outlet', 'wall']

    patches = ['obstacle', 'wall']
    #pdb.set_trace()
    vtkBoundaryFiles = [
               os.path.join(case, "VTK", patch, patch+"_" + str(maxIter) + ".vtk") for patch in patches]
    
    #pdb.set_trace()


    hdf5_file['interior_data'][i, ..., 0] = interior_data['Cx']
    hdf5_file['interior_data'][i, ..., 1] = interior_data['Cy']
    hdf5_file['interior_data'][i, ..., 2] = interior_data['Ux']
    hdf5_file['interior_data'][i, ..., 3] = interior_data['Uy']
    hdf5_file['interior_data'][i, ..., 4] = interior_data['p']

    
    boundary_data, raw_datab = extract_boundary_data(vtkBoundaryFiles)
    hdf5_file['boundary_data'][i, ..., 0] = boundary_data['Cx']
    hdf5_file['boundary_data'][i, ..., 1] = boundary_data['Cy']
    hdf5_file['boundary_data'][i, ..., 2] = boundary_data['Ux']
    hdf5_file['boundary_data'][i, ..., 3] = boundary_data['Uy']
    hdf5_file['boundary_data'][i, ..., 4] = boundary_data['p']
    #raw_values['Ux'].append(raw_data['Ux'])
    #raw_values['Uy'].append(raw_data['Uy'])
    #raw_values['p'].append(raw_data['p'])

    normalization_info['u_max'] = max(normalization_info['u_max'], np.max(np.concatenate([raw_data['Ux'], raw_datab['Ux']])))
    normalization_info['u_min'] = min(normalization_info['u_min'], np.min(np.concatenate([raw_data['Ux'], raw_datab['Ux']])))
    normalization_info['v_max'] = max(normalization_info['v_max'], np.max(np.concatenate([raw_data['Uy'], raw_datab['Uy']])))
    normalization_info['v_min'] = min(normalization_info['v_min'], np.min(np.concatenate([raw_data['Uy'], raw_datab['Uy']])))

    normalization_info['p_max'] = max(normalization_info['p_max'], np.max(np.concatenate([raw_data['p'], raw_datab['p']])))
    normalization_info['p_min'] = min(normalization_info['p_min'], np.min(np.concatenate([raw_data['p'], raw_datab['p']])))


hdf5_file.close()

# info = dict(
#         min_u = np.min(np.concatenate(raw_values['Ux'])),
#         max_u = np.max(np.concatenate(raw_values['Ux'])),
#         min_v = np.min(np.concatenate(raw_values['Uy'])),
#         max_v = np.max(np.concatenate(raw_values['Uy'])),
#         min_p = np.min(np.concatenate(raw_values['p'])),
#         max_p = np.max(np.concatenate(raw_values['p'])),
# )
# print (info)

print (normalization_info)
print ("creating .hdf5 done!")

'''
