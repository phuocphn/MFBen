import glob
import os
from tqdm import tqdm


GEO_DIR = "extra"
GEO_DIR = "low2high_fidelity_500nodes/test"
simdir = f"simulation_data/{GEO_DIR}"
maxIter = 100000
curr_path = os.path.dirname(os.path.abspath(__file__))
pbar = tqdm(glob.glob(simdir + "/*"), desc="Generating ...")
for fd in pbar:
    #print (fd)
    os.chdir(curr_path)
    os.chdir(fd)
    pbar.set_description ("working on: " + fd)
    
    os.system("rm -rf [1-9]*")
    os.system("rm -rf VTK")
    os.system("rm -rf 0/Ux 0/Uy 0/Uz 0/C*")
    os.system("simpleFoam > sim.log")
    basename = os.path.basename(fd)
    os.system(f"touch {basename}.foam")

    '''
    if os.path.exists(str(maxIter)):
        print (f"simulation is not coverged, deleting {fd}")
        os.chdir(curr_path)
        os.system(f"rm -rf {fd}")
        continue
    '''
    #os.system("postProcess -func writeCellCentres > log.cells")
    os.system("postProcess -funcs '(components(U))' > writeComponents.log")
    #os.system("foamToVTK -ascii -legacy> foamToVTK.log")
    os.system("foamToVTK -ascii > foamToVTK.log")
