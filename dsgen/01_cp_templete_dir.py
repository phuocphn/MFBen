import glob
import os
import pathlib

import os
import numpy as np
import random
from itertools import product

def updateBoundaryFile():
	mode = 0
	with open('boundary', 'w') as fw:
		with open('constant/polyMesh/boundary', 'r') as fr:
			for line in fr:
				# line = line.replace("\n", "")
				if "physicalType" in line:
					continue
				elif "frontAndBack" in line:
					mode = 1
				elif "wall" in line:
					mode = 2
				elif "obstacle" in line: 
					mode = 3
				elif "inlet" in line:
					mode = 4
				elif "outlet" in line:
					mode = 5
				# else:
				# 	mode = 0

				if "type" in line:
					# print ("found: ", line, "mode: ", mode)
					if mode == 1:
						fw.write(line.replace("patch", "empty"))
					elif mode == 2 or mode == 3:
						fw.write(line.replace("patch", "wall"))
					else:
						fw.write(line)
					continue
					pass

				fw.write(line)

				# print (line)
	os.system("rm constant/polyMesh/boundary")
	os.system("mv boundary constant/polyMesh/")
 
 
GEO_DIR = "low2high_fidelity_500nodes/test"
SIM_DIR = f"simulation_data/{GEO_DIR}"
pathlib.Path(SIM_DIR).mkdir(parents=True, exist_ok=True)
for fn in glob.glob(f"geo/{GEO_DIR}/*.geo"):

    casename = os.path.basename(fn).replace(".geo","")
    #print (dirname)
    os.system(f"cp -r template {SIM_DIR}/{casename}")
    os.system(f"cp -r {fn} {SIM_DIR}/{casename}")

    cwd = os.getcwd()
    #import pudb; pu.db
    os.chdir(f"{SIM_DIR}/{casename}")
    os.system(f"gmsh {casename}.geo -3 -format msh2")
    os.system(f"gmshToFoam {casename}.msh")
    updateBoundaryFile()
    
    os.chdir(cwd)





