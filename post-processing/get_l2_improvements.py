import os
import glob
import csv
import json
import numpy as np
import itertools
import pandas as pd


def dir2res(result_dir) -> dict:
    """Obtain the average L2 u/v/p errors across different geometries (square_2, rectangle_1, etc)

    :param result_dir: dir path to the experiment logs, the dir path should be similar to: `experiment-data/non-kd/mlpconv.h6.n64/`
    :return: a dict contains avg u/v/p values
    """
    u_dict = []
    v_dict = []
    p_dict = []
    for json_file in glob.glob(os.path.join(result_dir, "results", "*.json")):
        with open(json_file, "r") as fp:
            data = json.load(fp)
            u_dict.append(data["avg.u"])
            v_dict.append(data["avg.v"])
            p_dict.append(data["avg.p"])

    num_cases = len(u_dict)
    return {
        "avg.u": sum(u_dict) / num_cases,
        "avg.v": sum(v_dict) / num_cases,
        "avg.p": sum(p_dict) / num_cases,
    }


def multidirs2tab(case_dirs) -> dict:
    """Obtain the average L2/ u,v,p errors across different network configurations (mlpconv.h3.n128, etc)

    :param case_dirs: _description_
    :return: _description_
    """

    data = []
    for num_hidden, num_neurons in list(itertools.product([3, 6, 10], [32, 64, 128])):
        result_dir = os.path.join(case_dirs, f"mlpconv.h{num_hidden}.n{num_neurons}")
        result_data = dir2res(result_dir)
        data.append(result_data | {"dir": result_dir})

    # obtain the pointnetcfd (if available)
    result_dir = os.path.join(case_dirs, f"pointnetcfd")
    if os.path.exists(result_dir):
        result_data = dir2res(result_dir)
        data.append(result_data | {"dir": result_dir})

    return data


def multidict2pd(data):
    df = pd.DataFrame.from_dict(data)
    return df


if __name__ == "__main__":
    print(dir2res("experiment-data/non-kd/mlpconv.h6.n64/"))
    print(dir2res("experiment-data/non-kd/mlpconv.h6.n64"))
    print(multidirs2tab("experiment-data/non-kd/"))

    data = multidirs2tab("experiment-data/non-kd/")
    df = multidict2pd(data)

    with pd.ExcelWriter("data.xlsx") as writer:
        df.to_excel(writer, sheet_name="non-kd")

    # kd
    for kd_T in [1, 2, 5, 10, 32]:
        data = multidirs2tab(f"experiment-data/kd/{kd_T}")
        df = multidict2pd(data)

        with pd.ExcelWriter("data.xlsx", mode="a") as writer:
            df.to_excel(writer, sheet_name=f"kd-{kd_T}")
