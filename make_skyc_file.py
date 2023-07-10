import numpy as np
from path_planning_and_obstacle_avoidance.Scene_construction import construction
from path_planning_and_obstacle_avoidance.Classes import Construction, Drone
import pickle
from path_planning_and_obstacle_avoidance.Trajectory_planning import *
import re
import sys
import trio
from trio import sleep
import json
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, List, Union, Any, cast, Callable, Dict
from scipy.special import comb
import os
import shutil
import zipfile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gurobipy
# from time import time as real_time
from trio import current_time
from math import ceil
import motioncapture


def cleanup(files: List[str], folders: List[str]):
    # function meant for deleting unnecessary files
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted {folder} folder")


def write_trajectory(Data, type: str):
    # this is the format that a TrajectorySpecification requires:
    assert type == "POLY4D" or type == "COMPRESSED"
    if type == "COMPRESSED":
        assert degree == 3
    json_dict = {
        "version": 1,
        "points": Data,
        "takeoffTime": Data[0][0],
        "landingTime": Data[-1][0],
        "type": type
    }
    json_object = json.dumps(json_dict, indent=2)
    with open("trajectory.json", "w") as f:
        f.write(json_object)


def write_to_skyc(Skyc_Data, type: str):
    # delete every file that we can generate that might have been left over from previous sessions
    name = "Palko_Demo"
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   f"{name}.skyc",
                   "trajectory.json"],
            folders=["drones"])
    # Create the 'drones' folder if it doesn't already exist
    os.makedirs('drones', exist_ok=True)
    drones = []
    for index, Data in enumerate(Skyc_Data):
        # The trajectory is saved to a json file with the data below
        write_trajectory(Data, type=type)

        lights = {
            "data": "BqQUDAME//8ABQoFAg8NBP//AAUJTAUCjAEKlgEA",
            "version": 1
        }
        json_object = json.dumps(lights, indent=2)
        with open("lights.json", "w") as f:
            f.write(json_object)

        drone_settings = {
            "trajectory": {"$ref": f"./drones/drone_{index}/trajectory.json#"},
            "lights": {"$ref": f"./drones/drone_{index}/lights.json#"},
            "home": Data[0][1],
            "landAt": Data[-1][1],
            "name": f"drone_{index}"
        }
        drones.append({
            "type": "generic",
            "settings": drone_settings
        })

        # Create the 'drone_1' folder if it doesn't already exist
        drone_folder = os.path.join('drones', f'drone_{index}')
        os.makedirs(drone_folder, exist_ok=True)


        # Copy 'lights.json' and 'trajectory.json' to 'drone_1' folder
        shutil.move('lights.json', drone_folder)
        shutil.move('trajectory.json', drone_folder)
    # This wall of text below is just overhead that is required to make a skyc file.
    ########################################CUES.JSON########################################
    items = [{"time": Skyc_Data[0][0],
              "name": "start"}]
    cues = {
        "version": 1,
        "items": items
    }
    json_object = json.dumps(cues, indent=2)
    with open("cues.json", "w") as f:
        f.write(json_object)
    #######################################SHOW.JSON###########################################
    validation = {
        "maxAltitude": 2.0,
        "maxVelocityXY": 2.0,
        "maxVelocityZ": 1.5,
        "minDistance": 0.8
    }
    cues = {
        "$ref": "./cues.json"
    }
    settings = {
        "cues": cues,
        "validation": validation
    }
    meta = {
        "id": f"{name}.py",
        "inputs": [f"{name}.py"]
    }
    show = {
        "version": 1,
        "settings": settings,
        "swarm": {"drones": drones},
        "environment": {"type": "indoor"},
        "meta": meta,
        "media": {}
    }
    json_object = json.dumps(show, indent=2)
    with open("show.json", "w") as f:
        f.write(json_object)

    # Create a new zip file
    with zipfile.ZipFile(f"{name}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the first file to the zip
        zipf.write("show.json")

        # Add the second file to the zip
        zipf.write("cues.json")

        # Recursively add files from the specified folder and its sub-folders
        for root, _, files in os.walk("drones"):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path)

    print('Compression complete. The files and folder have been zipped as demo.zip.')

    os.rename(f'{name}.zip', f'{name}.skyc')
    # Delete everything that's not 'trajectory.skyc'
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   "trajectory.json",
                   "lights.json"],
            folders=["drones"])
    print("Demo skyc file ready!")


def determine_knots(time_vector, N):
    '''returns knot vector for the BSplines according to the incoming timestamps and the desired number of knots'''
    # Problems start to arise when part_length becomes way smaller than N, so generally keep them longer :)
    part_length = len(time_vector) // N
    result = time_vector[::part_length][:N]
    result.append(time_vector[-1])
    return result


degree = 3
drone_IDs = ["06", "07"]
main_folder_name = "trajectories"
current_dir = os.getcwd()
main_folder_path = os.path.join(current_dir, main_folder_name)
Skyc_Data = []
REST_TIME = 3
takeoff_time = 3
number_of_bezier_segments = 40
for drone_ID in drone_IDs:
    sub_folder_path = os.path.join(main_folder_path, drone_ID)
    pickle_folder_path = os.path.join(sub_folder_path, "pickle")
    Bezier_Data = []
    for file_name in sorted(os.listdir(pickle_folder_path)):
        file_path = os.path.join(pickle_folder_path, file_name)
        with open(file_path, 'rb') as f:
            t, x, y, z = pickle.load(f)
            if len(Bezier_Data) < 1:  # if this is our first segment, we have to add a takeoff segment
                Bezier_Data = [[0, [x[0], y[0], 0], []],  # start here
                               [takeoff_time, [x[0], y[0], z[0]], []]]  # take off to the start position in takeoff_time
            last_time = Bezier_Data[-1][0]
            last_point = Bezier_Data[-1][1]
            Bezier_Data.append([last_time+REST_TIME, last_point, []])  # add a hover before the new segment
            t = [x + last_time + REST_TIME for x in t]
            # plt.plot(t, y)
            knots = determine_knots(t, number_of_bezier_segments)[1:-1]
            xSpline = interpolate.splrep(t, x, k=degree, task=-1, t=knots)
            ySpline = interpolate.splrep(t, y, k=degree, task=-1, t=knots)
            zSpline = interpolate.splrep(t, z, k=degree, task=-1, t=knots)
            x_PPoly = interpolate.PPoly.from_spline(xSpline)
            y_PPoly = interpolate.PPoly.from_spline(ySpline)
            z_PPoly = interpolate.PPoly.from_spline(zSpline)
            x_BPoly = interpolate.BPoly.from_power_basis(x_PPoly)
            y_BPoly = interpolate.BPoly.from_power_basis(y_PPoly)
            z_BPoly = interpolate.BPoly.from_power_basis(z_PPoly)
            BPoly = list(zip(list(x_BPoly.x)[degree + 1:-degree],
                             list(x_BPoly.c.transpose())[degree:-degree],
                             list(y_BPoly.c.transpose())[degree:-degree],
                             list(z_BPoly.c.transpose())[degree:-degree]))
            BPoly = [[element[0]] + list(zip(*list(element[1:]))) for element in BPoly]
            plot_time_start = last_time
            for Bezier_Curve in BPoly:
                # ######plotting######
                plot_time = np.linspace(plot_time_start, Bezier_Curve[0])
                plot_time_start = Bezier_Curve[0]
                bernstein_x = interpolate.BPoly(np.array([element[0] for element in Bezier_Curve[1:]]).reshape(degree+1, 1), np.array([0, 1]))
                bernstein_y = interpolate.BPoly(np.array([element[1] for element in Bezier_Curve[1:]]).reshape(degree+1, 1), np.array([0, 1]))
                bernstein_z = interpolate.BPoly(np.array([element[2] for element in Bezier_Curve[1:]]).reshape(degree+1, 1), np.array([0, 1]))
                plt.plot(plot_time, bernstein_y(np.linspace(0, 1)))
                # ######plotting######
                curve_to_append = [Bezier_Curve[0],
                                   Bezier_Curve[-1],
                                   Bezier_Curve[2:-1]]
                Bezier_Data.append(curve_to_append)
    land_segment = [Bezier_Data[-1][0] + takeoff_time,
                    [Bezier_Data[-1][1][0], Bezier_Data[-1][1][1], 0],
                    []]
    Bezier_Data.append(land_segment)
    Skyc_Data.append(Bezier_Data)

# t = [segment[0] for segment in Skyc_Data[0]]
# x = [segment[1][1] for segment in Skyc_Data[0]]
# plt.plot(t, x)
plt.show()

write_to_skyc(Skyc_Data, type="COMPRESSED")