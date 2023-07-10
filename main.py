import numpy as np
from path_planning_and_obstacle_avoidance.Scene_construction import construction
from path_planning_and_obstacle_avoidance.Classes import Construction, Drone
import pickle
from path_planning_and_obstacle_avoidance.Trajectory_planning import *


number_of_targets, graph = construction()
scene = Construction()
demo_time = 20  # sec

# SETUP DRONES
target_zero = len(graph['graph'].nodes()) - number_of_targets
target_list = np.arange(target_zero, len(graph['graph'].nodes()), 1)
scene.free_targets = target_list

np.random.seed(11)
drone_num = 2
drones = []
end_of_trajectories = []

home_positions = scene.free_targets[-drone_num:]
scene.free_targets = scene.free_targets[:-drone_num]


for i in range(drone_num):
    drone = Drone()
    drone.serial_number = i
    drone.start_vertex = home_positions[i]
    drone.target_vetrex = np.random.choice(scene.free_targets)
    # bar the other drones from selecting the node we're going to
    scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.target_vetrex)
    spline_path, speed_profile, _, _ = generate_trajectory(drone=drone, G=graph, dynamic_obstacles=[],
                                                           other_drones=drones, Ts=scene.Ts,
                                                           safety_distance=scene.general_safety_distance)

    drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
    drone.flight_time = drone.trajectory['speed_profile'][0][-1]
    add_coll_matrix_to_elipsoids([drone], graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                                 scene.general_safety_distance)
    drones.append(drone)


drone_idx = 0
choose_target(scene, drones[drone_idx])