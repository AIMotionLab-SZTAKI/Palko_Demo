import cProfile
import pstats
import time

from .Classes import Construction, Drone
from .Util_files.Util_trajectory_planning import *


def generate_trajectory(drone, G, dynamic_obstacles, other_drones, Ts, safety_distance):

    # FIND ROUTE
    route, speed = find_route(drone.constant_speeds, G['graph'], dynamic_obstacles, other_drones,
                              drone.start_vertex, drone.target_vetrex, drone.start_time)
    #route = simplify_route(Ts, cmin, cmax, speed, G, dynamic_obstacles, other_drones, static_obstacles,
    #                       drone.start_vertex, drone.target_vetrex, drone.start_time, point_cloud_density,
    #                       safety_distance, route)

    # FIT SPLINE
    line_buffer = 5
    spline_points = extend_route(route, G['graph'], line_buffer)
    spline = fit_spline(spline_points)
    spline_path, length = parametrize_by_path_length(spline)

    # DESIGN SPEED PROFILE
    speed_profile, flight_time = optimize_speed_profile(drone, other_drones, dynamic_obstacles, spline_path, length,
                                                        speed, Ts, safety_distance)

    # STANDARD RETURN
    if flight_time is not None:
        return spline_path, speed_profile, flight_time, length
    #...................................................................................................................
    # HANDLE STANDING OBSTACLES
    print_WARNING("Finished drone/dynamic obstacle is blocking the way.")

    store_dyobs = [obs.collision_matrix for obs in dynamic_obstacles]
    store_drones = [obs.collision_matrix for obs in other_drones]

    for obs in dynamic_obstacles:
        occupied_points = np.where(obs.collision_matrix[-1, :] > 0)[0][1:]
        obs.collision_matrix[:, occupied_points] = np.inf

    # FIND ROUTE
    route, speed = find_route(drone.constant_speeds, G['graph'], dynamic_obstacles, other_drones,
                              drone.start_vertex, drone.target_vetrex, drone.start_time)

    # FIT SPLINE
    spline_points = extend_route(route, G['graph'], line_buffer)
    spline = fit_spline(spline_points)
    spline_path, length = parametrize_by_path_length(spline)

    # DESIGN SPEED PROFILE
    speed_profile, flight_time = optimize_speed_profile(drone, other_drones, dynamic_obstacles, spline_path, length,
                                                        speed, Ts, safety_distance)

    for i, coll_matrix in enumerate(store_dyobs):
        dynamic_obstacles[i].collision_matrix = coll_matrix
    for i, coll_matrix in enumerate(store_drones):
        other_drones[i].collision_matrix = coll_matrix

    if flight_time is None:
        print_WARNING("DEADLOCK")
 
    return spline_path, speed_profile, flight_time, length


if __name__ == '__main__':
    """
    Measure detailed computational times.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    graph = pickle_load("Pickle_saves/Construction_saves/base_graph.pickle")
    dynamic_obstacles = pickle_load("Pickle_saves/Construction_saves/dynamic_obstacles.pickle")
    static_obstacles = pickle_load("Pickle_saves/Construction_saves/static_obstacles.pickle")
    number_of_targets = pickle_load("Pickle_saves/Construction_saves/number_of_targets.pickle")
    scene = Construction()
    add_coll_matrix_to_shepres(dynamic_obstacles, graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                               scene.general_safety_distance)

    target_zero = len(graph['graph'].nodes()) - number_of_targets
    targets = np.arange(target_zero, len(graph['graph'].nodes()), 1)

    np.random.seed(225)
    drone_num = 5
    past_drones = []

    for i in range(drone_num):
        t0 = time.time()
        drone = Drone()
        drone.serial_number = i
        drone.start_vertex = np.random.choice(targets)
        targets = np.delete(targets, targets == drone.start_vertex)
        drone.target_vetrex = np.random.choice(targets)
        targets = np.delete(targets, targets == drone.target_vetrex)
        spline_path, speed_profile, fligth_time, length = generate_trajectory(drone, graph, dynamic_obstacles,
                                                                              static_obstacles, past_drones, scene.Ts,
                                                                              scene.cmin, scene.cmax,
                                                                              scene.general_safety_distance,
                                                                              scene.point_cloud_density)

        drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
        drone.flight_time = fligth_time
        add_coll_matrix_to_elipsoids([drone], graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                                     scene.general_safety_distance)
        past_drones.append(drone)

        t1 = time.time()
        print("-------------------------------------------------------------------------------------")
        print("Trajectory generation:", t1 - t0, "sec")
    print("-------------------------------------------------------------------------------------")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()