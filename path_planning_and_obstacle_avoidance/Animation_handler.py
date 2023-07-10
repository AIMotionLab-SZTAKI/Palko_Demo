from matplotlib.animation import FuncAnimation
import time

from Util_files.Util_visualization import *
from Util_files.Util_trajectory_planning import *
from Trajectory_planning import generate_trajectory
from Classes import Construction, Drone


if __name__ == '__main__':
    t0 = time.time()

    # LOAD CONSTRUCTION
    scene = Construction()
    static_obstacles = pickle_load("Pickle_saves/Construction_saves/static_obstacles.pickle")
    dynamic_obstacles = [] # pickle_load("Pickle_saves/Construction_saves/dynamic_obstacles.pickle")
    number_of_targets = pickle_load("Pickle_saves/Construction_saves/number_of_targets.pickle")
    graph = pickle_load("Pickle_saves/Construction_saves/base_graph.pickle")

    # INIT PLOT
    plot_arena(scene)
    fig = plt.gcf()
    ax = fig.gca()
    demo_time = 20 # sec

    # PLACE STATIC OBSTACLES
    plot_static_obstacles(static_obstacles.corners, scene.obstacles_visibility)

    # PLACE DYNAMIC OBJECTS
    place_spheres(ax, dynamic_obstacles, scene.resoulution_of_obstacles, scene.obstacles_visibility, 'grey')
    add_coll_matrix_to_shepres(dynamic_obstacles, graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                               scene.general_safety_distance)

    # SETUP DRONES
    target_zero = len(graph['graph'].nodes()) - number_of_targets
    target_list = np.arange(target_zero, len(graph['graph'].nodes()), 1)
    scene.free_targets = target_list

    np.random.seed(225)

    drone_num = 5
    drones = []
    end_of_trajectories = []
    for i in range(drone_num):
        t1 = time.time()
        drone = Drone()
        drone.serial_number = i
        drone.start_vertex = np.random.choice(scene.free_targets)
        scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.start_vertex)
        drone.target_vetrex = np.random.choice(scene.free_targets)
        scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.target_vetrex)

        spline_path, speed_profile, fligth_time, length = generate_trajectory(drone, graph, dynamic_obstacles,
                                                                              static_obstacles, drones, scene.Ts,
                                                                              scene.cmin, scene.cmax,
                                                                              scene.general_safety_distance,
                                                                              scene.point_cloud_density)
        drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
        drone.flight_time = fligth_time

        #plot_trajectoty(drone.trajectory['spline_path'], length, i)
        add_coll_matrix_to_elipsoids([drone], graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                                     scene.general_safety_distance)
        drones.append(drone)
        end_of_trajectories.append(fligth_time)
        t2 = time.time()
        print("-------------------------------------------------------------------------------------")
        print("Start:", drone.start_vertex, "Target:", drone.target_vetrex)
        print("Path length:",  length, "m")
        print("Fligth time:", fligth_time, "sec")
        print("Trajectory generation:", t2 - t1, "sec")
    print("-------------------------------------------------------------------------------------")

    place_spheres(ax, drones, scene.resoulution_of_obstacles, 1, 'blue')
    #place_drones(ax, drones)

    time_init = time.time()
    print("Setup was done under", time_init-t0, "sec")

    # UPDATE ANIMATION
    def update(_):
        time_now = time.time() - time_init - scene.time_in_pause

        # Avoid:
            # TODO: unknown obs avoidance

        # Async:
        fastest = np.argmin(end_of_trajectories)
        if demo_time >= time_now >= end_of_trajectories[fastest]:
            t_start_calculation = time.time()
            fastest_drone = drones.pop(fastest)

            # Choose new target
            choose_target(scene, fastest_drone)

            # Generate trajectory
            fastest_drone.start_time = end_of_trajectories[fastest] + fastest_drone.rest_time
            spline_path, speed_profile, fligth_time, length = generate_trajectory(fastest_drone, graph, dynamic_obstacles,
                                                                                  static_obstacles, drones, scene.Ts,
                                                                                  scene.cmin, scene.cmax,
                                                                                  scene.general_safety_distance,
                                                                                  scene.point_cloud_density)

            # Update drone
            fastest_drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
            fastest_drone.flight_time = fligth_time
            #plot_trajectoty(fastest_drone.trajectory['spline_path'], length, fastest_drone.serial_number)
            add_coll_matrix_to_elipsoids([fastest_drone], graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                                         scene.general_safety_distance)

            # Update flight times and select the next drone
            drones.insert(fastest, fastest_drone)
            end_of_trajectories[fastest] = fastest_drone.start_time + fligth_time

            calculation_time = time.time() - t_start_calculation
            scene.time_in_pause = calculation_time + scene.time_in_pause
            print("-------------------------------------------------------------------------------------")
            print("Path length:", length, "m")
            print("Fligth time:", fligth_time, "sec")
            print("Time of generaion:", calculation_time, "sec")
            if calculation_time > fastest_drone.rest_time:
                print_WARNING("The calculation time (" + str(calculation_time) + " sec) was longer than the resting"
                              " time (" + str(fastest_drone.rest_time) + " sec).")

        # Plot:
        anim_ojbects(time_now, dynamic_obstacles)
        anim_ojbects(time_now, drones)
        #anim_drones(time_now, drones)

        if demo_time < time_now:
            tellme("End of demo")

    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()
