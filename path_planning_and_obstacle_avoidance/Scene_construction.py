import cProfile
import pstats


from .Util_files.Util_constuction import *
from .Util_files.Util_visualization import *
from .Classes import Construction, Static_obstacles


# ======================================================================================================================
    # CONSTRUCTION MAIN
def construction():
    """
    The scene constructor sets up the static obstacles and the paths of the virtual moving obstacles.
    It also generates the searching graphs.

    The static obstacles can be given virtually or scanned with Optitrack.
    !!! The dimensions have to be set up manually either way at Classes -> Static_obstacles !!!
    """
# ======================================================================================================================
    # DIMENSIONS OF THE  FLYING ARENA
    scene = Construction()

    plot_arena(scene)

# ======================================================================================================================
    # PLACE STATIC OBSTACLES
    static_obstacles = Static_obstacles()

    if scene.real_obstacles:
        obstacle_measurements = get_obstacles_positions_from_optitrack(scene.get_new_measurement)
        static_obstacles.enclosed_space = add_dimension_to_obstacles(obstacle_measurements,
                                                                     scene.real_obstacles_side_lengths)
    else:
        static_obstacles.enclosed_space = select_fix_obstacle_set(scene.static_obstacle_layout)

    static_obstacles.corners = calculate_corners(static_obstacles.enclosed_space)
    static_obstacles.enclosed_space_of_safe_zone = add_safety_zone_to_static_obstacles(static_obstacles.enclosed_space,
                                                                                       scene.general_safety_distance)
    static_obstacles.corners_of_safe_zone = calculate_corners(static_obstacles.enclosed_space_of_safe_zone)

    plot_static_obstacles(static_obstacles.corners, scene.obstacles_visibility)
    plot_static_obstacles(static_obstacles.corners_of_safe_zone,  scene.safety_zone_visibility)

# ======================================================================================================================
    # GRAPH GENERATION
    V_fix = select_fix_vertex_set(scene.fix_vertex_layout)
    V_fix = add_vertices_above_obstacles(static_obstacles.enclosed_space_of_safe_zone, V_fix, scene.dimensions[-1])

    base_graph, base_vertices = generate_base_graph(scene.dimensions, static_obstacles,
                                                    scene.base_minimal_vertex_distance, scene.base_vertex_number,
                                                    scene.base_rand_seed, V_fix)
    base_graph, base_point_cloud = create_point_cloud(base_graph, scene.point_cloud_density)
    solve_target_point_collisions(base_graph, base_point_cloud, len(V_fix), scene.general_safety_distance)
    print_graph_info(base_graph, base_point_cloud)
    if scene.generate_dense_graph:
        densed_graph = extend_base_graph(scene, static_obstacles, scene.dense_minimal_vertex_distance,
                                         scene.dense_vertex_number, scene.dense_rand_seed, base_vertices)
        densed_graph, densed_point_cloud = create_point_cloud(densed_graph, scene.point_cloud_density)
        solve_target_point_collisions(densed_graph, densed_point_cloud, len(V_fix), scene.general_safety_distance)
        print_graph_info(base_graph, base_point_cloud)
    else:
        densed_graph = None
        densed_point_cloud = None

    if scene.show_base_graph:
        plot_graph(graph=base_graph, size=1, color='black')
    if scene.show_dense_graph and scene.generate_dense_graph:
        plot_graph(graph=densed_graph, size=0.5, color='grey')
    if scene.show_fix_vertices:
        plot_vertices(V_set=V_fix, size=5, color='blue')

# ======================================================================================================================
    # ADD SIMULATED DYNAMIC OBSTACLES
    if scene.keep_existing_paths:
        paths_points = load_paths()
    else:
        paths_points = []

    if scene.show_paths_of_dynamic_obstacles:
        for points in paths_points:
            spline = fit_spline(points)
            plot_spline(spline)

    if scene.ask_for_new_paths:
        new_paths_points = ask_for_paths(scene.dimensions)
        paths_points.extend(new_paths_points[:])

    dynamic_obstacles = generate_dynamic_obstacles(paths_points, scene.speed_of_individual_dynamic_obstacles,
                                                   scene.default_speed, scene.radii_of_individual_dynamic_obstacles,
                                                   scene.default_radius, scene.path_of_dynamic_obstacles)

# ======================================================================================================================
    # SAVES

    pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/number_of_targets.pickle",
                len(V_fix))
    pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/base_graph.pickle",
                {'graph': base_graph, 'point_cloud': base_point_cloud})
    pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/densed_graph.pickle",
                {'graph': densed_graph, 'point_cloud': densed_point_cloud})
    pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/static_obstacles.pickle",
                static_obstacles)
    pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/paths_of_dynamic_obstacles.pickle",
                paths_points)
    pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/dynamic_obstacles.pickle",
                dynamic_obstacles)
    print("ALL SAVED")
    plt.show()
    graph = {
        "graph": base_graph,
        "point_cloud": base_point_cloud
    }
    return len(V_fix), graph
# ======================================================================================================================
    # SELECTIONS

def select_fix_obstacle_set(index_of_obstacle_set):
    """
    :param: index_of_obstacle_set: the serial number of the choosen obstacle set
    the position of an obstacle is the center of top of it
    the dimensions of an obstacle is the width (x direction) and depth (y direction) of it
    :return: the obstacle parameters (if there wewre no matching obstacle set retun an empty obstacles set)
    """
    if index_of_obstacle_set == 0:
        return np.array([])
    elif index_of_obstacle_set == 1:
        obstacle_positions = np.array([[1.2, 1.3, 2]])
        obstacle_dimensions = np.array([[0.6, 0.6]])/2
    elif index_of_obstacle_set == 2:
        obstacle_positions = np.array([[0, 0, 1], [0.5, 0.5, 2]])
        obstacle_dimensions = np.array([[0.75, 0.2], [0.5, 0.5]])/2
    else:
        return np.array([])

    obstacle_dimensions = match_dimensions(obstacle_dimensions, obstacle_positions)
    return np.column_stack((obstacle_positions, obstacle_dimensions))


def select_fix_vertex_set(index_of_verex_set):
    """
    V_fix = [[x_0,y_0,z_0],[x_1,y_1,z_1],...[x_N,y_N,z_N]]
    """
    if index_of_verex_set == 0:
        return np.array([])
    elif index_of_verex_set == 1:
        # Shape: X_________________________
        #           03/19 02/18 01/17 00/16|
        #           07/23 06/22 05/21 04/20|   2 layer
        #           11/27 10/26 09/25 08/24|
        #           15/31 14/30 13/29 12/28|
        #                                  Y
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 1.1
        low_layer = 0.4
        offset = 0.15

        V_fix = [[-outer_xy, -outer_xy, high_layer], [-inner_xy, -outer_xy, high_layer], [inner_xy, -outer_xy, high_layer], [outer_xy, -outer_xy, high_layer],
                 [-outer_xy, -inner_xy, high_layer], [-inner_xy - offset, -inner_xy - offset, high_layer], [inner_xy + offset, -inner_xy - offset, high_layer], [outer_xy, -inner_xy, high_layer],
                 [-outer_xy, inner_xy, high_layer], [-inner_xy - offset, inner_xy + offset, high_layer], [inner_xy + offset, inner_xy + offset, high_layer], [outer_xy, inner_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [-inner_xy, outer_xy, high_layer], [inner_xy, outer_xy, high_layer], [outer_xy, outer_xy, high_layer]]
    elif index_of_verex_set == 2:
        # Shape: X_________________________
        #           03 02 01 00|
        #           07       04|   2 layer
        #           11       08|
        #           15 14 13 12|
        #                                  Y
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 1.1

        V_fix = [[-outer_xy, -outer_xy, high_layer], [-inner_xy, -outer_xy, high_layer], [-outer_xy, -inner_xy, high_layer], [outer_xy, -outer_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [outer_xy, inner_xy, high_layer], [inner_xy, outer_xy, high_layer], [outer_xy, outer_xy, high_layer]]
    elif index_of_verex_set == 3:
        # Shape: X_________________________
        #           03 02 01 00|
        #           07       04|   2 layer
        #           11       08|
        #           15 14 13 12|
        #                                  Y
        inner_xy = 0.4
        outer_xy = 3 * inner_xy
        high_layer = 1.1

        V_fix = [[-outer_xy, -outer_xy, high_layer], [outer_xy, -outer_xy, high_layer],
                 [-outer_xy, outer_xy, high_layer], [outer_xy, outer_xy, high_layer]]
    else:
        return np.array([])
    return np.array(V_fix)


# ======================================================================================================================
    # MAIN
if __name__ == '__main__':
    """
    Measure detailed computational times.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    construction()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    #stats.print_stats()
    plt.show()
