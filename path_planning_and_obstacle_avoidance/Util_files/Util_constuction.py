import copy
from scipy.spatial import Delaunay
import networkx as nx
import motioncapture
import matplotlib.pyplot as plt
from gurobipy import *

from path_planning_and_obstacle_avoidance.Util_files.Util_general import *
from path_planning_and_obstacle_avoidance.Classes import Dynamic_obstacle, Drone

#=======================================================================================================================
# FOR: STATIC OBSTACLES


def get_obstacles_positions_from_optitrack(new_measurement: bool):
    """
    Measure the positions of the obstacles and return a dictinary as {'name':position}.
    It evaluates the mean value of "sample_size" number of measurements.
    Gives a warning if the deviation is bigger than the "maximum_measurement_deviation".

    :param new_measurement: True -> make a new optittrack measurement and save it too
                            False -> use the last measurements
    :return: obstacles_dict:  {'name':position}, where: names -> optitrack rigidBody names,
                                                        positions -> coordinates of the top center of the obstacles [x,y,z]
    """

    if new_measurement:
        SAMPLE_SIZE = 100
        obstacles_dict = get_measurement(SAMPLE_SIZE)
        check_maximum_deviation(obstacles_dict)
        for name in obstacles_dict:
            obstacles_dict[name] = np.sum(obstacles_dict[name], axis=0) / len(obstacles_dict[name])
        pickle_save("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/obstacle_measurement.pickle", obstacles_dict)

    else:
        try:
            obstacles_dict = pickle_load("path_planning_and_obstacle_avoidance/Pickle_saves/Construction_saves/obstacle_measurement.pickle")
        except FileNotFoundError:
            print("\033[93mWARNING: No existing obstacle measurements available!!!\033[0m ")
            obstacles_dict = []

    return obstacles_dict


def get_measurement(sample_size: int):
    """
    Connect to the Optitrack data stream and collect "sample_size" number of measurements.

    :param sample_size: number of measurements
    :return: obstacles_dict: {'name':position}, where: names -> optitrack rigidBody names,
                                                       positions -> list of measured coordinates of the top center of
                                                                    the obstacles np.array([[x,y,z]...[x,y,z]])
    """
    mc = motioncapture.MotionCaptureOptitrack("192.168.1.141")
    obstacles_dict = {}
    for _ in range(sample_size):
        mc.waitForNextFrame()
        for name, obj in mc.rigidBodies.items():
            if name in obstacles_dict:
                obstacles_dict[name] = np.row_stack((obstacles_dict[name], obj.position))
            else:
                obstacles_dict[name] = obj.position

    return obstacles_dict


def check_maximum_deviation(obstacles_dict: dict):
    """
    Check the deviation of the measurements and give a warning if it is bigger than the "maximum_measurement_deviation".

    :param obstacles_dict: {'name':position}, where: names -> optitrack rigidBody names,
                                                     positions -> list of measured coordinates of the top center of
                                                                  the obstacles np.array([[x,y,z]...[x,y,z]])
    """
    max_deviation = 0
    PERMISSIBLE_DEVIATION = 1
    for name in obstacles_dict:
        deviation = max([max(obstacles_dict[name][:, 0]) - min(obstacles_dict[name][:, 0]),
                         max(obstacles_dict[name][:, 1]) - min(obstacles_dict[name][:, 1]),
                         max(obstacles_dict[name][:, 2]) - min(obstacles_dict[name][:, 2])])
        if deviation > max_deviation:
            max_deviation = deviation
    if max_deviation > PERMISSIBLE_DEVIATION:
        print_WARNING("The maximum deviation in the obstacle position measurments is " + str(max_deviation) +
                      "mm, which is bigger than the given "+str(PERMISSIBLE_DEVIATION) + "mm maximum!!!")


def add_dimension_to_obstacles(obstacle_measurements: dict, obstacles_side_lengths: dict):
    """
    Since the optitrack measurements only give thetop center coordinates of the static obstacles
    the width of them has to be set manualy. This function sets the width of the obstacles based on their names.

    The width of the obstacles can be set in Classes/Construction/self.real_obstacles_side_lengths

    :param obstacle_measurements: {'name':position}, where: names -> optitrack rigidBody names,
                                                     positions -> list of measured coordinates of the top center of
                                                                  the obstacles as np.array([[x,y,z]...[x,y,z]])
    :param obstacles_side_lengths: {'name':width}, where: names -> static oostacle names,
                                                          width -> float(w)
    :return: enclosed_space: main parameters of the obstacles as np.array([[x,y,z,w,w]...[x,y,z,w,w]])
    """
    enclosed_space = []
    start_points = []
    for obstacle in obstacle_measurements:
        x = obstacle_measurements[obstacle][0]
        y = obstacle_measurements[obstacle][1]
        z = obstacle_measurements[obstacle][2]
        if obstacle[0:2] == "bu":
            w = obstacles_side_lengths["buildings"] / 2
            enclosed_space.append([x, y, z, w, w])
        if obstacle[0:2] == "ob":
            w = obstacles_side_lengths["poles"] / 2
            enclosed_space.append([x, y, z, w, w])
        if obstacle[0:2] == "cf":
            w = obstacles_side_lengths["landing_pads"] / 2
            start_points.append([x, y, z, w, w])
    enclosed_space.extend(start_points)
    return np.array(enclosed_space)


def match_dimensions(obstacle_dimensions, obstacle_positions):
    """
    If the dimensions of the obstacle is the same for all abstacles generate an obstacle dimension set which number of
    rows is as the obstacles positions.
    """
    if len(obstacle_dimensions) == 1 and len(obstacle_positions) > 1:
        obstacle_dimensions = obstacle_dimensions * np.ones((len(obstacle_positions), 1))
    elif 1 < len(obstacle_dimensions) != len(obstacle_positions):
        sys.exit("Not matching obstacle positions and dimensions")
    return obstacle_dimensions


def calculate_corners(obstacles_params):
    """
    :param obstacles_params: positions and dimensions of the static obstacles
    :return: corner points of the static obstacles
    """
    corners_of_static_obstacles = []
    for obstacle in obstacles_params:
        obstacle_corners = []
        for corner_x, corner_y, corner_z in zip([-1, 1, 1, -1, -1, -1, 1, 1], [-1, -1, 1, 1, 1, -1, -1, 1],
                                                [-1, -1, -1, -1, 0, 0, 0, 0]):
            obstacle_corners.append([obstacle[0] + corner_x * obstacle[3], obstacle[1] + corner_y * obstacle[4],
                                     obstacle[2] + corner_z * obstacle[2]])
        corners_of_static_obstacles.append(obstacle_corners)
    return np.array(corners_of_static_obstacles)


def add_safety_zone_to_static_obstacles(enclosed_spaces, safety_distance):
    """
    The size of the obstacles are enlarged by the safety zone.
    If there are no static obstacles present return with an empty array.
    """
    if len(enclosed_spaces) == 0:
        return enclosed_spaces

    enclosed_space_of_safe_zone = copy.deepcopy(enclosed_spaces)
    enclosed_space_of_safe_zone[:, 2:] = enclosed_spaces[:, 2:]+[safety_distance, safety_distance, safety_distance]

    return enclosed_space_of_safe_zone


#=======================================================================================================================
# FOR: GRAPH GENERATION

def add_vertices_above_obstacles(obstacles, V_fix, z_max):
    """
    Create vertices above the static obstacles and add them to the fix vertices.
    Vertices above the flying zone are ignored and a warning is raised.
    """
    if len(obstacles) == 0:
        return V_fix

    hover_heigth = 0.2

    V_above_obstacles = obstacles[:, :3] + [0, 0, hover_heigth]
    V_inside_zone = V_above_obstacles[V_above_obstacles[:, 2] < z_max]

    if len(V_inside_zone) != len(V_above_obstacles):
        print("\033[93mWARNING: Some obstacles are too tall to place a vertex above them!!!\033[0m ")

    if len(V_fix) == 0:
        return V_inside_zone

    vertices = np.row_stack((V_fix, V_inside_zone))

    return vertices


def generate_base_graph(dimensions, static_obstacles, thres, number_of_vertices, rand_seed, V_fix):
    """
    Generate a graph around the static obstacles.
    """
    graph = nx.Graph()
    check_fix_vertices(V_fix, static_obstacles.enclosed_space_of_safe_zone)
    vertices = generate_vertices(dimensions, static_obstacles.enclosed_space_of_safe_zone, thres, number_of_vertices,
                                 rand_seed, V_fix)
    graph = load_vertices_to_graph(graph, vertices)
    edges = generate_edges(graph, static_obstacles.enclosed_space_of_safe_zone,
                           static_obstacles.corners_of_safe_zone)
    graph = load_edges_to_graph(graph, edges)

    return graph, vertices


def check_fix_vertices(V_fix, occupied_spaces):
    """
    Check if any fix vertex is inside an obstacle, which can cause some inconvinience later.
    Print a WARNING if needed but do not exit
    """
    V_fix_ = remove_vertices_in_obstacles(V_fix, occupied_spaces)
    if len(V_fix) != len(V_fix_):
        print("\033[93mWARNING: Some fix vertises are inside the obstacles!!!\033[0m ")


def generate_vertices(dimensions, occupied_spaces, thres, number_of_vertices, rand_seed, V_fix):
    """
    Generate random vertices outside from the static obstacles with a minimum distance from each other
    """
    enclosing_vertices = generate_enclosing_mesh(dimensions)
    random_vertices = generate_random_vertices(number_of_vertices, dimensions, rand_seed)
    vertices = np.row_stack((enclosing_vertices, random_vertices, V_fix))
    vertices = remove_redundant_vertices(V_fix, vertices, thres)
    vertices = remove_vertices_in_obstacles(vertices, occupied_spaces)

    return vertices


def generate_enclosing_mesh(dimensions):
    """
    Generate a mesh like border around the flying zone.
    Whitout it there will be undesiredly long edges at the sides of the flying zone.
    :param dimensions: the area of the flying zones
    :return: a 3 x n array containing the coordinates of the vertices which cover each side of the flying area
    """
    x_points = np.linspace(dimensions[0], dimensions[1], math.ceil((dimensions[1]-dimensions[0])*2))
    y_points = np.linspace(dimensions[2], dimensions[3], math.ceil((dimensions[3]-dimensions[2])*2))
    z_points = np.linspace(dimensions[4], dimensions[5], math.ceil((dimensions[5]-dimensions[4])*3))

    Xv, Yv = np.meshgrid(x_points, y_points)
    Xh, Zh = np.meshgrid(x_points, z_points)
    Yh, Zh = np.meshgrid(y_points, z_points)

    Xv = Xv.reshape((np.prod(Xv.shape),))
    Yv = Yv.reshape((np.prod(Yv.shape),))
    Zh = Zh.reshape((np.prod(Zh.shape),))
    Yh = Yh.reshape((np.prod(Yh.shape),))
    Xh = Xh.reshape((np.prod(Xh.shape),))

    bottom = np.ones(len(Xv)) * dimensions[4]
    enclosing_mesh = list(zip(Xv, Yv, bottom))
    top = np.ones(len(Xv)) * dimensions[5]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(Xv, Yv, top))))
    front = np.ones(len(Zh)) * dimensions[0]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(front, Yh, Zh))))
    back = np.ones(len(Zh)) * dimensions[1]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(back, Yh, Zh))))
    left = np.ones(len(Zh)) * dimensions[2]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(Xh, left, Zh))))
    right = np.ones(len(Zh)) * dimensions[3]
    enclosing_mesh = np.concatenate((enclosing_mesh, list(zip(Xh, right, Zh))))

    return enclosing_mesh


def generate_random_vertices(number_of_vertices, dimensions, rand_seed):
    """
    Generate given number of vertices scattered randomly inside the flying zone
    return:  number_of_vertices x 3 size array
    """
    DIMENSION = 3
    np.random.seed(rand_seed)
    random_vertices = np.multiply(dimensions[1::2] - dimensions[0::2],
                                  np.random.rand(number_of_vertices, DIMENSION)) - [dimensions[1], dimensions[3],
                                                                                    - dimensions[4]]

    return random_vertices


def remove_redundant_vertices(V_fix, vertices, thres):
    """
    remove vertices wich are closer to other vertices than the threshold value
    the fix vertices will be not removed
    """
    number_of_fix_vertices = len(V_fix)
    removal = []
    for i in range(len(vertices) - number_of_fix_vertices - 1):  # Removing vertices that are too close
        if min(np.linalg.norm(vertices[i, :] - vertices[i + 1:, :], axis=1)) < thres:
            removal.append(i)
    vertices = np.delete(vertices, removal, axis=0)

    return vertices


def remove_vertices_in_obstacles(vertices, occupied_spaces):
    """
    Remove the vertices that are inside any static obstacle
    """
    for occupied_space in occupied_spaces:
        # Sides of obstacle
        front_side = occupied_space[0] + occupied_space[3]
        back_side = occupied_space[0] - occupied_space[3]
        rigth_side = occupied_space[1] + occupied_space[4]
        left_side = occupied_space[1] - occupied_space[4]
        top = occupied_space[2]

        # check verteces outside from obstacle
        in_front_of = front_side < vertices[:, 0]
        behinde = back_side > vertices[:, 0]
        to_the_rigth = rigth_side < vertices[:, 1]
        to_the_left = left_side > vertices[:, 1]
        above = top < vertices[:, 2]

        outside = np.bitwise_or(in_front_of, behinde)
        outside = np.bitwise_or(outside, to_the_rigth)
        outside = np.bitwise_or(outside, to_the_left)
        outside = np.bitwise_or(outside, above)
        vertices = vertices[outside]

    return vertices


def load_vertices_to_graph(graph, vertices):
    """
    Add the vertices to the nx.Graph object, with their index and positions
    """
    index_start = len(graph) # needed for addind nodes to already exsting graph
    for i, position in enumerate(vertices):
        graph.add_node(i+index_start, pos=position)

    return graph


def generate_edges(graph, occupied_spaces, corners):
    """
    Generate edges with delanuay triangulation and remove those which intersect any static obstacles.
    It gives back an adjacenci matrix wich is processeed by tho load_edges function
    """
    vertices = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    tri = Delaunay(vertices)  # Delaunay triangulation of a set of points
    # Del_tri:[A,B,C],[A,B,D] -> Adj_graph:A[B,C,D],B[A,C,D],C[A,B],D[A,B]
    graph_adj = [list() for _ in range(len(vertices))]
    for simplex in tri.simplices:
        for j in range(4):
            a = simplex[j]
            graph_adj[a].extend(simplex)
            graph_adj[a].remove(a)
            graph_adj[a] = remove_duplicates(graph_adj[a])
            graph_adj[a] = [e for e in graph_adj[a] if not intersect(vertices[a], vertices[e], occupied_spaces,
                                                                     corners)]

    return graph_adj


def remove_duplicates(duplist):
    """
    Ensure that the adjacency matrix contains the vertices only once
    """
    noduplist = []
    for i in duplist:
        if i not in noduplist:
            noduplist.append(i)
    return noduplist


def intersect(v1, v2, occupied_spaces, corners):
    """
    Check if an edge is intersecting a static obstacle or not
    """
    xmin = min(v1[0], v2[0])
    xmax = max(v1[0], v2[0])
    ymin = min(v1[1], v2[1])
    ymax = max(v1[1], v2[1])
    zmin = min(v1[2], v2[2])

    for i, occupied_space in enumerate(occupied_spaces):
        front_side = occupied_space[0] + occupied_space[3]
        back_side = occupied_space[0] - occupied_space[3]
        rigth_side = occupied_space[1] + occupied_space[4]
        left_side = occupied_space[1] - occupied_space[4]
        top = occupied_space[2]

        if xmax < back_side or xmin > front_side or ymax < left_side or ymin > rigth_side or zmin > top:
            continue

        a1, b1 = equation_plane(corners[i][0], corners[i][1], corners[i][2])
        a2, b2 = equation_plane(corners[i][4], corners[i][6], corners[i][5])
        a3, b3 = equation_plane(corners[i][0], corners[i][4], corners[i][5])
        a4, b4 = equation_plane(corners[i][0], corners[i][5], corners[i][6])
        a5, b5 = equation_plane(corners[i][2], corners[i][7], corners[i][4])
        a6, b6 = equation_plane(corners[i][7], corners[i][1], corners[i][6])

        dist = multiDimenDist(v1, v2)
        q = findVec(v1, v2, True)

        opt_mod = Model("intersection")
        opt_mod.setParam('OutputFlag', False)
        opt_mod.setParam('TimeLimit', 0.01)
        lmd = opt_mod.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=dist, name='lmd')
        opt_mod.setObjective(1, GRB.MINIMIZE)
        opt_mod.addConstr(a1.dot(np.transpose(v1)) + lmd * (a1.dot(np.transpose(q))) <= b1)
        opt_mod.addConstr(a2.dot(np.transpose(v1)) + lmd * (a2.dot(np.transpose(q))) <= b2)
        opt_mod.addConstr(a3.dot(np.transpose(v1)) + lmd * (a3.dot(np.transpose(q))) <= b3)
        opt_mod.addConstr(a4.dot(np.transpose(v1)) + lmd * (a4.dot(np.transpose(q))) <= b4)
        opt_mod.addConstr(a5.dot(np.transpose(v1)) + lmd * (a5.dot(np.transpose(q))) <= b5)
        opt_mod.addConstr(a6.dot(np.transpose(v1)) + lmd * (a6.dot(np.transpose(q))) <= b6)
        opt_mod.optimize()

        try:
            lmd = opt_mod.objVal  # Inside
            return True
        except AttributeError as e:
            pass

    return False


def equation_plane(p1, p2, p3):
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    b = np.dot(cp, p3)
    return a, b


def multiDimenDist(point1, point2):
    # find the difference between the two points, it's really the same as below
    deltaVals = [point2[dimension] - point1[dimension] for dimension in range(len(point1))]
    runningSquared = 0
    # because the pythagarom theorm works for any dimension we can just use that
    for coOrd in deltaVals:
        runningSquared += coOrd ** 2
    return runningSquared ** (1 / 2)


def findVec(point1, point2, unitSphere):
    # unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
    finalVector = [0 for coOrd in point1]
    for dimension, coOrd in enumerate(point1):
        # finding total differnce for that co-ordinate(x,y,z...)
        deltaCoOrd = point2[dimension] - coOrd
        # adding total difference
        finalVector[dimension] = deltaCoOrd
    if unitSphere:
        totalDist = multiDimenDist(point1, point2)
        unitVector = []
        for dimen in finalVector:
            unitVector.append(dimen / totalDist)
        return unitVector
    else:
        return finalVector


def load_edges_to_graph(graph, edges):
    """
    Load the edges to the nx.Graph object with the edge lengths as weights
    """
    for vertex, neighbours in enumerate(edges):
        for neighbour in neighbours:
            graph.add_edge(vertex, neighbour, weight=np.linalg.norm(graph.nodes.data('pos')[vertex] -
                                                                    graph.nodes.data('pos')[neighbour]))

    return graph


def extend_base_graph(scene, static_obstacles, thres, number_of_vertices, rand_seed, base_vertices):
    """
    Generate a graph with more vertices and edges based on the input graph.
    """
    graph = nx.Graph()
    graph = copy_vertices(scene.base_graph, graph)
    graph = dense_base_edges(scene.base_graph, graph)
    vertices_all, vertices_extra = generate_vertices_dense(scene, static_obstacles.enclosed_space_of_safe_zone, thres,
                                                           number_of_vertices, rand_seed, base_vertices)
    graph = load_vertices_to_graph(graph, vertices_extra)
    edges = generate_edges(graph, static_obstacles.enclosed_space_of_safe_zone,
                           static_obstacles.corners_of_safe_zone)
    graph = load_edges_to_graph(graph, edges)

    return graph


def copy_vertices(base_graph, graph):
    """
    Load the vertices of the original graph to the new nx.Graph object
    """
    for vertex in base_graph.nodes.data('pos'):
        graph.add_node(vertex[0], pos=vertex[1])

    return graph


def dense_base_edges(base_graph, graph):
    """
    Add edges and new vertices to the nx.Graph object based on the edges of the original  graph.
    The new edges are shorter and chained together to replace the original edges.
    """
    for edge in base_graph.edges.data('weight'):
        vertex_1 = base_graph.nodes.data('pos')[edge[0]]
        vertex_2 = base_graph.nodes.data('pos')[edge[1]]
        edge_length = edge[2]

        number_of_plus_vertices = math.ceil(edge_length / 0.1 + 0.7)
        plus_vertices_on_edge = np.linspace(vertex_1, vertex_2, number_of_plus_vertices)

        index_start = len(graph.nodes())
        index_stop = len(plus_vertices_on_edge) - 2

        # if there is no need for adding an extra vertex to the edge
        if index_stop == 0:
            graph.add_edge(edge[0], edge[1], weight=edge_length)
            continue

        for i, position in enumerate(plus_vertices_on_edge):
            if i == index_stop:
                graph.add_edge(index_start + i - 1, edge[1], weight=np.linalg.norm(vertex_2 - position))
                break
            graph.add_node(index_start + i, pos=position)
            if i == 0:
                graph.add_edge(edge[0], index_start, weight=np.linalg.norm(vertex_1 - position))
            else:
                graph.add_edge(index_start + i - 1, index_start + i, weight=np.linalg.norm(plus_vertices_on_edge[i - 1]
                                                                                           - position))

    return graph


def generate_vertices_dense(scene, occupied_spaces, thres, number_of_vertices, rand_seed, V_fix):
    """
    Generate random vertices outside from the static obstacles with a minimum distance from each other
    """
    random_vertices = generate_random_vertices(number_of_vertices, scene.dimensions, rand_seed)
    vertices = np.row_stack((random_vertices, V_fix))
    vertices = remove_redundant_vertices(V_fix, vertices, thres)
    vertices = remove_vertices_in_obstacles(vertices, occupied_spaces)
    # print vertex data
    print("Number of random extra vertices:", len(random_vertices))
    print("Number of all vertices:", len(vertices))
    print("Number of nonredundant vertices:", len(vertices))

    return vertices, vertices[:-len(V_fix)]


def create_point_cloud(graph, density):
    first_iteration = True
    point_cloud = None
    for edge in list(graph.edges):
        v0 = graph.nodes.data('pos')[edge[0]]
        v1 = graph.nodes.data('pos')[edge[1]]
        edge_length = np.linalg.norm(v0-v1)
        point_number = math.ceil(edge_length / density)
        edge_points = np.linspace(v0, v1, point_number)
        if first_iteration:
            point_cloud = edge_points
            first_iteration = False
        else:
            point_cloud = np.row_stack((point_cloud, edge_points))
        graph[edge[0]][edge[1]]["point_range"] = [len(point_cloud)-len(edge_points), len(point_cloud)-1]

    return graph, point_cloud


def solve_target_point_collisions(graph, point_cloud, number_of_targets, safety_distance):
    """
    Add the index of the target vertices to the edges which are in the corresponding taget zone
    """
    ax = plt.gca()
    example_drone = Drone()
    radius = example_drone.radius
    downwash = example_drone.DOWNWASH
    vertices = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    dist_m = calculate_eplis_rel_dist(vertices[-number_of_targets:], point_cloud, downwash, radius, safety_distance)
    nx.set_edge_attributes(graph, None, 'touching_targets')
    for edge in graph.edges.data():
        touching_targets = np.unique(np.where(dist_m[:, edge[2]['point_range'][0]:edge[2]['point_range'][1]] <= 1)[0])
        edge[2]['touching_targets'] = touching_targets + (len(graph.nodes())-number_of_targets)


def print_graph_info(G, point_cloud):
    print("Number of points:", len(point_cloud))
    print("Number of vertices:", len(G.nodes))
    print("Nuber of edges:", len(G.edges))
    sum_L = 0
    max_L = 0
    for edge in G.edges.data():
        sum_L = sum_L + edge[2]['weight']
        if max_L < edge[2]['weight']:
            max_L = edge[2]['weight']
    print("Average edge length:", sum_L/len(G.edges), "m")
    print("Longest edge is: ", max_L, "m")




#=======================================================================================================================
# FOR: DYNAMIC OBSTACLES

def load_paths():
    """
    Try to load in previously generated moving obstacles paths
    """
    try:
        existing_paths = pickle_load("Pickle_saves/Construction_saves/paths_of_dynamic_obstacles.pickle")
        return existing_paths
    except (OSError, IOError):
        return []


def generate_dynamic_obstacles(paths_points, individual_speeds, default_speed, individual_radii, default_radius, desired_paths):
    dynamic_obstacles = []
    ID = 0
    for i, points in enumerate(paths_points):
        if i not in desired_paths:
            continue

        spline = fit_spline(points)
        spline, length = parametrize_by_path_length(spline)

        if i in individual_speeds[:, 0]:
            speed = individual_speeds[i, 1]
        else:
            speed = default_speed
        if i in individual_radii[:, 0]:
            radius = individual_radii[i, 1]
        else:
            radius = default_radius

        dynamic_obstacles.append(Dynamic_obstacle(path_tck=spline, path_length=length, speed=speed, radius=radius))
        ID += 1
    return dynamic_obstacles
