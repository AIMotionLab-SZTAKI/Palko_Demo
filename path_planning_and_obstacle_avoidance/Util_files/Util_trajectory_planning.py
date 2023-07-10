from queue import PriorityQueue
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

from .Util_constuction import intersect, create_point_cloud
from .Util_general import *


def add_coll_matrix_to_shepres(obstacles, point_cloud, Ts, cmin, cmax, safety_distance):
    """
    Fill the collision matrix parameter of the obastacles.
    The collision matrix has the costs of the points of the graph edges in different time points.
    Handle the obstacles as floating spheres.
    """
    drone_r = 0.12

    for obs in obstacles:
        fligth_time = math.ceil((obs.path_length/obs.speed)*10)/10
        time_grid = np.arange(0, fligth_time + Ts, Ts)
        positions = obs.move(time_grid)
        dist_matrix = distance_matrix(positions, point_cloud) # (time x edge)
        rel_distances = dist_matrix/(obs.radius+drone_r+2*safety_distance)
        coll_matrix = (1 - rel_distances) * (cmax - cmin)
        coll_matrix = np.where(coll_matrix >= 0, cmin + coll_matrix, 0)
        obs.collision_matrix = np.column_stack((time_grid, coll_matrix))


def add_coll_matrix_to_elipsoids(drones, point_cloud, Ts, cmin, cmax, safety_distance):
    """
    Fill the collision matrix parameter of the drones.
    The collision matrix has the costs of the points of the graph edges in different time points.
    Handle the obstacles as floating elipsoids.
    """
    for drone in drones:
        time_grid = np.arange(0, drone.flight_time + Ts, Ts) + drone.start_time
        positions = drone.move(time_grid)
        rel_distances = calculate_eplis_rel_dist(positions, point_cloud, drone.DOWNWASH, drone.radius, safety_distance)
        coll_matrix = (1 - rel_distances) * (cmax - cmin)
        coll_matrix = np.where(coll_matrix >= 0, cmin + coll_matrix, 0)
        drone.collision_matrix = np.column_stack((time_grid, coll_matrix))


def find_route(speeds, G, dynamic_obstacles, other_drones, start_point, target_point, start_time):
    """
    Find the best route in the graph from the start point to the target point.
    """
    final_route = np.array([])
    time_of_final_route = 1000000
    best_speed = speeds[0]

    for speed in speeds:
        route, time_of_route = A_star(G, dynamic_obstacles, other_drones, start_point, target_point, start_time, speed)
        if time_of_route < time_of_final_route:
            time_of_final_route = time_of_route
            final_route = route
            best_speed = speed

    return final_route, best_speed


def simplify_route(Ts, cmin, cmax, speed, G, dynamic_obstacles, other_drones, static_obstacles, start_point,
                   target_point, start_time, point_cloud_density, safety_distance, base_route):
    G_mini = generate_mini_graph(base_route, G, static_obstacles.enclosed_space_of_safe_zone,
                                 static_obstacles.corners_of_safe_zone)
    G_mini, mini_point_cloud = create_point_cloud(G_mini, point_cloud_density)
    store_dyobs = [obs.collision_matrix for obs in dynamic_obstacles]
    store_drones = [obs.collision_matrix for obs in other_drones]
    add_coll_matrix_to_shepres(dynamic_obstacles, mini_point_cloud, Ts, cmin, cmax, safety_distance)
    add_coll_matrix_to_elipsoids(other_drones, mini_point_cloud, Ts, cmin, cmax, safety_distance)
    simple_route, _ = A_star(G_mini, dynamic_obstacles, other_drones, start_point, target_point, speed)
    for i, coll_matrix in enumerate(store_dyobs):
        dynamic_obstacles[i].collision_matrix = coll_matrix
    for i, coll_matrix in enumerate(store_drones):
        other_drones[i].collision_matrix = coll_matrix

    return simple_route


def A_star(G, dynamic_obstacles, other_drones, start_point, target_point, start_time, drone_speed):
    """
    A modified A* algorithm, which ties to avoid the dynamic obstacles.
    """
    nv = max(G.nodes) + 1
    visited = np.full(nv, False, dtype=bool)
    prev = np.zeros(nv)
    tt = np.full(nv, np.inf)
    tt[start_point] = 0
    cost = np.full(nv, np.inf)
    cost[start_point] = 0
    pq = PriorityQueue()
    pq.put((0, start_point))
    target_position = G.nodes.data('pos')[target_point]

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        if current_vertex == target_point:
            break
        visited[current_vertex] = True
        t = tt[current_vertex]
        for neighbour in list(G[current_vertex].keys()):
            if visited[neighbour]:
                continue
            # Time to reach the target in a straight line
            neighbours_position = G.nodes.data('pos')[neighbour]
            time_to_target = np.linalg.norm(neighbours_position-target_position)/drone_speed
            # Time cost of the neighbour
            dt = list(G[neighbour][current_vertex].values())[0] / drone_speed
            # Collisions with dínamic obstacles while traversing the edge
            tspan = np.array([t, t + dt]) + start_time
            edege_collisions = G[current_vertex][neighbour]['point_range']
            cc = 0
            cc = summ_collision_costs(dynamic_obstacles, tspan, edege_collisions, cc)
            cc = summ_collision_costs(other_drones, tspan, edege_collisions, cc)
            if np.any(len(G[current_vertex][neighbour]['touching_targets'])) and \
               (not (any(start_point == G[current_vertex][neighbour]["touching_targets"]) or
                     any(target_point == G[current_vertex][neighbour]["touching_targets"]))):
                    cc = np.inf

            # Update route
            old_cost = cost[neighbour]
            new_cost = cost[current_vertex] + dt + cc + time_to_target
            if new_cost < old_cost:
                pq.put((new_cost, neighbour))
                cost[neighbour] = new_cost
                prev[neighbour] = current_vertex
                tt[neighbour] = tt[current_vertex] + dt

    vk = target_point
    k = nv - 1
    route = np.zeros(nv)
    while vk != start_point:
        route[k] = vk
        k = k - 1
        vk = int(prev[vk])
    route[k] = vk
    route = route.astype(int)
    route = route[k:]
    time_of_route = tt[route][-1]

    return route, time_of_route


def summ_collision_costs(obstacles, tspan, edege_collisions, cc):
    for obs in obstacles:
        time_zone = np.where((obs.collision_matrix[:, 0] >= (tspan[0] - 0.1)) &
                             (obs.collision_matrix[:, 0] <= (tspan[1] + 0.1)))[0]

        if any(time_zone):
            collision_cost = np.sum(obs.collision_matrix[time_zone, edege_collisions[0]:edege_collisions[1]+1])
        else:
            if tspan[1] < obs.collision_matrix[0, 0]:
                collision_cost = np.sum(obs.collision_matrix[0, edege_collisions[0]:edege_collisions[1] + 1])
            else:
                collision_cost = np.sum(obs.collision_matrix[-1, edege_collisions[0]:edege_collisions[1] + 1])

        # if the obstacles arrives its destination or does not start moving yet
        if (tspan[1] >= obs.collision_matrix[-1, 0] or tspan[0] <= obs.collision_matrix[0, 0]) and collision_cost > 0:
            cc = np.inf

        cc = cc + collision_cost
    return cc


def generate_mini_graph(route, G, occupied_spaces, corners):
    """
    Constuct a mini graph based on the route found by the A*
    """
    graph_adj = {}
    for i, vertex in enumerate(route):
        graph_adj[vertex] = route[i + 1:]
        graph_adj[vertex] = [neighbour for neighbour in graph_adj[vertex] if not intersect(G.nodes()[vertex]['pos'],
                                                                                           G.nodes()[neighbour]['pos'],
                                                                                           occupied_spaces, corners)]
    G_mini = nx.Graph()
    for i in route:
        G_mini.add_node(i, pos=G.nodes()[i]['pos'])
    for i in graph_adj:
        for j in graph_adj[i]:
            G_mini.add_edge(i, j, weight=abs(np.linalg.norm(G.nodes()[i]['pos'] - G.nodes()[j]['pos'])))

    return G_mini


def extend_route(route, G, LINE_BUFFER):
    """
    Adds aditional points to the route to keep close the spline which will be fitted to it later
    """
    spline_points = np.array([])
    for i in range(len(route) - 1):
        v1 = G.nodes.data('pos')[route[i]]
        v2 = G.nodes.data('pos')[route[i+1]]
        if i == 0:
            spline_points = np.linspace(v1, v2, 2 + LINE_BUFFER)[:-1]
        else:
            spline_points = np.row_stack((spline_points, np.linspace(v1, v2, 2 + LINE_BUFFER)[:-1]))
    spline_points = np.row_stack((spline_points, G.nodes.data('pos')[route[-1]]))

    return spline_points


def optimize_speed_profile(drone, other_drones, dynamic_obstacles, spline_path, length, speed, Ts, safety_distance):
    J_p = 5
    J_v = 1
    flight_time_multiplier = np.array([2, 4, 8, 16])

    time_windows = flight_time_multiplier * (length / speed)
    for other_drone in other_drones:
        time_windows = time_windows[time_windows > other_drone.flight_time]

    for time_window in time_windows:
        H = math.ceil(time_window / Ts)
        tgrid = np.arange(0, H * Ts, Ts) + drone.start_time
        sgrid = np.linspace(0, length, math.ceil(length / (0.25 * drone.radius)))
        drone_pos = np.transpose(interpolate.splev(sgrid, spline_path))
        table = np.zeros((len(tgrid), 201))
        table[:, 0] = np.transpose(tgrid)
        collision_counter = 0
        table, collision_counter = fill_table_spheres(dynamic_obstacles, drone_pos, tgrid, sgrid, drone.radius,
                                                      safety_distance, table, collision_counter)
        table, collision_counter = fill_table_elipsoids(other_drones, drone_pos, tgrid, sgrid, drone.radius,
                                                        drone.DOWNWASH, safety_distance, table, collision_counter)
        # TODO: poles

        opt_mod = run_gurobi(drone, collision_counter, table, H, Ts, length, tgrid, speed, J_p, J_v)

        try:
            ans = opt_mod.objVal
            break
        except AttributeError as e:
            print("Not enough time")
            if time_window == time_windows[-1]:
                return None, None
            else:
                pass

    s_result = [var.x for var in opt_mod.getVars() if "s" in var.VarName]
    if len(tgrid) != len(s_result):
        tgrid = np.append(tgrid, tgrid[-1] + Ts)

    s_result, tgrid = trim_trajectory(s_result, tgrid)
    speed_profile = interpolate.splrep(tgrid, s_result, k=5)
    flight_time = tgrid[-1]-drone.start_time

    return speed_profile, flight_time


def fill_table_spheres(dynamic_obstacles, drone_pos, tgrid, sgrid, drone_radius, safety_distance, table,
                       collision_counter):
    for obs in dynamic_obstacles:
        active = False
        obs_pos = obs.move(tgrid)
        d_m = distance_matrix(obs_pos, drone_pos)
        for j in range(0, len(tgrid)):
            svals = sgrid[d_m[j, :] <= (drone_radius + obs.radius + safety_distance)]
            if svals.any():
                if not active:
                    active = True
                    collision_counter = collision_counter + 1
                table[j, 2 * collision_counter - 1], table[j, 2 * collision_counter] = np.amin(svals), np.amax(
                    svals)
            else:
                active = False
    return table, collision_counter


def fill_table_elipsoids(other_drones, drone_pos, tgrid, sgrid, drone_radius, drone_downwash, safety_distance, table,
                         collision_counter):
    for obs in other_drones:
        active = False
        obs_pos = obs.move(tgrid)
        d_m = calculate_eplis_rel_dist(obs_pos, drone_pos, drone_downwash, drone_radius, safety_distance)
        for j in range(0, len(tgrid)):
            svals = sgrid[d_m[j, :] <= 1]
            if svals.any():
                if not active:
                    active = True
                    collision_counter = collision_counter + 1
                table[j, 2 * collision_counter - 1], table[j, 2 * collision_counter] = np.amin(svals), np.amax(svals)
            else:
                active = False
    return table, collision_counter


def run_gurobi(drone, collision_counter, table, H, Ts, length, tgrid, speed, J_p, J_v):
    opt_mod = Model(name="linear program")
    opt_mod.setParam('OutputFlag', False)

    a = opt_mod.addVars(H, name='a', vtype=GRB.CONTINUOUS, lb=-drone.MAX_ACCELERATION, ub=drone.MAX_ACCELERATION)
    v = opt_mod.addVars(H + 1, name='v', vtype=GRB.CONTINUOUS, lb=0, ub=drone.MAX_SPEED)
    s = opt_mod.addVars(H + 1, name='s', vtype=GRB.CONTINUOUS)
    adir = opt_mod.addVars(collision_counter, name='adir', vtype=GRB.BINARY)
    cost_binary = opt_mod.addVars(H + 1, name='c_binary', vtype=GRB.BINARY)

    sBM = length + 1
    bigM = 1000

    opt_mod.addConstr(s[0] == 0.00001)  # not 0 to negate the invincible start point effect
    opt_mod.addConstr(s[len(s) - 1] == length - 0.00001)
    opt_mod.addConstr(v[0] == 0)
    opt_mod.addConstr(v[len(v) - 1] == 0)
    opt_mod.addConstrs(v[k + 1] == v[k] + a[k] * Ts
                       for k in range(H))
    opt_mod.addConstrs(s[k + 1] == s[k] + v[k] * Ts + 0.5 * (Ts ** 2) * a[k]
                       for k in range(H))
    opt_mod.addConstrs(s[k] - adir[obs] * sBM <= table[k, 2 * obs + 1]
                       for k in range(H)
                       for obs in range(collision_counter)
                       if table[k, 2 * obs + 2] > 0)
    opt_mod.addConstrs(s[k] + (1 - adir[obs]) * sBM >= table[k, 2 * obs + 2]
                       for k in range(H)
                       for obs in range(collision_counter)
                       if table[k, 2 * obs + 2] > 0)
    opt_mod.addConstrs(s[k] + 0.0001 - bigM * (1 - cost_binary[k]) <= length - 0.001
                       for k in range(H + 1))

    opt_mod.addConstrs(s[k] + bigM * cost_binary[k] >= length - 0.001
                       for k in range(H + 1))

    tgrid_H = np.append(tgrid, H)-drone.start_time
    s_opt = [min(speed * t, length) for t in tgrid_H]
    difference_in_position = [(s[x] - s_opt[x]) for x in range(len(s))]
    difference_in_position = np.array(difference_in_position)
    difference_in_position = difference_in_position.dot(np.transpose(difference_in_position))
    difference_in_speed = [(v[x] - speed) for x in range(len(v))]
    difference_in_speed = np.array(difference_in_speed)
    difference_in_speed = difference_in_speed.dot(np.transpose(difference_in_speed))
    weighted_difference = J_p * difference_in_position + J_v * difference_in_speed
    d_compensation = 0
    for x in range(len(v)):
        d_compensation += speed ** 2 * (1 - cost_binary[x]) * J_v

    opt_mod.setObjective(weighted_difference - d_compensation)

    opt_mod.ModelSense = GRB.MINIMIZE
    opt_mod.optimize()

    return opt_mod


def trim_trajectory(s_result, tgrid):
    standing = np.where(np.array(s_result) == s_result[-1])
    standing = np.flip(standing)[0]
    remove = False
    for i in range(len(standing)-1):
        if standing[i] - standing[i + 1] > 1:
            break
        remove = standing[i]
    if remove:
        s_result = s_result[:remove]
        tgrid = tgrid[:remove]
    return s_result, tgrid


def choose_target(scene, drone, return_home: bool = False):
    if drone.start_vertex not in scene.home_positions:
        scene.free_targets = np.append(scene.free_targets, drone.start_vertex)
    drone.start_vertex = drone.target_vetrex
    if return_home:
        drone.target_vetrex = np.random.choice(scene.home_positions)
        scene.home_positions = np.delete(scene.home_positions, scene.home_positions == drone.target_vetrex)
        # print(f"home positions: {scene.home_positions}")
    else:
        drone.target_vetrex = np.random.choice(scene.free_targets)
        scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.target_vetrex)
    # print(f"Start:target: {drone.start_vertex, drone.target_vetrex}")
