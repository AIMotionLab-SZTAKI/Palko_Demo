from path_planning_and_obstacle_avoidance.Scene_construction import construction
from path_planning_and_obstacle_avoidance.Classes import Construction, Drone
import pickle
from path_planning_and_obstacle_avoidance.Trajectory_planning import *
import re
import sys
import trio
from trio import sleep, sleep_until
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


def write_to_skyc(type: str):
    Skyc_Data = []
    for drone_ID in drone_IDs:
        sub_folder_path = os.path.join(traj_folder_path, drone_ID)
        pickle_folder_path = os.path.join(sub_folder_path, "pickle")
        Bezier_Data = []
        for file_name in sorted(os.listdir(pickle_folder_path)):
            file_path = os.path.join(pickle_folder_path, file_name)
            with open(file_path, 'rb') as f:
                t, x, y, z = pickle.load(f)
                if len(Bezier_Data) < 1:  # if this is our first segment, we have to add a takeoff segment
                    Bezier_Data = [[0, [x[0], y[0], 0], []],  # start here
                                   [TAKEOFF_TIME, [x[0], y[0], z[0]],
                                    []]]  # take off to the start position in takeoff_time
                last_time = Bezier_Data[-1][0]
                last_point = Bezier_Data[-1][1]
                Bezier_Data.append([last_time + REST_TIME, last_point, []])  # add a hover before the new segment
                t = [x + last_time + REST_TIME for x in t]
                # plt.plot(t, y)
                knots = determine_knots(t, 20)[1:-1]  # here we can't have as many segments as if we sent the trajectories one by one
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
                    bernstein_x = interpolate.BPoly(
                        np.array([element[0] for element in Bezier_Curve[1:]]).reshape(degree + 1, 1), np.array([0, 1]))
                    bernstein_y = interpolate.BPoly(
                        np.array([element[1] for element in Bezier_Curve[1:]]).reshape(degree + 1, 1), np.array([0, 1]))
                    bernstein_z = interpolate.BPoly(
                        np.array([element[2] for element in Bezier_Curve[1:]]).reshape(degree + 1, 1), np.array([0, 1]))
                    # plt.plot(plot_time, bernstein_y(np.linspace(0, 1)))
                    # ######plotting######
                    curve_to_append = [Bezier_Curve[0],
                                       Bezier_Curve[-1],
                                       Bezier_Curve[2:-1]]
                    Bezier_Data.append(curve_to_append)
        land_segment = [Bezier_Data[-1][0] + TAKEOFF_TIME,
                        [Bezier_Data[-1][1][0], Bezier_Data[-1][1][1], 0],
                        []]
        Bezier_Data.append(land_segment)
        Skyc_Data.append(Bezier_Data)



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


def splines_to_json(spline_path: List[Union[np.ndarray, List[np.ndarray], int]],
                    speed_profile: Tuple[np.ndarray, np.ndarray, int], drone_ID: str, traj_num):
    # spline_path is a 3D BSpline with path length as its variable. 0: knots, 1: List of arrays x-y-z
    # speed_profile is a BSpline for distance at each time segment. 0: knots, 1: List of distances
    t_abs = list(np.linspace(speed_profile[0][0], speed_profile[0][-1], granularity))
    t = [x - t_abs[0] for x in t_abs]
    distances = interpolate.splev(t_abs, speed_profile)
    x_abs, y_abs, z_abs = interpolate.splev(distances, spline_path)
    if traj_type.lower() == 'abs' or traj_type.lower() == 'absolute':
        x = x_abs
        y = y_abs
        z = z_abs
    else:
        x = [element - x_abs[0] for element in x_abs]
        y = [element - y_abs[0] for element in y_abs]
        z = [element - z_abs[0] for element in z_abs]

    knots = determine_knots(t, num_of_segments)[1:-1]
    # We need to give the splrep inside knots. I think [0] and [-1] should also technically be inside knots, but apparently
    # not. I seem to remember that the first k-1 and last k-1 knots are the outside knots. Anyway, slprep seems to add k
    # knots both at the end and at the beginning, instead of k-1 knots which is what would make sense to me. How it decides
    # what those knots should be is a mystery to me, but upon checking them, they are the exact first and last knots that I
    # would've added, so it works out kind of.
    xSpline = interpolate.splrep(t, x, k=degree, task=-1, t=knots)
    ySpline = interpolate.splrep(t, y, k=degree, task=-1, t=knots)
    zSpline = interpolate.splrep(t, z, k=degree, task=-1, t=knots)

    # BPoly can be constructed from PPoly but not from BSpline. PPoly can be constructed from BSPline. BSpline can
    # be fitted to points. So Points->PPoly->BPoly. The coeffs of the BPoly representation are the control points.
    x_PPoly = interpolate.PPoly.from_spline(xSpline)
    y_PPoly = interpolate.PPoly.from_spline(ySpline)
    z_PPoly = interpolate.PPoly.from_spline(zSpline)
    x_BPoly = interpolate.BPoly.from_power_basis(x_PPoly)
    y_BPoly = interpolate.BPoly.from_power_basis(y_PPoly)
    z_BPoly = interpolate.BPoly.from_power_basis(z_PPoly)
    # These two lines below seem complicated but all they do is pack the data above into a convenient form: a list
    # of lists where each element looks like this: [t, (x,y,z), (x,y,z), (x,y,z)]. Note that this can almost
    # definitely be done in a simpler way :)
    BPoly = list(zip(list(x_BPoly.x)[degree + 1:-degree],
                     list(x_BPoly.c.transpose())[degree:-degree],
                     list(y_BPoly.c.transpose())[degree:-degree],
                     list(z_BPoly.c.transpose())[degree:-degree]))
    BPoly = [[element[0]] + list(zip(*list(element[1:]))) for element in BPoly]
    # adding a takeoff segment:
    Data = [[t[0], [x[0], y[0], z[0]], []]]
    for Bezier_Curve in BPoly:
        curve_to_append = [Bezier_Curve[0],
                           Bezier_Curve[-1],
                           Bezier_Curve[2:-1]]
        Data.append(curve_to_append)
    type = "COMPRESSED"  # for now, stick to compressed trajectories (although we could use POLY4D: TODO)
    json_dict = {
        "version": 1,
        "points": Data,
        "takeoffTime": Data[0][0],
        "landingTime": Data[-1][0],
        "type": type
    }
    json_object = json.dumps(json_dict, indent=2)
    sub_folder = os.path.join(traj_folder_path, drone_ID)
    json_folder = os.path.join(sub_folder, "json")
    json_path = os.path.join(json_folder, f"{traj_num}.json")
    with open(json_path, "w") as f:
        f.write(json_object)

    # add hover segments inbetween
    pickle_folder = os.path.join(sub_folder, 'pickle')
    pickle_path = os.path.join(pickle_folder, f"{traj_num}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump((t, x_abs, y_abs, z_abs), f)
    return json_object


def determine_id(string):
    '''takes a name as found in optitrack and returns the ID found in it, for example cf6 -> 06'''
    number_match = re.search(r'\d+', string)
    if number_match:
        number = number_match.group(0)
        if len(number) == 1:
            number = '0' + number
        return number
    return None


def determine_home_position(drone_ID: str):
    mocap = motioncapture.MotionCaptureOptitrack("192.168.1.141")
    mocap.waitForNextFrame()
    items = mocap.rigidBodies.items()
    # let's put the rigid bodies containing cf index into a dictionary with their IDs.
    drones = {determine_id(name): list(obj.position[:-1]) for name, obj in items if 'cf' in name}
    # and then select the drone which we're inspecting from the dictionary
    drone_pos = drones[drone_ID]
    # this line below looks scary but what it does is it selects the x-y coordinates of the starting nodes, and packs
    # them into a tuple with their associated index in graph['graph'].nodes.data('pos')
    starting_positions = [(index, graph['graph'].nodes.data('pos')[index][:-1]) for index in scene.home_positions]

    def calculate_distance(xy1: List[float], xy2: List[float]) -> float:
        '''calculates eucledian distance between xy points '''
        return np.linalg.norm(np.array(xy1) - np.array(xy2))

    min_distance = float('inf')
    # what we want is not the starting position, but the index of the starting position's node in this ugly looking
    # variable: graph['graph'].nodes.data('pos'). So we look at which xy pair is closest to the drone's position and we
    # return its index
    for pos_idx, xy in starting_positions:
        distance = calculate_distance(drone_pos, xy)
        if distance < min_distance:
            min_distance = distance
            home_pos_index = pos_idx
    assert min_distance < 0.2  # if the closest starting position is further than 0.2m, we messed up probably!
    return home_pos_index


def verify_drones(drone_IDs: List[str]):
    '''Function that asserts we have exactly the drones that we want in frame.'''
    mocap = motioncapture.MotionCaptureOptitrack("192.168.1.141")
    mocap.waitForNextFrame()
    items = mocap.rigidBodies.items()
    drones = [(determine_id(name), list(obj.position[:-1])) for name, obj in items if 'cf' in name]
    assert set(drone[0] for drone in drones) == set(drone_IDs)


class TcpCommand:
    command: str
    arg: Union[float, str, None]
    def __init__(self, command, arg):
        self.command = command
        self.arg = arg


class DroneHandler:
    socket: trio.SocketStream
    send_channel: trio.MemorySendChannel
    receive_channel: trio.MemoryReceiveChannel
    takeoff_height: float
    drone: Drone
    traj: Dict
    drone_ID: str
    live_demo: bool
    # trajectory: Tuple[List[Union[np.ndarray, List[np.ndarray], int]], Tuple[np.ndarray, np.ndarray, int]]

    def __init__(self, socket: trio.SocketStream, takeoff_height: float, drone_ID: str):
        self.live_demo = False
        self.drone_ID = drone_ID
        self.socket = socket  # the socket to talk to the server
        # In trio, memory channels serve as queues, where send_channel puts items in the queue and receive_channel
        # consumes them. Our queue is of length 100, which is more than enough. In this queue we will store the commands
        # sent to the drone. With the current implementation, we shouldn't need a queue, but it is possible to imagine
        # scenarios where we have commands waiting for the previous command(s) to be completed.
        self.send_channel, self.receive_channel = trio.open_memory_channel(100)
        self.takeoff_height = takeoff_height
        self.cmd_dict = {  # these are the commands recognized by the demo, the dictionary points to their functions
            "takeoff": self._takeoff,
            "land": self._land,
            "upload": self._upload,
            "start": self._start,
            "calculate": self._calculate
        }
        self.traj_id = 1  # this is here in case we want to limit how many trajectories should be in a demo
        self._start_deadline = None  # this is the timestamp by which trajectories must be started
        self.landing = False  # show that the drone is finished with the demo and is about to land
        self.on_ground = False
        self.limiter = RateLimiter(500)  # for safe communication, limit the rate of outgoing messages to 500Hz

        # make folders for trajectory files and log files:
        sub_folder_path = os.path.join(traj_folder_path, drone_ID)
        os.makedirs(sub_folder_path)
        json_folder_path = os.path.join(sub_folder_path, 'json')
        os.makedirs(json_folder_path)
        pickle_folder_path = os.path.join(sub_folder_path, 'pickle')
        os.makedirs(pickle_folder_path)
        log_file_path = os.path.join(log_folder_path, drone_ID)
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
        open(log_file_path, 'a').close()
        self.log = log_file_path

    async def send_and_ack(self, data: bytes):
        await self.limiter.limit()  # for safe communication, limit the rate of outgoing messages
        with open(self.log, 'a') as log:  # note the command to the log file
            log.write(f"{elapsed_time():.3f}: {data}\n")
        if self.live_demo:
            await self.socket.send_all(data)
            ack = b""
            while ack != b"ACK":  # wait for a response, if none comes, then this will block the other commands
                ack = await self.socket.receive_some()
                await sleep(0.01)
            with open(self.log, 'a') as log:
                log.write(f"{elapsed_time():.3f}: {ack}\n")

    async def _calculate(self, last: bool):
        print(f"[{elapsed_time():.3f}] {self.drone_ID}: Got calculate command. Trajectory should start in {self.drone.rest_time} sec")
        choose_target(scene, self.drone, last)  # last==True will mean that the target chosen will be the home position
        self.drone.start_time = elapsed_time() + self.drone.rest_time  # required for trajectory calculation
        self._start_deadline = current_time() + self.drone.rest_time  # deadline for starting the trajectory
        other_drones = [drone for drone in drones if drone.cf_id != self.drone_ID]
        # print(f"{self.drone_ID}: other drones are {[drone.cf_id for drone in other_drones]}")
        spline_path, speed_profile, duration, length = generate_trajectory(drone=self.drone, G=graph,
                                                                           dynamic_obstacles=[],
                                                                           other_drones=other_drones,
                                                                           Ts=scene.Ts,
                                                                           safety_distance=scene.general_safety_distance)

        self.drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
        # self.drone.flight_time = self.drone.trajectory['speed_profile'][0][-1]
        self.drone.flight_time = duration
        add_coll_matrix_to_elipsoids([self.drone], graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                                     scene.general_safety_distance)
        traj_json = splines_to_json(spline_path, speed_profile, self.drone_ID, self.traj_id)
        self.traj_id += 1
        if last:
            self.landing = True
        await self.enqueue_command("upload", traj_json)  # calculations are always followed by an upload

    async def _takeoff(self, arg):
        height = float(arg)
        print(f"[{elapsed_time():.3f} s] {self.drone_ID}: Taking off to {height}")
        await self.send_and_ack(f"CMDSTART_takeoff_{height:.4f}_EOF".encode())
        await sleep(TAKEOFF_TIME)  # while the takeoff is under way, block the handler

    async def _upload(self, arg):
        self.traj = arg
        await self.send_and_ack(f"CMDSTART_upload_{self.traj}_EOF".encode())
        await sleep_until(self._start_deadline)
        await self.enqueue_command("start", traj_type)  # upload commands are always followed by a start command

    async def _land(self, arg):
        assert arg is None
        print(f"[{elapsed_time():.3f} s] {self.drone_ID}:Landing")
        await self.send_and_ack(f"CMDSTART_land_EOF".encode())
        await sleep(TAKEOFF_TIME)  # landing should take about as long as taking off
        self.on_ground = True

    async def _start(self, arg):
        if str(arg).lower() in ["relative", "rel", "absolute", "abs"]:  # trajectory may be absolute or relative
            # self.drone.flight_time could be read from self.traj as well
            await self.send_and_ack(f"CMDSTART_start_{arg}_EOF".encode())
            print(f"[{elapsed_time():.3f} s] {self.drone_ID}: Beginning trajectory lasting {self.drone.flight_time}")
            await sleep(self.drone.flight_time)  # while the drone is traversing the trajectory, wait
            if self.landing:  # self.landing signals that we don't need any more trajectories
                await self.enqueue_command("land", None)
            else:  # if elapsed time is more than demo time, that means we should calculate the return to home
                await self.enqueue_command("calculate", elapsed_time() > demo_time)
        else:
            print(f"Invalid trajectory type (not absolute or relative)")

    async def enqueue_command(self, command: str, arg):
        # this is the safe way to send a command to the drone: put it in the command queue. A command may have a command
        # and an argument (such as the takeoff height, or the trajectory's type)
        command = TcpCommand(command=command, arg=arg)
        await self.send_channel.send(command)

    async def continuously_pop_que(self):
        # continuously scan the queue for items to be popped
        while not self.on_ground:
            command: TcpCommand = await self.receive_channel.receive()
            await self.cmd_dict[command.command](command.arg)  # do we want this to run in a nursery?
            # await sleep(0.001)


def elapsed_time():
    """returns the time since it was first called, in order to make time.time() more usable, since time.time() is
    a big number"""
    if not hasattr(elapsed_time, 'start_time'):
        elapsed_time.start_time = current_time()
    return current_time() - elapsed_time.start_time


class RateLimiter:
    '''Class to prevent a code part from running too often. Call its limit() function in the code part which you need
    to prevent from running more often than 'rate' Hz.'''
    def __init__(self, rate: float):
        self.rate = rate
        self.interval = 1 / rate
        self.last_call_timestamp = elapsed_time()

    async def limit(self):
        current_time = elapsed_time()
        time_since_last_call = current_time - self.last_call_timestamp
        if time_since_last_call < self.interval:
            # print(f"limiting rate")
            delay = self.interval - time_since_last_call
            await trio.sleep(delay)
        self.last_call_timestamp = elapsed_time()


async def establish_connection_with_handler(drone_id: str):
    drone_stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", PORT)
    await sleep(0.01)
    request = f"REQ_{drone_id}"
    print(f"Requesting handler for drone {drone_id}")
    await drone_stream.send_all(request.encode('utf-8'))
    acknowledgement: bytes = await drone_stream.receive_some()
    if acknowledgement.decode('utf-8') == f"ACK_{drone_id}":
        print(f"successfully created server-side handler for drone {drone_id}")
        return drone_stream
    else:
        return None

async def demo():
    print("Welcome to Palkovits Máté's drone demo!")
    await sleep(1)
    # we will have a handler process for each drone, which are completely independent. For ease of access, we will put
    # them into a dictionary to look up a handler by its associated cf ID, however, we will also save the ID in the
    # handler itself.
    handlers: Dict[str, DroneHandler] = {}
    for drone_ID in drone_IDs:
        # run the initialization, where we establish a TCP socket for each drone
        socket = await establish_connection_with_handler(drone_ID)
        # designate a TCP socket and an associated handler for each drone
        if socket is not None:
            handlers[drone_ID] = DroneHandler(socket=socket, takeoff_height=0.5, drone_ID=drone_ID)
            handlers[drone_ID].live_demo = LIVE_DEMO
        else:
            raise NotImplementedError  # immediately stop if we couldn't reach one of the drones
        await sleep(0.01)
    async with trio.open_nursery() as nursery:
        for ID, handler in handlers.items():
            # each handler will continuously scan its queue for commands to be handled
            nursery.start_soon(handler.continuously_pop_que)
        demo_start_deadline = current_time() + TAKEOFF_TIME + REST_TIME # this is when we start the first trajectories
        elapsed_time.start_time = current_time()  # reset elapsed_time to 0
        for i, drone_ID in enumerate(drone_IDs):
            drone = [d for d in drones if d.cf_id == drone_ID][0]
            traj_json = splines_to_json(drone.trajectory['spline_path'], drone.trajectory['speed_profile'], drone_ID, 0)
            # figure out what height the trajectories start at for each drone, for accurate takeoff
            takeoff_height = spline_path[1][2][0]
            handlers[drone_ID].drone = drone  # connect the handler with the Drone() object
            handlers[drone_ID]._start_deadline = demo_start_deadline  # set the drone to start after the hover period
            await handlers[drone_ID].enqueue_command("takeoff", takeoff_height)
            await handlers[drone_ID].enqueue_command("upload", traj_json)

# This list has to be set manually to the drones we want to fly. This is inconvenient, but that's the point: it forces
# us to manually double check if we are using the correct drones.
drone_IDs = ["04", "06", "07", "08"]
# drone_IDs = ["08", "04"]
verify_drones(drone_IDs)  # throw an error if the drones in frame are not exactly the drones we set above

# create the folders where we will store trajectory json files (for backup, logging and skyc generation)
# folders are "trajectories" inside which we will have a folder for each drone. within those will be separate folders
# for json files (mostly for debugging) and pickles with raw datapoints (for skyc file making). We will also have a log
# folder where we store the communication that occured with the server: timestamps and exact commands sent, as well as
# acknowledgement answers from the server
current_dir = os.getcwd()
traj_folder_name = "trajectories"
traj_folder_path = os.path.join(current_dir, traj_folder_name)
if os.path.exists(traj_folder_path):
    shutil.rmtree(traj_folder_path)
os.makedirs(traj_folder_path)
log_folder_name = "logs"
log_folder_path = os.path.join(current_dir, log_folder_name)
if os.path.exists(log_folder_path):
    shutil.rmtree(log_folder_path)
os.makedirs(log_folder_path)

np.random.seed(1001)
number_of_targets, graph = construction()
scene = Construction()
demo_time = 20  # sec

# SETUP DRONES
target_zero = len(graph['graph'].nodes()) - number_of_targets
target_list = np.arange(target_zero, len(graph['graph'].nodes()), 1)
scene.free_targets = target_list


NUM_OF_DRONES = len(drone_IDs)
# Set this to true if we want to dispatch the commands we calculate. A Skyc file will be generated regardless
LIVE_DEMO = False
REST_TIME = 3  # This is how much drones will hover between trajectories. Calculations are run during these rests.
# This is NOT how long it takes to perform takeoff. That is determined by the server and the firmware. This is how much
# we wait after dispatching the takeoff command, before continuing. Set this to a common sense value that will obviously
# be longer than how much time an actual takeoff takes, that way we don't try starting a trajectory while taking off.
TAKEOFF_TIME = 2
# this is how many bezier curves are generated for a trajectory. Directly influences how much memory it will take to
# store the trajectory. More means more accurate trajectory but more memory
num_of_segments = 40
# this is how many datapoints we evaluate to construct bezier curves from. The more the better, but the slower the
# calculations will run. Try to use a multiple of num_of_segments + 1 (this last tip may be outdated: TODO)
granularity = 1001
# currently only degree 3 may be used. This is because the drone can only handle degree 0-1-3-7, but the spline fitting
# only goes up to 5. I guess we could also use degree 1, but come on, we're professionals over here.
degree = 3
PORT = 6000  # The TCP port for server side communication
traj_type = "absolute"

scene.home_positions = scene.free_targets[-NUM_OF_DRONES:]  # the home positions are appended to the end of the nodes
scene.free_targets = scene.free_targets[:-NUM_OF_DRONES]  # let's not fly to the home positions, only the 'normal' nodes
drones: List[Drone] = []

for i, drone_ID in enumerate(drone_IDs):
    # let's calculate the first trajectories, since they should be handled differently from the rest
    drone = Drone()
    drone.rest_time = REST_TIME
    drone.cf_id = drone_ID
    # It could be possible, for example, that the home position associated witht he index home_positions[i] is actually
    # not the position where the drone in question is. The function below calculates which home position is the closest.
    drone.start_vertex = determine_home_position(drone_ID)
    drone.target_vetrex = np.random.choice(scene.free_targets)
    # bar the other drones from selecting the node we're going to. i.e. delete the target vertex from the free vertices
    scene.free_targets = np.delete(scene.free_targets, scene.free_targets == drone.target_vetrex)
    # the function below works correctly if the trajectory and flight_time of the other_drones is set correctly. It
    # returns the xyz(distance_travelled), distance_travelled(time), as well as the full duration and length
    spline_path, speed_profile, duration, length = generate_trajectory(drone=drone, G=graph, dynamic_obstacles=[],
                                                                       other_drones=drones, Ts=scene.Ts,
                                                                       safety_distance=scene.general_safety_distance)
    drone.trajectory = {'spline_path': spline_path, 'speed_profile': speed_profile}
    drone.flight_time = duration
    add_coll_matrix_to_elipsoids([drone], graph['point_cloud'], scene.Ts, scene.cmin, scene.cmax,
                                 scene.general_safety_distance)
    drones.append(drone)  # for the next drone, this current drone will count as part of 'other_drones': avoid collision

trio.run(demo)

# at this point the demo is over and the trajectories are ready to be processed
write_to_skyc(type="COMPRESSED")





