# Palko_Demo
Client script for Palkovits Máté's drone demo to be used with the AIMotionlab skybrush server. This readme explains the basics
of how the demo functions, in a high level abstraction. For the inner workings of the script, such as how the nurseries and queues
and such are handled, check the plentiful comments in the code itself.

Setting the parameters
----------------------------------------
When the demo is run, first order of business is setting the following variables correctly:
`LIVE_DEMO, drone_IDs, demo_time, REST_TIME, TAKEOFF_TIME, traj_type`
- The IDs in`drone_IDs` *must* be the same
as the actual drones found in the flight area, else the demo throws an error. This is
intentional, to ensure that we are using the correct drones. 
- `LIVE_DEMO` shall be set to
False if we don't intend to dispatch trajectories live, as they are being calculated, meaning
that we will use a skyc file instead. This skyc file will be generated regardless of whether
this variable is set or not. 
- `demo_time` is the time after which the drones will calculate a
return to home maneuver as their next trajectory instead of path to a random point. 
- `REST_TIME`
is the hover time for the drones between segments. This is the time during which the next segment 
is calculated, so it's advisable to leave it long enough for the drones to not
desync, which can happen if the calculations take too long. 
- `TAKEOFF_TIME` is *not* how long the takeoff
physically takes, instead, this is how much time we leave for the drone to take off before we dispatch the
first trajectory.

Before starting, the 3D representation of the drones' home positions, buildings and obstacles can be viewed 
to confirm their placement. Closing this window will start the demonstration.

The Demonstration's flow
--------------------------------------
- Initialization: for each drone, a TCP socket is made. On one side of the socket is
the client's handler, which is a DroneHandler object, which is responsible for generating the commands to be sent to the drone .
On the other side is the server's handler, which is responsible for processing the commands sent. The client DroneHandlers each
enter an infinite loop with `continuously_pop_que`, listening for the commands of the demo script. 
- Before starting the demonstration proper, we calculate the first trajectory for each drone. 
- Each drone is sent a takeoff command and once `TAKEOFF_TIME + TIME_REST` have passed, the first
trajectory is started.
- Whenever a trajectory is started, a timer for the trajectory's duration is started alongside it.
- Once the timer is up, a calculate command is appended to the queue of the handler, as if to say: 'okay, I'm done with my trajectory,
I'm ready for the next one. This way, we don't need to manually check which drone's trajectory is over first by checking a list
of their finish times: whenever the next drone is up for a trajectory, it tells its handler. Later on, this may be modified so
that the *server side handler* tells the client that the drone's trajectory is finished.
- The calculate command randomly selects a point in the flight area, and runs Palkovits Máté's script to find an optimal path
to it, without colliding with the other drones or structures. At the start of the calculation, the time at which the trajectory
we're calculating should start is selected, as a 'deadline': the current time + TIME_REST. This is why the proper selection of TIME_REST
is so important: if we want to start the trajectory 3 seconds from now, we need to know where the other drones will be 3 seconds from now.
The calculations are run with this in mind, however, if they take longer than 3 seconds, then the trajectory will be started later than
the deadline. This means that the other drones will be in slightly different positions than anticipated during the trajectory planning,
and may crash into one another. Luckily, the planning takes less than half a second currently.
- Once the calculations are over, an upload command is appended to the queue.
- Once the deadline for starting is up, a start command is appended to the drone's queue, closing the loop where each command
automatically appends the next command to the queue: start->calculate->upload->start. This goes on until we are past `demo_time`,
at which point the next point to be appended is not randomly selected, but it is a landing position instead. If the demo is signalled
to be over this way, then the start command will append a land command instead of a calculate command. The land command does not put
anything in the queue, thereby breaking the cycle. 
- Once each drone has landed, the handlers stop listening and the demo is exited.