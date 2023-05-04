import numpy as np
from collections import deque
from team_code.render import render, render_self_car, find_peak_box
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json
import dill
import joblib

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

def downsample_waypoints(waypoints, precision=0.2):
    """
    waypoints: [float lits], 10 * 2, m
    """
    downsampled_waypoints = []
    downsampled_waypoints.append(np.array([0, 0]))
    last_waypoint = np.array([0.0, 0.0])
    for i in range(10):
        now_waypoint = waypoints[i]
        dis = np.linalg.norm(now_waypoint - last_waypoint)
        if dis > precision:
            interval = int(dis / precision)
            move_vector = (now_waypoint - last_waypoint) / (interval + 1)
            for j in range(interval):
                downsampled_waypoints.append(last_waypoint + move_vector * (j + 1))
        downsampled_waypoints.append(now_waypoint)
        last_waypoint = now_waypoint
    return downsampled_waypoints

def collision_detections(map1, map2, threshold=0.04):
    """
    map1: rendered surround vehicles
    map2: self-car
    """
    assert map1.shape == map2.shape
    overlap_map = (map1 > 0.01) & (map2 > 0.01)
    ratio = float(np.sum(overlap_map)) / np.sum(map2 > 0)
    ratio2 = float(np.sum(overlap_map)) / np.sum(map1 > 0)
    if ratio < threshold:
        return True
    else:
        return False

def get_max_safe_distance(meta_data, downsampled_waypoints, t, collision_buffer, threshold):
    surround_map = render(meta_data.reshape(20, 20, 7), t=t)[0][:100, 40:140]
    if np.sum(surround_map) < 1:
        return np.linalg.norm(downsampled_waypoints[-3])
    # need to render self-car map
    hero_bounding_box = np.array([2.45, 1.0]) + collision_buffer
    safe_distance = 0.0
    for i in range(len(downsampled_waypoints) - 2):
        aim = (downsampled_waypoints[i + 1] + downsampled_waypoints[i + 2]) / 2.0
        loc = downsampled_waypoints[i]
        ori = aim - loc
        self_car_map = render_self_car(loc=loc, ori=ori, box=hero_bounding_box)[
            :100, 40:140
        ]
        if collision_detections(surround_map, self_car_map, threshold) is False:
            break
        safe_distance = max(safe_distance, np.linalg.norm(loc))
    return safe_distance




"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.
---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---
**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.
**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.
Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, buffer_capacity=1000000, batch_size=64, num_states=23, num_actions=1):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        target_actor,
        gamma,
        target_critic,
        actor_model,
        critic_model,
        critic_optimizer,
        actor_optimizer,

    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self, target_actor, gamma, target_critic, actor_model, critic_model, critic_optimizer, actor_optimizer):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, target_actor, gamma, target_critic, actor_model, critic_model, critic_optimizer, actor_optimizer)

    

    





















class InterfuserController(object):
    def __init__(self, config):
        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP,
            K_I=config.speed_KI,
            K_D=config.speed_KD,
            n=config.speed_n,
        )
        self.collision_buffer = np.array(config.collision_buffer)
        self.config = config
        self.detect_threshold = config.detect_threshold
        self.stop_steps = 0
        self.forced_forward_steps = 0

        self.red_light_steps = 0
        self.block_red_light = 0

        self.in_stop_sign_effect = False
        self.block_stop_sign_distance = (
            0  # If this is 3 here, it means in 3m, stop sign will not take effect again
        )
        self.stop_sign_trigger_times = 0

        self.prev_timestamp = 0
        self.steering_previous = 0
        self._eps_lookahead = 10e-3 # Epsilon distance approximation threshold (m)
        self.prev_state = np.zeros(23)
        self.upper_bound = 1
        self.lower_bound = -1
        self.num_states = 23
        self.num_actions = 1

        """
        ## Training hyperparameters
        """

        self.std_dev = 0.2


        try:
            n = open('StoredOUActionNoise.pkl', 'rb')
            self.ou_noise = joblib.load(n)
            print("StoredOUActionNoise Loaded")
        except:
            self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1)) 


        # load the instance from the file 
            

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        try:
            # Load the weights
            self.actor_model.load_weights("Lateral_actor.h5")
            self.critic_model.load_weights("Lateral_critic.h5")

            self.target_actor.load_weights("Lateral_target_actor.h5")
            self.target_critic.load_weights("Lateral_target_critic.h5")

            print("Weights Loaded")
        except:

            # Making the weights equal initially
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())


        # Learning rate for actor-critic models
        self.critic_lr = 0.00001
        self.actor_lr = 0.00001

        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(self.actor_lr)


        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005


        try:
            b = open('StoredBuffer.pkl', 'rb')
            self.buffer = joblib.load(b)
            print("Buffer Loaded")
        except:
            self.buffer = Buffer(1000000, 64, self.num_states, self.num_actions)

        # To store reward history of each episode
        try:
            erl = open('StoredEpisodeRewardList.pkl', 'rb')
            self.ep_reward_list = joblib.load(erl)
            print("StoredEpisodeRewardList Loaded")
        except:
            self.ep_reward_list = []


        # To store average reward history of last few episodes
        try:
            arl = open('StoredAverageRewardList.pkl', 'rb')
            self.avg_reward_list = joblib.load(arl)
            print("StoredAverageRewardList Loaded")
        except:
            self.avg_reward_list = []

        
        self.episodic_reward = 0
        self.prev_action = 0
        self.ep = 0
        self.terminate = False
        self.last_meta = 0
        self.stopped = 0



    def get_distance(self, x1, y1, x2, y2):
            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


    def get_lookahead_point_index(self, x, y, waypoints, lookahead_dis):
        for i in range(len(waypoints)):
            dis = self.get_distance(x, y, waypoints[i][0], waypoints[i][1])
            if abs(dis - lookahead_dis) <= self._eps_lookahead:
                return i
        return len(waypoints)-1


    def get_predicted_vehicle_location(self, x, y, steering_angle, yaw, v):
        wheel_heading = yaw + steering_angle
        wheel_traveled_dis = v * (self.timestamp - self.prev_timestamp)
        return [x + wheel_traveled_dis * np.cos(wheel_heading), y + wheel_traveled_dis * np.sin(wheel_heading)]


    """
    Here we define the Actor and Critic networks. These are basic Dense models
    with `ReLU` activation.
    Note: We need the initialization for last layer of the Actor to be between
    `-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
    the initial stages, which would squash our gradients to zero,
    as we use the `tanh` activation.
    """


    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(1000, activation="relu")(inputs)
        out = layers.Dense(1000, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(1000, activation="relu")(state_input)
        state_out = layers.Dense(1000, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(1000, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(1000, activation="relu")(concat)
        out = layers.Dense(1000, activation="relu")(out)
        outputs = layers.Dense(1, activation ="linear")(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model


    """
    `policy()` returns an action sampled from our Actor network plus some noise for
    exploration.
    """


    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]


    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))



    

    def run_step(
        self, speed, waypoints, junction, traffic_light_state, stop_sign, meta_data, timestamp
    ):
        """
        speed: int, m/s
        waypoints: [float lits], 10 * 2, m
        junction: float, prob of the vehicle not at junction
        traffic_light_state: float, prob of the traffic light state is Red or Yellow
        stop_sign: float, prob of not at stop_sign
        meta_data: 20 * 20 * 7
        """

        self.timestamp = timestamp

    
        if self.terminate == True:
            return 0, 0, 0, self.last_meta


        yaw = np.pi/2
        point1 = [0,0]
        waypont_dists = np.array([math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) for point2 in waypoints])
        x = 0
        y = 0

        angle_deviations = np.array([np.arctan2((-w[1]),w[0])-yaw for w in waypoints])

        velocity = speed


        aim = (waypoints[0] + waypoints[1]) / 2.0
        aim[1] *= -1


        # Heading error
        y_delta=aim[1]-y
        x_delta=aim[0]-x
        heading_error=np.arctan2(y_delta,x_delta)-yaw

        # crosstrack error
        cross_track_error = aim[0]

        parameters = np.concatenate((waypont_dists, angle_deviations), axis=0)
        
        parameters = np.concatenate((parameters, np.array([velocity, heading_error, cross_track_error])), axis=0)
        
        scaler = MinMaxScaler()

        parameters = parameters.reshape(1, -1)
        
        state = scaler.fit_transform(parameters)[0]

        current_deviation_angle = heading_error
        current_lateral_distance = cross_track_error
        max_deviation_angle = 1
        max_lateral_distance = 5
        max_velocity_threshold = 5
        C = 10

        print("heading error: ", np.degrees(heading_error))
        print("cross track error: ", cross_track_error)

        
        if traffic_light_state > 0.2 or stop_sign < 0.8:
            self.stopped = 0


        reward = (((max_lateral_distance-abs(current_lateral_distance)) * (max_deviation_angle-abs(current_deviation_angle)))/max_deviation_angle*max_lateral_distance)*(velocity/max_velocity_threshold)*C
        print("reward: ", reward)
        #print(self.stop_steps)
        if (abs(current_deviation_angle) >= max_deviation_angle or abs(current_lateral_distance) >= max_lateral_distance or self.stopped >= 150) and self.timestamp > 2.5:
            reward = -10000
            self.terminate = True


        if timestamp >= 1 and traffic_light_state <= 0.4 and stop_sign >= 0.6:
            self.buffer.record((self.prev_state, self.prev_action, reward, state))
            self.episodic_reward += reward
            self.buffer.learn(self.target_actor, self.gamma, self.target_critic, self.actor_model, self.critic_model, self.critic_optimizer, self.actor_optimizer)
            self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
            self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
            print("updated")

        # Recieve state and reward from environment.
        #state, reward, done, info = env.step(action)

      

        # End this episode when `done` is True
        #if done:
        #    break

        self.prev_state = state

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(self.prev_state), 0)

        self.prev_action = self.policy(tf_prev_state, self.ou_noise)[0]

        steer = self.prev_action


        #aim = (waypoints[1] + waypoints[0]) / 2.0
        #aim[1] *= -1
        #heading_error = np.pi / 2 - np.arctan2(aim[1], aim[0])
        #if speed < 0.01:
        #    heading_error = 0
        
        #steer = -np.degrees(steer_output) / 90

        #print("2:",steer)
        #steer = self.turn_controller.step(angle)
        #steer = np.clip(steer, -1.0, 1.0)
        #print("3:",steer)
        



        if speed < 0.2:
            self.stop_steps += 1
            self.stopped += 1
        else:
            self.stop_steps = max(0, self.stop_steps - 10)

        if speed < 0.06 and self.in_stop_sign_effect:
            self.in_stop_sign_effect = False

        if junction < 0.3:
            self.stop_sign_trigger_times = 0

        if traffic_light_state > 0.7:
            self.red_light_steps += 1
        else:
            self.red_light_steps = 0
        if self.red_light_steps > 1000:
            self.block_red_light = 80
            self.red_light_steps = 0
        if self.block_red_light > 0:
            self.block_red_light -= 1
            traffic_light_state = 0.01

        if stop_sign < 0.6 and self.block_stop_sign_distance < 0.1:
            self.in_stop_sign_effect = True
            self.block_stop_sign_distance = 2.0
            self.stop_sign_trigger_times = 3

        self.block_stop_sign_distance = max(
            0, self.block_stop_sign_distance - 0.05 * speed
        )
        if self.block_stop_sign_distance < 0.1:
            if self.stop_sign_trigger_times > 0:
                self.block_stop_sign_distance = 2.0
                self.stop_sign_trigger_times -= 1
                self.in_stop_sign_effect = True


       


        '''
        ###################################################################### POP ############################################################################################
        
        steer_list=np.arange(-1.2,1.2,0.1)
        v = speed
        x = 0
        y = 0

        yaw = np.pi/2

        aim = (waypoints[1] + waypoints[0]) / 2.0
        aim[1] *= -1

        steering_list = self.steering_previous + steer_list # List of steering angles in the neighbourhood (left and right) of previous steering angle

        lookahead_point = aim

        min_dist = float("inf") # Initialize minimum distance value to infinity

        steering = self.steering_previous # Set steering angle to previous value if the following optimization problem yields no acceptable solution (sanity check)

        for i in range(len(steering_list)):


            wheel_heading = yaw + steering_list[i]
            wheel_traveled_dis = v * (timestamp - self.prev_timestamp)

            predicted_vehicle_location = [x + wheel_traveled_dis * np.cos(wheel_heading), y + wheel_traveled_dis * np.sin(wheel_heading)] # Get predicted vehicle location based on its current state and control input (i-th steering angle from the list)

            dist_to_lookahead_point = math.sqrt((predicted_vehicle_location[0]- lookahead_point[0])**2 + (predicted_vehicle_location[1] - lookahead_point[1])**2) # Compute the distance between predicted vehicle location and lookahead point

            if dist_to_lookahead_point < min_dist: # Optimization problem (Minimize distance between predicted vehicle location and lookahead point to ensure effective path-tracking)

                steering = steering_list[i] # Select the steering angle that minimizes distance between predicted vehicle location and lookahead point

                min_dist = dist_to_lookahead_point # Update the minimum distance value
                
        self.steering_previous = steering # Update previous steering angle value
 
        return steering



        
        #############################################################################################################################################################################
        '''













        '''
        
        ################################################################# Bang Bang Controller ###############################################################################
        # crosstrack error
        e_r = 0
        min_idx = 0
        # Get the minmum distance between the vehicle and target trajectory
        for idx in range(len(waypoints)):
            dis = np.linalg.norm(waypoints[idx])
            if idx == 0:
                e_r = dis
            if dis < e_r:
                e_r = dis
                min_idx = idx

        min_path_yaw = np.arctan(waypoints[min_idx][1]/waypoints[min_idx][0])
        cross_yaw_error = min_path_yaw - (np.pi/2)
        if cross_yaw_error > np.pi/2:
            cross_yaw_error -= np.pi
        if cross_yaw_error < - np.pi/2:
            cross_yaw_error += np.pi 

        if cross_yaw_error > 0:
            e_r = e_r
        else:
            e_r = -e_r
        crosstrack_error = np.arctan(e_r/(speed+1.0e-6))        

        if crosstrack_error > 0:
            steer_output = 1.22*0.1
        elif crosstrack_error < 0:
            steer_output = -1.22*0.1
        else:
            steer_output = 0




        '''


        ###############################################################################################################################################################################
        '''

        ############################################################## MPC #########################################################################################################
        # MPC control
        # Discrete steering angle from -1.2 to 1.2 with interval of 0.1.
        steer_list=np.arange(-1.2,1.2,0.1)
        j_min = 0
        steer_output = 0
        for idx in range(len(steer_list)):
            vehicle_heading_yaw = np.pi/2 + steer_list[idx]
            t_diff = timestamp - self.prev_timestamp
            pred_x = 0 + speed*t_diff*np.cos(vehicle_heading_yaw)
            pred_y = 0 + speed*t_diff*np.sin(vehicle_heading_yaw)
            delta_dis = math.sqrt((waypoints[-1][0] - pred_x)**2 + (waypoints[-1][1] - pred_y)**2)                 
            j = 0.1*delta_dis**2 + steer_list[idx]**2
            if idx == 0:
                j_min = j
            if j < j_min:
                j_min = j
                steer_output = steer_list[idx]
        # Obey the max steering angle bounds
        if steer_output > 1.22:
            steer_output = 1.22
        if steer_output < -1.22:
            steer_output = -1.22
            





        ###############################################################################################################################################################################
        '''
        ########################################################## STANLEY CONTROLLER #################################################################################################
        '''
        # crosstrack error
        e_r = 0
        min_idx = 0
        # Get the minmum distance between the vehicle and target trajectory
        for idx in range(len(waypoints)):
            dis = np.linalg.norm(waypoints[idx])
            if idx == 0:
                e_r = dis
            if dis < e_r:
                e_r = dis
                min_idx = idx

        min_path_yaw = np.arctan(waypoints[min_idx][1]/waypoints[min_idx][0])
        cross_yaw_error = min_path_yaw - (np.pi/2)
        if cross_yaw_error > np.pi/2:
            cross_yaw_error -= np.pi
        if cross_yaw_error < - np.pi/2:
            cross_yaw_error += np.pi 

        if cross_yaw_error > 0:
            e_r = e_r
        else:
            e_r = -e_r
        delta_error = np.arctan(0.1*e_r/(speed+1.0e-6))        
        steer_output = heading_error + delta_error
        print("steer: "+str(steer_output))
        if steer_output>1.22:
            steer_output=1.22
        if steer_output<-1.22:
            steer_output=-1.22
        '''
        #################################################################################################################################################################################



        ########################################################## PURE PERSUIT CONTROL #################################################################################################
        '''
        # Pure Persuit Control
        y_delta=aim[1]
        x_delta=aim[0]
        alpha=np.arctan(y_delta/x_delta)-(np.pi/2)


        if alpha > np.pi/2:
            alpha -= np.pi
        if alpha < - np.pi/2:
            alpha += np.pi 

        

        steer_output=np.arctan(2*np.sin(alpha)/(15*speed))
        # Obey the max steering angle bounds
        if steer_output>1.22:
            steer_output=1.22
        if steer_output<-1.22:
            steer_output=-1.22
        '''
        ####################################################################################################################################################################################


        #print("1:",steer_output)



        
        

        brake = False
        # get desired speed
        downsampled_waypoints = downsample_waypoints(waypoints)
        d_0 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=0,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_05 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=0.5,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_075 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=0.75,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_1 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=1,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_15 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=1.5,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )
        d_2 = get_max_safe_distance(
            meta_data,
            downsampled_waypoints,
            t=2,
            collision_buffer=self.collision_buffer,
            threshold=self.detect_threshold,
        )

        d_05 = min(d_0, d_05, d_075)
        d_1 = min(d_05, d_075, d_15, d_2)

        safe_dis = min(d_05, d_1)
        d_0 = max(0, d_0 - 2.0)
        d_05 = max(0, d_05 - 2.0)
        d_1 = max(0, d_1 - 2.0)

        if d_0 < max(3, speed):
            brake = True
            desired_speed = 0.0
        else:
            desired_speed = max(
                0,
                min(
                    4 * d_05 - speed - max(0, speed - 2.5),
                    self.config.max_speed,
                    2 * d_1 - 0.5 * speed - max(0, speed - 2.5),
                ),
            )
            if junction > 0.0 and traffic_light_state > 0.3:
                brake = True
                desired_speed = 0.0
        desired_speed = desired_speed if brake is False else 0.0

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)

        if speed > desired_speed * self.config.brake_ratio:
            brake = True

        '''
        meta_info_1 = "d0:%.1f, d05:%.1f, d1:%.1f, desired_speed:%.2f" % (
            d_0,
            d_05,
            d_1,
            desired_speed,
        )
        '''
        meta_info_1 = "speed: %.2f, target_speed: %.2f" % (
            speed,
            desired_speed,
        )
        meta_info_2 = "on_road_prob: %.2f, red_light_prob: %.2f, stop_sign_prob: %.2f" % (
            junction,
            traffic_light_state,
            1 - stop_sign,
        )
        meta_info_3 = "stop_steps:%d, block_stop_sign_distance:%.1f" % (
            self.stop_steps,
            self.block_stop_sign_distance,
        )

        if self.stop_steps > 1200:
            self.forced_forward_steps = 12
            self.stop_steps = 0
        if self.forced_forward_steps > 0:
            throttle = 0.8
            brake = False
            self.forced_forward_steps -= 1
        if self.in_stop_sign_effect:
            throttle = 0
            brake = True

        self.prev_timestamp = self.timestamp
        
        self.last_meta = (meta_info_1, meta_info_2, meta_info_3, safe_dis)

        return steer, throttle, brake, (meta_info_1, meta_info_2, meta_info_3, safe_dis)

    def save_DDPG(self):

 
        self.ep_reward_list.append(self.episodic_reward)
        self.ep += 1
        # Mean of last 40 episodes
        avg_reward = np.mean(self.ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(self.ep, avg_reward))
        self.avg_reward_list.append(avg_reward)
        print(self.ep_reward_list)
        print(len(self.buffer.state_buffer[0]))



        """
        If training proceeds correctly, the average episodic reward will increase with time.
        Feel free to try different learning rates, `tau` values, and architectures for the
        Actor and Critic networks.
        The Inverted Pendulum problem has low complexity, but DDPG work great on many other
        problems.
        Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
        more episodes to obtain good results.
        """


        # Save the weights
        self.actor_model.save_weights("Lateral_actor.h5")
        self.critic_model.save_weights("Lateral_critic.h5")

        self.target_actor.save_weights("Lateral_target_actor.h5")
        self.target_critic.save_weights("Lateral_target_critic.h5")

        # Stores Noise
        n = open('StoredOUActionNoise.pkl', 'wb')
        joblib.dump(self.ou_noise , n)

        # Stores Buffer
        b = open('StoredBuffer.pkl', 'wb')
        joblib.dump(self.buffer , b)

        # Stores Stored Episode Reward List
        erl = open('StoredEpisodeRewardList.pkl', 'wb')
        joblib.dump(self.ep_reward_list , erl)

        # Stores Average Reward List
        arl = open('StoredAverageRewardList.pkl', 'wb')
        joblib.dump(self.avg_reward_list , arl)

        """
        Before Training:
        ![before_img](https://i.imgur.com/ox6b9rC.gif)
        """

        """
        After 100 episodes:
        ![after_img](https://i.imgur.com/eEH8Cz6.gif)
        """
