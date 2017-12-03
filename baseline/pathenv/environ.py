import gym
import numpy
import gym.spaces
from scipy.spatial.distance import euclidean
import random
from .utils_compiled import build_distance_map, get_flat_state, check_finish_achievable

from .tasks import BY_PIXEL_ACTIONS, BY_PIXEL_ACTION_DIFFS, TaskSet

from sfml import sf
import sfml.graphics as sf_graph
import sfml.system as sf_sys

WINDOW_W = 400
WINDOW_H = 400

class PathFindingByPixelWithDistanceMapEnv(gym.Env):
    def __init__(self): #initialize
        self.VISUALIZE = True
        self.task_set = None
        self.cur_task = None
        self.observation_space = None
        self.obstacle_punishment = None
        self.local_goal_reward = None
        self.done_reward = None

        self.distance_map = None

        self.action_space = gym.spaces.Discrete(len(BY_PIXEL_ACTIONS))
        self.cur_position_discrete = None
        self.goal_error = None

        self.map_squares = None
        self.map_window = None

    def _configure(self,
                   tasks_dir='data/imported/paths', #task directory
                   maps_dir='data/imported/maps', #maps directory
                   obstacle_punishment=1, #deafult reward and punishment values
                   local_goal_reward=5,
                   done_reward=10,
                   greedy_distance_reward_weight=0.1,
                   absolute_distance_reward_weight=0.1,
                   vision_range=20, #default vision range
                   target_on_border_reward=5,
                   absolute_distance_observation_weight=0.1,
                   visualize=False):
        self.VISUALIZE = visualize
        self.task_set = TaskSet(tasks_dir, maps_dir)
        self.task_ids = list(self.task_set.keys())
        self.cur_task_i = 0
        self.vision_range = vision_range

        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,
                                                shape=(2 * self.vision_range + 1, 2 * self.vision_range + 1))

        self.obstacle_punishment = abs(obstacle_punishment)
        self.local_goal_reward = local_goal_reward
        self.done_reward = done_reward

        self.greedy_distance_reward_weight = greedy_distance_reward_weight
        self.absolute_distance_reward_weight = absolute_distance_reward_weight

        self.target_on_border_reward = target_on_border_reward
        self.absolute_distance_observation_weight = absolute_distance_observation_weight

        self.map_window = sf.RenderWindow(sf.VideoMode(WINDOW_W, WINDOW_H), "MAP")

    def __repr__(self):
        return self.__class__.__name__

    def _reset(self):
        self.cur_task = self.task_set[self.task_ids[self.cur_task_i]] #set next task
        self.cur_task_i += 1
        if self.cur_task_i >= len(self.task_ids):
            self.cur_task_i = 0

        rand = random.Random()
        if self.cur_task is not None:
            local_map = self.cur_task.local_map  #choose start and finish point
            while True:
                self.start = (rand.randint(0, self.cur_task.local_map.shape[0] - 1),
                              rand.randint(0, self.cur_task.local_map.shape[1] - 1))
                self.finish = (rand.randint(0, self.cur_task.local_map.shape[0] - 1),
                               rand.randint(0, self.cur_task.local_map.shape[1] - 1))
                if local_map[self.start] == 0 \
                        and local_map[self.finish] == 0 \
                        and self.start != self.finish \
                        and check_finish_achievable(numpy.array(local_map, dtype=numpy.float),
                                                    numpy.array(self.start, dtype=numpy.int),
                                                    numpy.array(self.finish, dtype=numpy.int)):
                    break
        if (self.VISUALIZE):
            print(self.cur_task.local_map)
            self.map_squares = [[0] * self.cur_task.local_map.shape[1] for _ in range(self.cur_task.local_map.shape[0])]
            self.map_window.clear(sf_graph.Color.WHITE)
            self.map_window.display()
            for row in range(self.cur_task.local_map.shape[0]):
                for col in range(self.cur_task.local_map.shape[1]):
                    self.map_squares[row][col] = sf_graph.RectangleShape((20, 20))
                    self.map_squares[row][col].outline_color = sf.Color.BLACK
                    self.map_squares[row][col].outline_thickness = 1
                    # self.map_squares[row][col].size = sf_sys.Vector2(20, 20)
                    if (self.cur_task.local_map[row][col]):
                        self.map_squares[row][col].fill_color = sf_graph.Color.RED
                    self.map_squares[row][col].position = (col * 20, row * 20)
                    self.map_window.draw(self.map_squares[row][col])
            self.map_squares[self.start[0]][self.start[1]].fill_color = sf_graph.Color.GREEN
            self.map_squares[self.finish[0]][self.finish[1]].fill_color = sf_graph.Color.BLUE
            self.map_window.draw(self.map_squares[self.start[0]][self.start[1]])
            self.map_window.draw(self.map_squares[self.finish[0]][self.finish[1]])
            self.map_window.display()
        return self._init_state()

    def _init_state(self):
        local_map = numpy.array(self.cur_task.local_map, dtype=numpy.float)
        self.distance_map = build_distance_map(local_map,
                                               numpy.array(self.finish, dtype=numpy.int))

        m = self.cur_task.local_map
        self.obstacle_points_for_vis = [(x, y)
                                        for y in xrange(m.shape[0])
                                        for x in xrange(m.shape[1])
                                        if m[y, x] > 0]
        self.cur_episode_state_id_seq = [tuple(self.start)]
        self.cur_position_discrete = self.start
        return self._get_state()

    def _get_base_state(self, cur_position_discrete):
        return get_flat_state(self.cur_task.local_map,
                              tuple(cur_position_discrete),
                              self.vision_range,
                              self.done_reward,
                              self.target_on_border_reward,
                              self.start,
                              self.finish,
                              self.absolute_distance_observation_weight)

    def _get_state(self):
        cur_pos = tuple(self.cur_position_discrete)
        if cur_pos != self.cur_episode_state_id_seq[-1]:
            self.cur_episode_state_id_seq.append(cur_pos)
        result = [self._get_base_state(pos)
                  for pos in self.cur_episode_state_id_seq[:-2:-1]]
        if len(result) < 1:
            empty = numpy.zeros_like(result[0])
            for _ in xrange(1 - len(result)):
                result.append(empty)
        return numpy.stack(result)

    def _step(self, action):
        new_position = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]
        # print(action)
        # print(new_position)

        done = numpy.allclose(new_position, self.finish)
        if done:
            reward = self.done_reward
        else:
            goes_out_of_field = any(new_position < 0) or any(new_position + 1 > self.cur_task.local_map.shape)
            invalid_step = goes_out_of_field or self.cur_task.local_map[tuple(new_position)] > 0
            if invalid_step:
                reward = -self.obstacle_punishment
            else:
                local_target = self.finish
                cur_target_dist = euclidean(new_position, local_target)
                if cur_target_dist < 1:
                    reward = self.local_goal_reward
                    done = True
                else:
                    reward = self._get_usual_reward(self.cur_position_discrete, new_position)
                self.cur_position_discrete = self.cur_position_discrete + BY_PIXEL_ACTION_DIFFS[action]

        if (self.VISUALIZE):
            self.map_squares = [[0] * self.cur_task.local_map.shape[1] for _ in range(self.cur_task.local_map.shape[0])]
            self.map_window.display()
            for row in range(self.cur_task.local_map.shape[0]):
                for col in range(self.cur_task.local_map.shape[1]):
                    self.map_squares[row][col] = sf_graph.RectangleShape((20, 20))
                    self.map_squares[row][col].outline_color = sf.Color.BLACK
                    self.map_squares[row][col].outline_thickness = 1
                    # self.map_squares[row][col].size = sf_sys.Vector2(20, 20)
                    if (self.cur_task.local_map[row][col]):
                        self.map_squares[row][col].fill_color = sf_graph.Color.RED
                    self.map_squares[row][col].position = (col * 20, row * 20)
                    self.map_window.draw(self.map_squares[row][col])
            self.map_squares[self.start[0]][self.start[1]].fill_color = sf_graph.Color.GREEN
            self.map_squares[self.finish[0]][self.finish[1]].fill_color = sf_graph.Color.BLUE
            self.map_squares[self.cur_position_discrete[0]][self.cur_position_discrete[1]].fill_color = sf_graph.Color.BLACK
            self.map_window.draw(self.map_squares[self.start[0]][self.start[1]])
            self.map_window.draw(self.map_squares[self.cur_position_discrete[0]][self.cur_position_discrete[1]])
            self.map_window.draw(self.map_squares[self.finish[0]][self.finish[1]])
            self.map_window.display()

        observation = self._get_state()
        return observation, reward, done, None

    def _get_usual_reward(self, old_position, new_position):
        old_height = self.distance_map[tuple(old_position)]
        new_height = self.distance_map[tuple(new_position)]
        true_gain = old_height - new_height

        local_target = self.finish
        old_dist = euclidean(old_position, local_target)
        new_dist = euclidean(new_position, local_target)
        greedy_gain = old_dist - new_dist

        start_height = self.distance_map[tuple(self.start)]
        abs_gain = numpy.exp(-new_height / start_height)

        total_gain = sum(
            ((1 - self.greedy_distance_reward_weight - self.absolute_distance_reward_weight) * true_gain,
             self.greedy_distance_reward_weight * greedy_gain,
             self.absolute_distance_reward_weight * abs_gain))
        return total_gain
