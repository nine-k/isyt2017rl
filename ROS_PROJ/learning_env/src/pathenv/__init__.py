import gym.envs.registration

gym.envs.registration.register('TurtleBotObstEnv-v1',
                                       entry_point='pathenv.environ:TurtleBotObstEnv')
