from LimitedQLearn import LimitedQLearn


algo = LimitedQLearn(grid_size=16, max_steps=16, epochs=2000, learning_rate=0.1,
                     reward_scaling = (2,20), noise = 0.7,noise_decay = 0.05,
                     gamma=0.2, allow_do_nothing=True)

algo.train()