from LimitedQLearn import LimitedQLearn


algo = LimitedQLearn(grid_size=8, max_steps=8, epochs=500, learning_rate=0.6,
                     reward_scaling = (1,-102), noise = 0.9,noise_decay = 0.05)

algo.train()