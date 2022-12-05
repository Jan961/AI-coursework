from LimitedQLearn import LimitedQLearn


algo = LimitedQLearn(grid_size=22, max_steps=30, epochs=4)

algo.train()