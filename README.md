# Multiple-Selves
An exploration of modular vs. monolithic reinforcement learning in the context of competing drives

World_testing.py contains a single file that runs experiments

An example environment:

```python
env = World(n_actions = 4, 
            size = 10, 
            bounds = 3, 
            modes = 1, 
            field_of_view = 1, 
            stat_types = [0,0,0,0],
            variance = 2, 
            radius = 3,
            offset = 0.5,
            circle = True,
            stationary = True, 
            rotation_speed = np.pi,
            ep_length = 50,
            resource_percentile = 90,
            reward_type = 'HRRL') # HRRL, sq_dev, lin_dev, well_being, min_sq
```
