from gym.envs.registration import register

register(
    id='Organism-v0',
    entry_point='gym_organism.envs:OrganismEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=10250,

)