from gymnasium.envs.registration import register

register(
    id="Cat-v0",
    entry_point="env.cat_env:CatEnv",
    max_episode_steps=300,
)