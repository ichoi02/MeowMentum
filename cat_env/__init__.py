import os

from gymnasium.envs.registration import register

# Model files are resolved relative to this package, so the envs can be made from
# any working directory.
_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))

register(
    id="Cat-v0",
    entry_point="cat_env.cat_env:CatEnv",
    max_episode_steps=50,
    kwargs={"model_path": os.path.join(_MODEL_DIR, "cat.xml")},
)

register(
    id="CatNoTail-v0",
    entry_point="cat_env.cat_env:CatEnv",
    max_episode_steps=50,
    kwargs={"model_path": os.path.join(_MODEL_DIR, "cat_notail.xml")},
)
