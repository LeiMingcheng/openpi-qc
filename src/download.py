from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

model_name = "pi0_aloha"
model_link = "gs://openpi-assets/checkpoints/pi0_base"

config = config.get_config(model_name)
checkpoint_dir = download.maybe_download(model_link)

policy = policy_config.create_trained_policy(config, checkpoint_dir)

