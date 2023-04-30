# +
import sys
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])
import os
from params_proto.neo_hyper import Sweep

def get_dir(ff):
    if ff is not None:
        return ff
    return "none"


# -

if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        #Args.fourier_features = "lFF"
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.critic_type = 'state_asym_metric'
        Args.critic_loss_type = 'td'

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                Args.n_workers = [2, 8, 16, 20]
                Args.n_epochs = [50, 150, 200, 500]
            Args.fourier_features = ["LFF", None]
            Args.fourier_dim_ratio = [1, 20, 40]
            Args.seed = [100, 200, 300, 400, 500]
            Args.metric_embed_dim = [16,]
    """
    for i, deps in sweep.items():
      os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/{deps['Args.env_name']}/{get_dir(deps['Args.fourier_features'])}/{deps['Args.seed']}"
      main(deps)
    """
    depslist = []
    for i,deps in sweep.items():
        depslist.append((i,deps))
    for i,deps in depslist[my_task_id - 1: my_task_id]:
        os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/{deps['Args.env_name']}/{get_dir(deps['Args.fourier_features'])}/{deps['Args.fourier_dim_ratio']}/{deps['Args.seed']}"
        main(deps)
