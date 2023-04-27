import os
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from rl import Args, main

    with Sweep(Args) as sweep:
        Args.gamma = 0.99
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.critic_type = 'state_asym_metric'
        Args.critic_loss_type = 'td'

        Args.env_name = <ENV_NAME>
        Args.n_workers = <N_WORKERS>
        Args.n_epochs = <N_EPOCHS>
        Args.seed = <SEED>
        Args.metric_embed_dim = <METRIC_EMBED_DIM>

    for i, deps in sweep.items():
        os.environ["ML_LOGGER_ROOT"] = f"{os.getcwd()}/results/bvn/{deps['Args.env_name']}/{deps['Args.seed']}"
        main(deps)
