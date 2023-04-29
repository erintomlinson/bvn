mpirun -np 8 python3 ddpg_her/main.py \
  --Args.gamma=0.99 \
  --Args.agent_type='ddpg' \
  --Args.n_workers=20 \
  --Args.n_epochs=200 \
  --Args.n_cycles=50 \
  --Args.critic_type='fullrank-dot' \
  --MetricArgs.metric_embed_dim=16 \
  --Args.env_name=<ENV_NAME> \
  --Args.seed=<SEED>
