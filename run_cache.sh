export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_training.py \
  py_func=cache +training=train_planTF_ssl_pretrain \
  scenario_builder=nuplan_mini \
  scenario_builder.db_files=/home/ryan/nuplan/data/cache/mini_set \
  cache.cache_path=$HOME/nuplan/exp/cache_mini_ssl \
  cache.cleanup_cache=true \
  cache.force_feature_computation=true \
  scenario_filter=mini \
  worker=single_machine_thread_pool worker.max_workers=8
