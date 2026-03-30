 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_planTF \
    scenario_builder=nuplan_mini \
    scenario_builder.db_files=/home/ryan/nuplan/data/cache/mini_set \
    cache.cache_path=$HOME/nuplan/exp/cache_mini \
    cache.cleanup_cache=true \
    scenario_filter=mini \
    worker.threads_per_node=40
