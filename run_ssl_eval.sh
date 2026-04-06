CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=test +training=train_planTF_ssl_pretrain \
  worker=single_machine_thread_pool worker.max_workers=2 \
  scenario_builder=nuplan scenario_filter=mini \
  scenario_builder.db_files=$HOME/nuplan/data/nuplan-v1.1/splits/mini \
  cache.cache_path=/home/ryan/nuplan/exp/cache_mini_ssl cache.use_cache_without_dataset=true cache.force_feature_computation=false \
  data_loader.params.batch_size=4 data_loader.params.num_workers=2 \
  checkpoint=${1:-$HOME/Documents/planTF/pretrained_model.ckpt} \
  custom_trainer.plot_test_reconstructions=true \
  custom_trainer.test_plot_dir=$HOME/Documents/planTF/debug/ssl_test_plots \
  custom_trainer.test_summary_path=$HOME/Documents/planTF/debug/ssl_test_metrics.json \
  wandb.mode=disable wandb.project=nuplan wandb.name=plantf_ssl_eval
