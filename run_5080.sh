CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_planTF_ssl_pretrain \
  worker=single_machine_thread_pool worker.max_workers=8 \
  scenario_builder=nuplan scenario_filter=mini \
  scenario_builder.db_files=$HOME/nuplan/data/nuplan-v1.1/splits/mini \
  cache.cache_path=/home/ryan/nuplan/exp/cache_mini_ssl cache.use_cache_without_dataset=false cache.force_feature_computation=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=8 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=1.0 \
  custom_trainer.print_ssl_losses=False \
  custom_trainer.ssl_loss_print_interval=5 \
  custom_trainer.plot_debug_batches=False \
  custom_trainer.debug_plot_dir=$HOME/Documents/planTF/debug/ssl_train_plots \
  custom_trainer.debug_plot_max_batches=50 \
  wandb.mode=disable wandb.project=nuplan wandb.name=plantf
