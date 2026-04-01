CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_planTF_ssl_pretrain \
  worker=single_machine_thread_pool worker.max_workers=8 \
  scenario_builder=nuplan scenario_filter=mini \
  cache.cache_path=/home/ryan/nuplan/exp/cache_mini_ssl cache.use_cache_without_dataset=true cache.force_feature_computation=false \
  data_loader.params.batch_size=4 data_loader.params.num_workers=8 \
  lr=1e-3 epochs=2 warmup_epochs=1 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  custom_trainer.print_ssl_losses=true \
  custom_trainer.ssl_loss_print_interval=5 \
  custom_trainer.plot_debug_batches=true \
  custom_trainer.debug_plot_dir=$HOME/Documents/planTF/debug/ssl_train_plots \
  custom_trainer.debug_plot_max_batches=50 \
  wandb.mode=disabled wandb.project=nuplan wandb.name=plantf_debug
