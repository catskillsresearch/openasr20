import pytorch_lightning as pl

class ModelCheckpointAtEpochEnd(pl.callbacks.ModelCheckpoint):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        trainer.checkpoint_callback.on_validation_end(trainer, pl_module)
