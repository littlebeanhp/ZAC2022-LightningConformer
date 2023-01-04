import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate

import torch

pl.seed_everything(42, workers=True)


@hydra.main(version_base=None, config_path="configs")
def main(config):
    task = instantiate(config.task, _recursive_=False)

    if config.pretrained_model is not None:
        task.alignment_model.load_state_dict(
            torch.load(config.pretrained_model, map_location="cpu")
        )
        print(f"Loading checkpoint from {config.pretrained_model}...")

    callbacks = None
    if config.callbacks is not None:
        callbacks = [instantiate(cfg) for __, cfg in config.callbacks.items()]

    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(task)


if __name__ == "__main__":
    main()
