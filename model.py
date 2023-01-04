import warnings
from typing import Optional
from dataclasses import dataclass

from hydra.utils import instantiate
from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import _LRScheduler
from torchaudio.models import Conformer


def collate_data(batch):
    xs = [b[0] for b in batch]
    x_lens = [len(x) for x in xs]
    xs = pad_sequence(xs, batch_first=True)
    x_lens = torch.tensor(x_lens, dtype=torch.long)

    ys = [b[1] for b in batch]
    y_lens = [len(y) for y in ys]
    ys = pad_sequence(ys, batch_first=True)
    y_lens = torch.tensor(y_lens, dtype=torch.long)

    notes = [b[2] for b in batch]
    notes = pad_sequence(notes, batch_first=True, padding_value=-100)

    return xs, x_lens, ys, y_lens, notes


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class AcousticEncoder(nn.Module):
    def __init__(self, num_mels, d_model, num_layers) -> None:
        super().__init__()
        self.stride = 2
        self.subsampling = nn.Sequential(
            nn.Conv2d(1, d_model, 5, self.stride, 2),
            nn.ReLU(),
        )
        self.linear = nn.Linear(num_mels * d_model // self.stride, d_model)
        self.projection = Conformer(
            input_dim=d_model,
            num_layers=num_layers,
            num_heads=8,
            ffn_dim=d_model * 4,
            depthwise_conv_kernel_size=31,
            dropout=0.1,
        )

    def forward(self, features, feature_lengths):
        features = features.unsqueeze(1)
        features = self.subsampling(features)

        b, c, t, f = features.size()
        features = self.linear(features.transpose(1, 2).reshape(b, t, c * f))

        feature_lengths = torch.div(
            feature_lengths - 1, self.stride, rounding_mode="trunc"
        )
        feature_lengths = (feature_lengths + 1).type(torch.long)

        features, feature_lengths = self.projection(features, feature_lengths)

        return features, feature_lengths


class AlignmentModel(nn.Module):
    def __init__(self, num_mels, note_size, vocab_size, d_model, num_layers):
        super().__init__()

        self.note_size = note_size
        self.vocab_size = vocab_size

        self.backbone = AcousticEncoder(num_mels, d_model, num_layers)
        self.header = nn.Linear(d_model, (vocab_size * note_size))

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, xs, x_lens):
        enc_outs, enc_lens = self.backbone(xs, x_lens)
        enc_outs = self.header(enc_outs)

        b, t, d = enc_outs.size()
        enc_outs = enc_outs.reshape(b, t, self.vocab_size, self.note_size)

        return enc_outs, enc_lens

    def get_word_alignment(self, feature, tokens, transcript):
        xs = feature.unsqueeze(0)
        x_lens = torch.tensor([xs.size(1)], device=feature.device)

        outputs, output_lengths = self(xs, x_lens)
        outputs = torch.sum(outputs, dim=3)

        emission = outputs.log_softmax(dim=2)
        emission = emission[0].cpu().detach()

        trellis = self._get_trellis(emission, tokens, blank_id=0)
        path = self._backtrack(trellis, emission, tokens, blank_id=0)

        segments = self._merge_repeats(path, transcript)
        word_segments = self._merge_words(segments)

        return word_segments

    def _get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emission[t, blank_id],
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis


    def _backtrack(self, trellis, emission, tokens, blank_id=0):
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            path.append(Point(j - 1, t - 1, prob))

            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]


    def _merge_repeats(self, path, transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments


    def _merge_words(self, segments, separator="_"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(
                        seg.length for seg in segs
                    )
                    words.append(
                        Segment(word, segments[i1].start, segments[i2 - 1].end, score)
                    )
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words


class AlignmentTask(LightningModule):
    def __init__(self, dataset, model):
        super().__init__()
        self.save_hyperparameters()
        self.alignment_model = instantiate(model.alignment_model)
        self.criterion_lyrics = nn.CTCLoss(blank=0, zero_infinity=True)
        self.criterion_melody = nn.CrossEntropyLoss(ignore_index=-100)

    def train_dataloader(self):
        dataset = instantiate(self.hparams.dataset.train_ds)
        data_dl = DataLoader(
            dataset,
            **self.hparams.dataset.dataloader,
            shuffle=True,
            collate_fn=collate_data,
        )
        return data_dl

    def val_dataloader(self):
        dataset = instantiate(self.hparams.dataset.val_ds)
        data_dl = DataLoader(
            dataset,
            **self.hparams.dataset.dataloader,
            shuffle=False,
            collate_fn=collate_data,
        )
        return data_dl

    def _metrics(self, xs, x_lens, ys, y_lens, melody_gt):
        n_batch, n_frame, n_ch, n_p = xs.shape  # (batch, time, phone, pitch)

        y_lyrics = torch.sum(xs, dim=3)  # (batch, time, n_ch)
        y_lyrics = y_lyrics.log_softmax(dim=2).transpose(0, 1)  # (time, batch, n_ch)

        loss_lyrics = self.criterion_lyrics(y_lyrics, ys, x_lens, y_lens)

        y_melody = torch.sum(xs, dim=2)  # (batch, time, n_p)
        y_melody = y_melody.transpose(1, 2)  # (batch, n_p, time)

        loss_melody = self.criterion_melody(y_melody, melody_gt)

        return loss_lyrics, loss_melody

    def training_step(self, batch, batch_idx):
        xs, x_lens, ys, y_lens, notes = batch
        enc_outs, enc_lens = self.alignment_model(xs, x_lens)

        loss_lyrics, loss_melody = self._metrics(enc_outs, enc_lens, ys, y_lens, notes)
        loss = loss_lyrics + 0.5 * loss_melody

        self.log("train_lyrics_loss", loss_lyrics, sync_dist=True)
        self.log("train_melody_loss", loss_melody, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        xs, x_lens, ys, y_lens, notes = batch
        enc_outs, enc_lens = self.alignment_model(xs, x_lens)

        loss_lyrics, loss_melody = self._metrics(enc_outs, enc_lens, ys, y_lens, notes)
        loss = loss_lyrics + 0.5 * loss_melody

        self.log("val_lyrics_loss", loss_lyrics, sync_dist=True)
        self.log("val_melody_loss", loss_melody, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.model.optimizer)
        scheduler = NoamAnnealing(optimizer, **self.hparams.model.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class NoamAnnealing(_LRScheduler):
    def __init__(
        self, optimizer, *, d_model, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1
    ):
        self._normalize = d_model ** (-0.5)
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = max(1, self.last_epoch)

        for initial_lr in self.base_lrs:
            if initial_lr < self.min_lr:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )

        new_lrs = [self._noam_annealing(initial_lr=initial_lr, step=step) for initial_lr in self.base_lrs]
        return new_lrs

    def _noam_annealing(self, initial_lr, step):
        if self.warmup_steps > 0:
            mult = self._normalize * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        else:
            mult = self._normalize * step ** (-0.5)

        out_lr = initial_lr * mult
        if step > self.warmup_steps:
            out_lr = max(out_lr, self.min_lr)
        return out_lr
