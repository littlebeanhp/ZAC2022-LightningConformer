import re
import json
import random

import librosa
import parselmouth
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset


def load_dataset(filepath):
    with open(filepath) as f:
        dataset = f.read().split("\n")
        dataset = [json.loads(data) for data in dataset[:-1]]
    return dataset


def load_vocab(filepath):
    with open(filepath) as f:
        vocab = f.read().split("\n")
        vocab = [""] + vocab[:-1]
    return vocab


def extract_note(filepath):
    snd = parselmouth.Sound(filepath)

    pitch_floor = librosa.note_to_hz("D2")
    pitch_ceiling = librosa.note_to_hz("C6")
    
    pitch = snd.to_pitch(time_step=0.02, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    pitch = pitch.selected_array['frequency'].clip(min=1e-9)

    note = np.array(librosa.hz_to_note(pitch))
    note[pitch == 1e-9] = "S"

    return note.tolist()


def extract_feature(waveform, sample_rate):
    if waveform.size(0) != 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, sample_frequency=16000, num_mel_bins=80
    )
    fbank = (fbank - fbank.mean()) / (fbank.std() + 1e-9)

    return fbank


def tokenize_sentence(text, vocab):
    text = text.strip().lower()
    text = re.sub("\s+", "_", text)

    pattern = "".join(vocab)
    text = re.sub(f"[^{pattern}]", "", text)

    characters = list(text)
    tokens = [vocab.index(char) for char in characters]

    return tokens


class SpecAugment:
    def __init__(
        self,
        freq_masks=1,
        time_masks=10,
        freq_width=27,
        time_width=0.05,
    ):
        self._rng = random.Random()

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

        self.mask_value = 0.0

        if isinstance(time_width, int):
            self.adaptive_temporal_width = False
        else:
            if time_width > 1.0 or time_width < 0.0:
                raise ValueError(
                    "If `time_width` is a float value, must be in range [0, 1]"
                )

            self.adaptive_temporal_width = True

    def apply(self, input_spec):
        t, d = input_spec.shape

        for i in range(self.freq_masks):
            x_left = self._rng.randint(0, d - self.freq_width)

            w = self._rng.randint(0, self.freq_width)

            input_spec[:, x_left : x_left + w] = self.mask_value

        for i in range(self.time_masks):
            if self.adaptive_temporal_width:
                time_width = max(1, int(t * self.time_width))
            else:
                time_width = self.time_width

            y_left = self._rng.randint(0, max(1, t - time_width))

            w = self._rng.randint(0, time_width)

            input_spec[y_left : y_left + w, :] = self.mask_value

        return input_spec


class AlignmentDataset(Dataset):
    def __init__(self, dataset_filepath, vocab_filepath, augment) -> None:
        super().__init__()
        self.dataset = load_dataset(dataset_filepath)
        self.vocab = load_vocab(vocab_filepath)

        self.labels = [
            "S", "A2", "A3", "A4", "A5",
            "A♯2", "A♯3", "A♯4", "A♯5",
            "B2", "B3", "B4", "B5",
            "C3", "C4", "C5", "C6",
            "C♯3", "C♯4", "C♯5",
            "D2", "D3", "D4", "D5",
            "D♯2", "D♯3", "D♯4", "D♯5",
            "E2", "E3", "E4", "E5",
            "F2", "F3", "F4", "F5",
            "F♯2", "F♯3", "F♯4", "F♯5",
            "G2", "G3", "G4", "G5",
            "G♯2", "G♯3", "G♯4", "G♯5",
        ]

        if augment:
            self.augment = SpecAugment(1, 10, 27, 0.05)

    def __getitem__(self, index):
        data = self.dataset[index]
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript"]
        note = data["note"]

        waveform, sample_rate = torchaudio.load(audio_filepath)
        feature = extract_feature(waveform, sample_rate)
        if hasattr(self, "augment"):
            feature = self.augment.apply(feature)

        token = tokenize_sentence(transcript, self.vocab)
        token = torch.tensor(token, dtype=torch.long)

        note_len = len(note)
        spec_len = len(feature) // 2 + len(feature) % 2
        note = torch.tensor([self.labels.index(n) for n in note], dtype=torch.long)
        note = torch.nn.functional.pad(note, (0, spec_len - note_len))

        return feature, token, note

    def __len__(self):
        return len(self.dataset)
