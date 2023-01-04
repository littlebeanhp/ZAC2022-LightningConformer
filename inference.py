import os
import json

from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torchaudio

from dataset import load_vocab, tokenize_sentence, extract_feature
from model import AlignmentModel


def force_alignment(audio_filepath, transcript, vocabulary, model):

    waveform, sample_rate = torchaudio.load(audio_filepath)
    audio_length = waveform.size(1) / sample_rate

    feature = extract_feature(waveform, sample_rate)
    tokens = tokenize_sentence(transcript, vocabulary)

    new_transcript = "".join([vocabulary[t] for t in tokens])
    word_segments = model.get_word_alignment(feature, tokens, new_transcript)
    
    alignment = []
    spec_len = len(feature) // 2 + len(feature) % 2

    for seg in word_segments:
        d = seg.label
        s = seg.start / spec_len * audio_length
        e = seg.end / spec_len * audio_length
        alignment.append({"s": s, "e": e, "d": d})

    for i in range(1, len(alignment)):
        curr_align = alignment[i]
        prev_align = alignment[i - 1]

        gap = curr_align["s"] - prev_align["e"]
        if gap < 0.5:
            curr_align["s"] = prev_align["e"]

    return alignment


def write_label_audacity(alignment, filename):
    with open(f"dataset/public_test/labels/{filename}", "w") as f:
        for align in alignment:
            s = align["s"]
            e = align["e"]
            d = align["d"]
            f.write(f"{s}\t{e}\t{d}\n")


def write_label_zalo(alignment, filename):
    for align in alignment:
        align["s"] = int(align["s"] * 1000)
        align["e"] = int(align["e"] * 1000)

    with open(f"dataset/public_test/samples/{filename}") as f:
        samples = json.load(f)

    offset = 0
    for sentence in samples:
        sentence["s"] = alignment[offset]["s"]
        for segment in sentence["l"]:
            length = len(segment["d"].split())
            segment["s"] = alignment[offset]["s"]
            segment["e"] = alignment[offset + length - 1]["e"]
            segment["d"] = " ".join(
                [item["d"] for item in alignment[offset : offset + length]]
            )
            offset += length
        sentence["e"] = segment["e"]

    length_sample = len([word["d"] for sent in samples for word in sent["l"]])
    if length_sample != len(alignment):
        print(f"{filename} has a length mismatch")

    with open(f"dataset/public_test/submission/{filename}", "w") as f:
        f.write(json.dumps(samples, ensure_ascii=False))


if __name__ == "__main__":
    config = OmegaConf.load("config/finetune.yaml")
    model = instantiate(config.task.model.alignment_model)

    model.load_state_dict(torch.load("checkpoints/finetune.pt"))
    model.freeze()

    vocabulary = load_vocab("characters.txt")

    with open("dataset/public_test/testset.json") as f:
        dataset = f.read().split("\n")
        dataset = [json.loads(data) for data in dataset[:-1]]

    for data in tqdm(dataset):
        audio_filepath = data["audio_filepath"]
        transcript = data["transcript"]
        words = transcript.split()

        alignment = force_alignment(
            audio_filepath, transcript, vocabulary, model
        )

        assert len(words) == len(alignment), "Error in alignment!!!"
        for i, align in enumerate(alignment):
            align["d"] = words[i]

        label_filename = os.path.basename(audio_filepath)
        label_filename = label_filename.replace(".wav", "")

        write_label_audacity(alignment, label_filename + ".txt")
        write_label_zalo(alignment, label_filename + ".json")
