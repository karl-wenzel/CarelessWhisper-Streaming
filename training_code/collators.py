import torch
import numpy as np
import torch.nn.functional as F

WHISPER_TEXT_CTX = 448
WHISPER_EOT_TOKEN_ID = 50257


def _truncate_decoder_fields(feature: dict, max_text_ctx: int = WHISPER_TEXT_CTX, eot_token_id: int = WHISPER_EOT_TOKEN_ID) -> dict:
    dec_input_ids = torch.as_tensor(feature["dec_input_ids"], dtype=torch.long)
    labels = torch.as_tensor(feature["labels"], dtype=torch.long)

    # Requirement: keep decoder training tensors <= Whisper context to prevent positional embedding size crashes.
    min_len = min(dec_input_ids.shape[0], labels.shape[0])
    dec_input_ids = dec_input_ids[:min_len]
    labels = labels[:min_len]

    if min_len > max_text_ctx:
        dec_input_ids = dec_input_ids[:max_text_ctx].clone()
        labels = labels[:max_text_ctx].clone()
        dec_input_ids[-1] = eot_token_id
        labels[-1] = eot_token_id

    out = dict(feature)
    out["dec_input_ids"] = dec_input_ids
    out["labels"] = labels

    if "endpoints" in feature:
        endpoints = torch.as_tensor(feature["endpoints"], dtype=torch.float32)[:min_len]
        if min_len > max_text_ctx:
            endpoints = endpoints[:max_text_ctx].clone()
            if endpoints.shape[0] > 1:
                endpoints[-1] = endpoints[-2] + 0.5
        out["endpoints"] = endpoints

    return out


class WhisperDataCollatorWithPadding:
    def __call__(self, features):
        features = [_truncate_decoder_fields(feature) for feature in features]

        input_ids, labels, dec_input_ids, labels_classes, unique_ids = [], [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"].cpu().numpy())
            dec_input_ids.append(f["dec_input_ids"].cpu().numpy())
            labels_classes.append([int(item == WHISPER_EOT_TOKEN_ID) for item in f["labels"].tolist()])
            unique_ids.append(f.get("u_id", 0))

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        # labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=50257) for lab, lab_len in zip(labels, label_lengths)]
        labels_classes = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels_classes, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "labels_classes": labels_classes,
            "unique_id": unique_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        batch["input_ids"] = input_ids

        return batch


def pad_2d_sequences(arrays: list, dim: int = 0, padding_value: int = 0) -> torch.Tensor:
    lens = [array.shape[dim] for array in arrays]
    max_len = max(lens)

    padded_arrays = [F.pad(array, (0, int(dim == 1) * (max_len - array.shape[1]), 0, int(dim == 0) * (max_len - array.shape[0])), mode="constant", value=padding_value) for array in arrays]
    return torch.cat([padded_array[None] for padded_array in padded_arrays]) # adding batch dim and concatanating


class LoRAWhisperDataCollatorWithPadding:
    def __call__(self, features):
        features = [_truncate_decoder_fields(feature) for feature in features]

        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        dec_input_ids = [f["dec_input_ids"] for f in features]
        endpoints = [f["endpoints"] for f in features]

        input_ids = [x.squeeze(0) if x.ndim == 3 and x.shape[0] == 1 else x for x in input_ids]
        input_ids = torch.stack(input_ids)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        dec_input_ids = torch.nn.utils.rnn.pad_sequence(
            dec_input_ids, batch_first=True, padding_value=50257  # 50257 is whisper <eot>
        )

        endpoints = torch.nn.utils.rnn.pad_sequence(
            endpoints, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "dec_input_ids": dec_input_ids,
            "endpoints": endpoints,
        }

