import sys
sys.path.append("./")
import torch
import random
import evaluate
import careless_whisper_stream
import careless_whisper_stream.tokenizer as whisper_tokenizer
import jiwer
import time

from torch import nn, Tensor
from torch.optim.adamw import AdamW
from training_code.utils import Config
from careless_whisper_stream import StreamingWhisper
from careless_whisper_stream.audio import HOP_LENGTH
from careless_whisper_stream.streaming_model import EncoderCacheState
from careless_whisper_stream.normalizers import BasicTextNormalizer, EnglishTextNormalizer, GermanTextNormalizer
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from training_code.datasets_classes import TIMIT, WAVsDataset, AlignedTextGridDataset, PrecomputedAlignedDataset
from training_code.collators import WhisperDataCollatorWithPadding, LoRAWhisperDataCollatorWithPadding

class WhisperCustomModel(LightningModule):
    def __init__(self, cfg:Config, model_name="tiny", lang="en", train_dataset: str = None, eval_dataset: str = None, task="transcribe") -> None:
            super().__init__()
            self.save_hyperparameters()
            self.task = task
            self.lang = lang
            self.model = careless_whisper_stream.load_model(model_name)
            self.tokenizer: whisper_tokenizer = whisper_tokenizer.get_tokenizer(True, language=lang)

            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            self.metrics_wer = evaluate.load("wer")
            self.metrics_cer = evaluate.load("cer")

            self.params = self.model

            self.cfg = cfg
            self.__train_dataset = train_dataset
            self.__eval_dataset = eval_dataset

    def forward(self, x):
        return self.model(x)

    def calc_wer_val(self, out: Tensor, labels: Tensor):
        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.normalizer(self.tokenizer.decode(o)))
            l_list.append(self.normalizer(self.tokenizer.decode(l)))
            
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)
        return wer

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out, _, _ = self.model.decoder(dec_input_ids, audio_features)

        if batch_id == 0 and self.current_epoch == 0:
            logits = out[0]
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()

            try:
                pred_text = self.tokenizer.decode(pred_ids)
                print("PREDICTION:", pred_text)
            except:
                pass

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out, _, _ = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        wer = self.calc_wer_val(out, labels)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True, on_epoch=True)
        self.log("val/wer", wer, on_step=False, prog_bar=True, logger=True, on_epoch=True)
        self.log("val/wer_step", wer, on_step=True, prog_bar=True, logger=True, on_epoch=False)


        return {
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.params
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        lr=self.cfg.learning_rate,
                        eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = LinearLR(
            self.optimizer, start_factor=0.5, end_factor=0.8, total_iters=self.t_total
        )

        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def get_dataset(self, ds_path: str, split: str):
        return TIMIT(ds_path, self.tokenizer)
        
    def train_dataloader(self):
        dataset = self.get_dataset(self.__train_dataset, "train")
        return torch.utils.data.DataLoader(dataset,
                        batch_size=self.cfg.batch_size,
                        drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                        collate_fn=WhisperDataCollatorWithPadding()
                        )

    def val_dataloader(self):
        dataset = self.get_dataset(self.__eval_dataset, "val")
        return torch.utils.data.DataLoader(dataset,
                        batch_size=self.cfg.batch_size,
                        num_workers=self.cfg.num_worker,
                        collate_fn=WhisperDataCollatorWithPadding()
                        )
    

class LoRAStreamedWhisper(WhisperCustomModel):
    def __init__(self, cfg: Config, model_name="tiny", lang="en", train_dataset: str = None, eval_dataset: str = None, task="transcribe", rank=8, enc_emb_gran=15, enc_context=1, sim_stream=False, beam_size=None, use_kv_cache=False, use_ca_kv_cache=False, get_times=False, eval_script=False, calc_rwer_arwer=False) -> None:
        super().__init__(cfg, model_name, lang, train_dataset, eval_dataset, task)

        self.automatic_optimization = not cfg.streaming_train

        # if model_name != "large-v2" and not eval_script:
        print(f"enc_emb_gran: {enc_emb_gran}")
        if not cfg.use_from_ft_ckpt:
            self.model: StreamingWhisper = careless_whisper_stream.load_streaming_model_for_train(model_name, 
                                                                                    advisor_ckpt_path=None,
                                                                                    advisor_type=None,
                                                                                    rank=rank,
                                                                                    gran=enc_emb_gran,
                                                                                    extra_gran_blocks=enc_context,
                                                                                    encoder_positional_mode=cfg.encoder_positional_mode,
                                                                                    )
        else:
            self.model: StreamingWhisper = careless_whisper_stream.load_streaming_model(
                cfg.size,
                cfg.gran * 20,
                encoder_positional_mode=cfg.encoder_positional_mode,
            )
        
        for n, p in self.model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False

        if lang == "en":
            self.normalizer = EnglishTextNormalizer()
        elif lang == "de":
            self.normalizer = GermanTextNormalizer()
        else:
            self.normalizer = BasicTextNormalizer()
        self.model.encoder._use_mask(True)

        self.model_name = model_name
        self.rank = self.model.rank
        self.enc_emb_gran = self.model.gran
        self.enc_context = self.model.extra_gran_blocks
        self.encoder_positional_mode = self.model.encoder_positional_mode
        self.simulate_stream = sim_stream
        self.full_stream = cfg.streaming_train
        self.beam_size = beam_size
        self.language = lang
        self.use_kv_cache = use_kv_cache
        self.use_ca_kv_cache = use_ca_kv_cache
        self.get_times = get_times
        self.eval_script = eval_script
        self.calc_rwer_arwer = calc_rwer_arwer
        self.lmdb_paths = cfg.lmdb_paths
        self.stale_cache_bucket_weights = self._parse_stale_cache_bucket_weights(cfg.stale_cache_bucket_weights)

        # for stream mode train
        self.num_frames = self.model.dims.n_audio_ctx // self.enc_emb_gran # 1500 // enc_emb_gran
        self.mel_samples = self.enc_emb_gran * 2 * HOP_LENGTH

        self.params = None

        self.last_out = None
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def _parse_stale_cache_bucket_weights(self, spec: str) -> list[float]:
        try:
            weights = [float(value.strip()) for value in spec.split(",") if value.strip()]
        except ValueError as exc:
            raise ValueError("--stale_cache_bucket_weights must be comma-separated numbers.") from exc

        if not weights or any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("--stale_cache_bucket_weights must contain at least one positive weight.")

        return weights

    def _stale_cache_context_frames(self) -> int:
        raw_frames = int(self.cfg.stale_cache_context_seconds / 0.02)
        frames = (raw_frames // self.enc_emb_gran) * self.enc_emb_gran
        if frames <= 0:
            raise ValueError("--stale_cache_context_seconds is too small for the encoder granularity.")
        if frames > self.model.dims.n_audio_ctx:
            raise ValueError("--stale_cache_context_seconds must not exceed Whisper's 30s audio context.")
        return frames

    def _stale_cache_training_audio_seconds(self) -> float:
        if not self.cfg.stale_encoder_cache_train:
            return 30.0
        if self.cfg.stale_cache_max_stale_seconds < 0:
            raise ValueError("--stale_cache_max_stale_seconds must be non-negative.")
        return self.cfg.stale_cache_context_seconds + self.cfg.stale_cache_max_stale_seconds

    def _point_end_seconds(self, index: int) -> float:
        return (index + 1) * self.enc_emb_gran * 0.02

    def _point_stale_seconds(self, index: int, retained_frames: int) -> float:
        return max(0.0, ((index + 1) * self.enc_emb_gran - retained_frames) * 0.02)

    def _point_label_start_seconds(self, index: int, retained_frames: int) -> float:
        return max(0.0, ((index + 1) * self.enc_emb_gran - retained_frames) * 0.02)

    def _select_evenly(self, values: list[int], count: int) -> list[int]:
        if count <= 0 or not values:
            return []
        if count >= len(values):
            return list(values)
        if count == 1:
            return [values[len(values) // 2]]

        last = len(values) - 1
        return [values[round(i * last / (count - 1))] for i in range(count)]

    def _candidate_stream_points(self, endpoints: Tensor, input_ids: Tensor) -> list[int]:
        valid_endpoints = endpoints[endpoints != -100]
        biggest_endpoint = valid_endpoints.max().item() if valid_endpoints.numel() > 0 else 0.0
        available_granules = input_ids.shape[-1] // (self.enc_emb_gran * 2)
        text_granules = int((biggest_endpoint / 0.02) // self.enc_emb_gran) + int(1 / (self.enc_emb_gran * 0.02)) + 1
        max_granules = min(available_granules, max(self.enc_context + 1, text_granules))
        return list(range(self.enc_context, max_granules))

    def _bucket_for_staleness(self, stale_seconds: float, max_stale_seconds: float) -> int:
        if max_stale_seconds <= 0:
            return 0
        ratio = min(1.0, max(0.0, stale_seconds / max_stale_seconds))
        return min(len(self.stale_cache_bucket_weights) - 1, int(ratio * len(self.stale_cache_bucket_weights)))

    def _weighted_stale_points(self, candidates: list[int], count: int, retained_frames: int, deterministic: bool) -> list[int]:
        if count <= 0 or not candidates:
            return []
        if count >= len(candidates):
            return list(candidates)

        max_stale = max(self._point_stale_seconds(index, retained_frames) for index in candidates)
        buckets = [[] for _ in self.stale_cache_bucket_weights]
        for index in candidates:
            bucket = self._bucket_for_staleness(self._point_stale_seconds(index, retained_frames), max_stale)
            buckets[bucket].append(index)

        if deterministic:
            total_weight = sum(self.stale_cache_bucket_weights)
            raw_counts = [count * weight / total_weight for weight in self.stale_cache_bucket_weights]
            counts = [min(len(bucket), int(raw_count)) for bucket, raw_count in zip(buckets, raw_counts)]
            remaining = count - sum(counts)
            remainders = sorted(
                range(len(raw_counts)),
                key=lambda bucket: raw_counts[bucket] - int(raw_counts[bucket]),
                reverse=True,
            )
            for bucket in remainders:
                if remaining <= 0:
                    break
                room = len(buckets[bucket]) - counts[bucket]
                if room <= 0:
                    continue
                counts[bucket] += 1
                remaining -= 1

            selected = []
            for bucket, bucket_count in zip(buckets, counts):
                selected.extend(self._select_evenly(bucket, bucket_count))

            if len(selected) < count:
                leftovers = [index for index in candidates if index not in set(selected)]
                selected.extend(self._select_evenly(leftovers, count - len(selected)))
            return sorted(selected[:count])

        remaining = list(candidates)
        selected = []
        while remaining and len(selected) < count:
            max_stale = max(self._point_stale_seconds(index, retained_frames) for index in remaining)
            weights = [
                self.stale_cache_bucket_weights[
                    self._bucket_for_staleness(self._point_stale_seconds(index, retained_frames), max_stale)
                ]
                for index in remaining
            ]
            chosen = random.choices(remaining, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining.remove(chosen)
        return sorted(selected)

    def _get_stale_cache_sample_plan(self, endpoints: Tensor, input_ids: Tensor, step: str) -> list[tuple[int, bool]]:
        retained_frames = self._stale_cache_context_frames()
        candidates = self._candidate_stream_points(endpoints, input_ids)
        if not candidates:
            return []

        total_count = len(candidates)
        if self.cfg.streaming_fraction < 1:
            total_count = max(1, int(len(candidates) * self.cfg.streaming_fraction) + 1)
        total_count = min(total_count, len(candidates))

        fresh_fraction = min(1.0, max(0.0, self.cfg.stale_cache_fresh_fraction))
        fresh_count = min(total_count, max(1, round(total_count * fresh_fraction)))
        stale_count = max(0, total_count - fresh_count)

        deterministic = step != "train"
        if deterministic:
            fresh_points = self._select_evenly(candidates, fresh_count)
        else:
            fresh_points = sorted(random.sample(candidates, k=fresh_count))

        stale_candidates = [
            index for index in candidates
            if self._point_stale_seconds(index, retained_frames) > 0
        ]
        stale_points = self._weighted_stale_points(stale_candidates, stale_count, retained_frames, deterministic)

        plan = [(index, False) for index in fresh_points] + [(index, True) for index in stale_points]
        return sorted(plan, key=lambda item: (item[0], item[1]))

    def _calc_labels(self, labels: Tensor, endpoints: Tensor, index: int, out: Tensor = None):
        if self.cfg.self_supervision and out is not None:
            # get predicted tokens
            pred_tokens = torch.argmax(out, dim=-1)

            # find first eot in predictions
            eot_mask = (pred_tokens == self.tokenizer.eot)

            # create new labels
            clone_labels = labels.clone()

            for i in range(labels.shape[0]):
                eot_positions = torch.nonzero(eot_mask[i], as_tuple=False)
                if eot_positions.numel() > 0:
                    eot_idx = eot_positions[0].item()
                    clone_labels[i, :eot_idx + 1] = pred_tokens[i, :eot_idx + 1]
                    clone_labels[i, eot_idx] = self.tokenizer.eot
                    clone_labels[i, eot_idx + 1:] = -100
                else:
                    # no eot found, use all predictions
                    clone_labels[i, :] = pred_tokens[i, :]
            
            return clone_labels
            
        else:
            if not self.cfg.random_masking:
                t_seconds = (index + 1) * self.enc_emb_gran * 0.02
            else:
                t_seconds = index * 0.02
            
            # take only relevant labels into account
            mask = (endpoints <= t_seconds) & (endpoints != -100)
            
            clone_labels = labels.clone()
            
            # ignore irrelevant labels
            clone_labels[~mask] = -100
            
            # Mark the first unavailable token as EOT, or keep the true final EOT
            # once the full transcript is already available.
            for batch_idx in range(labels.shape[0]):
                row_mask = mask[batch_idx]
                false_positions = torch.nonzero(~row_mask, as_tuple=False)

                if false_positions.numel() > 0:
                    eot_idx = false_positions[0].item()
                else:
                    valid_positions = torch.nonzero(labels[batch_idx] != -100, as_tuple=False)
                    eot_idx = valid_positions[-1].item() if valid_positions.numel() > 0 else 0

                clone_labels[batch_idx, eot_idx] = self.tokenizer.eot

            return clone_labels

    def _get_sample_points(self, endpoints: Tensor):
        # base case
        if self.cfg.streaming_fraction == 1:
            return range(self.enc_context, self.num_frames)

        biggest_endpoint = endpoints.max().item()

        # determine last index.
        num_frames = min(int(((biggest_endpoint / 0.02) // self.enc_emb_gran) + int(1 / (self.enc_emb_gran * 0.02)) + 1), self.num_frames) # adding 1 sec of silence
        new_range = range(self.enc_context, num_frames)

        sample_points = random.sample(new_range, k=int(len(new_range) * self.cfg.streaming_fraction) + 1)

        if self.cfg.streaming_random:
            return sample_points

        return sorted(sample_points)

    def _get_sample_points_random_mask(self, endpoints: Tensor):
        biggest_endpoint = endpoints.max().item()
        ls_choices = list(range(5, 55, 5))
        ls_weights = [15, 20, 25, 20, 15, 1, 1, 1, 1, 1]
        lengths = [30]
        curr_sum = 30
        last_index = biggest_endpoint // 0.02

        while curr_sum < last_index:
            l = random.choices(ls_choices, weights=ls_weights, k=1)[0]
            
            if curr_sum + l > last_index:
                l = (int(last_index - curr_sum) // 5) * 5
                lengths.append(l)
                break

            lengths.append(l)
            curr_sum += l
        
        sample_points = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
        mask = torch.full((sample_points[-1], sample_points[-1]), float("-inf"))
        start = 0
        for l in lengths:
            end = start + l
            mask[start:end, :end] = 0
            start = end
        

        # Now sample self.cfg.slices_num points from sample_points, if there are less, return all.
        if len(sample_points) <= self.cfg.slices_num:
            sample_points = sorted(sample_points)
        else:
            sample_points = sorted(random.sample(sample_points, k=self.cfg.slices_num))

        return sample_points, mask

    def _calc_interval_labels(self, labels: Tensor, endpoints: Tensor, start_seconds: float, end_seconds: float):
        available = endpoints != -100
        mask = available & (endpoints <= end_seconds)
        if start_seconds > 0:
            mask = mask & (endpoints > start_seconds)

        clone_labels = labels.clone()
        clone_labels[~mask] = -100

        for batch_idx in range(labels.shape[0]):
            row_available = available[batch_idx]
            after_end = torch.nonzero(
                row_available & (endpoints[batch_idx] > end_seconds),
                as_tuple=False,
            )
            if after_end.numel() > 0:
                eot_idx = after_end[0].item()
            else:
                valid_positions = torch.nonzero(row_available, as_tuple=False)
                eot_idx = valid_positions[-1].item() if valid_positions.numel() > 0 else 0

            clone_labels[batch_idx, eot_idx] = self.tokenizer.eot

        return clone_labels

    def _encode_fresh_retained_window(self, input_ids: Tensor, index: int, retained_frames: int) -> Tensor:
        target_end_frame = (index + 1) * self.enc_emb_gran
        window_start_frame = max(0, target_end_frame - retained_frames)
        mel_window = input_ids[..., window_start_frame * 2: target_end_frame * 2]
        window_frames = mel_window.shape[-1] // 2

        was_stream = self.model.encoder.use_stream
        was_mask = self.model.encoder.use_mask
        try:
            self.model.encoder._use_stream(False)
            self.model.encoder._use_mask(True)
            return self.model.encoder(
                mel_window,
                index=[0, window_frames],
                mask=True,
            )
        finally:
            self.model.encoder._use_stream(was_stream)
            self.model.encoder._use_mask(was_mask)

    def _encode_stale_sliding_window(self, input_ids: Tensor, index: int, retained_frames: int) -> Tensor:
        cache_state = EncoderCacheState(use_sliding=True, max_frames=retained_frames)
        enc_kv_cache, enc_hooks = self.model.install_encoder_kv_cache_hooks(cache_state=cache_state)
        was_stream = self.model.encoder.use_stream
        was_mask = self.model.encoder.use_mask
        original_gran = self.model.encoder.gran

        batch_size = input_ids.shape[0]
        audio_features = torch.empty((batch_size, 0, self.model.dims.n_audio_state), device=input_ids.device)
        mel_buffer = input_ids[..., :0]
        min_mel_frames = self.enc_emb_gran * 2 * (self.enc_context + 1)

        try:
            self.model.encoder._use_stream(True)
            self.model.encoder._use_mask(False)

            for chunk_index in range(index + 1):
                mel_start = chunk_index * self.enc_emb_gran * 2
                mel_end = (chunk_index + 1) * self.enc_emb_gran * 2
                mel_buffer = torch.cat([mel_buffer, input_ids[..., mel_start:mel_end]], dim=-1)

                if mel_buffer.shape[-1] < min_mel_frames:
                    continue

                # Requirement: train on the same stale hidden-state regime as
                # sliding-cache evaluation, including one granule of boundary
                # recompute, while keeping the decoder-visible window <= 30s.
                overlap_frames = min(self.enc_emb_gran, cache_state.cached_frames)
                if overlap_frames > 0:
                    self.model.prune_encoder_kv_cache_tail(enc_kv_cache, overlap_frames)
                    cache_state.cached_frames -= overlap_frames
                    audio_features = audio_features[:, :-overlap_frames].detach()
                    self.model.encoder.gran = original_gran + overlap_frames

                try:
                    new_features = self.model.encoder(
                        mel_buffer,
                        kv_cache=enc_kv_cache,
                        mask=True if overlap_frames > 0 else None,
                    )
                finally:
                    self.model.encoder.gran = original_gran

                frames_to_prune = cache_state.commit(new_features.shape[1])
                self.model.prune_encoder_kv_cache(enc_kv_cache, frames_to_prune)

                retained = audio_features[:, frames_to_prune:] if frames_to_prune > 0 else audio_features
                audio_features = new_features if retained.shape[1] == 0 else torch.cat([retained, new_features], dim=1)

                target_mel_frames = cache_state.cached_frames * 2
                if target_mel_frames > 0 and mel_buffer.shape[-1] > target_mel_frames:
                    mel_buffer = mel_buffer[..., -target_mel_frames:]

                if chunk_index != index:
                    audio_features = audio_features.detach()

            if audio_features.shape[1] == 0:
                raise RuntimeError("Stale-cache training produced no encoder features.")
            return audio_features
        finally:
            self.model.encoder.gran = original_gran
            self.model.encoder._use_stream(was_stream)
            self.model.encoder._use_mask(was_mask)
            for hook in enc_hooks:
                hook.remove()

    def _anchor_stale_cache_loss_for_ddp(self, loss: Tensor) -> Tensor:
        anchor = None
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            term = param.reshape(-1)[0] * 0.0
            anchor = term if anchor is None else anchor + term

        if anchor is None:
            return loss

        # Requirement: stale-cache training intentionally reuses cached encoder
        # K/V tensors, so a particular loss may skip some LoRA projections. DDP
        # with find_unused_parameters=False still needs those trainable tensors
        # present in the graph; this zero term keeps behavior and gradients
        # unchanged while satisfying that contract.
        return loss + anchor.to(dtype=loss.dtype)

    def _forward_step_stale_cache(self, batch, step):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        endpoints = batch["endpoints"]
        retained_frames = self._stale_cache_context_frames()
        sample_plan = self._get_stale_cache_sample_plan(endpoints, input_ids, step)

        if not sample_plan:
            return self._forward_step_stream(batch, step)

        if step == "train":
            optimizer = self.optimizers()

        last_out = None
        last_loss = None
        last_labels = labels

        for index, use_stale_cache in sample_plan:
            if use_stale_cache:
                audio_features = self._encode_stale_sliding_window(input_ids, index, retained_frames)
            else:
                audio_features = self._encode_fresh_retained_window(input_ids, index, retained_frames)

            out = self.model.decoder(dec_input_ids, audio_features, dump_type="None")

            if step == "train":
                optimizer.zero_grad()

            end_seconds = self._point_end_seconds(index)
            start_seconds = self._point_label_start_seconds(index, retained_frames)
            frame_labels = self._calc_interval_labels(labels, endpoints, start_seconds, end_seconds)
            loss = self.loss_fn(out.view(-1, out.size(-1)), frame_labels.view(-1))
            if step == "train":
                loss = self._anchor_stale_cache_loss_for_ddp(loss)

            if step == "train":
                self.manual_backward(loss)
                optimizer.step()

            last_out = out
            last_loss = loss
            last_labels = frame_labels

        return {"out": last_out, "loss": last_loss, "eval_labels": last_labels}

    def _forward_step_stream(self, batch, step):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        endpoints = batch["endpoints"]

        if step == "train":
            optimizer = self.optimizers()

        # forward
        if self.cfg.random_masking:
            sample_points, mask_value = self._get_sample_points_random_mask(endpoints)
            mask_value = mask_value.to(input_ids.device)
        else:
            sample_points = self._get_sample_points(endpoints)

        for i in sample_points:
            if self.cfg.random_masking:
                audio_features = self.model.encoder(input_ids[..., :i * 2], index=[0, i], mask=mask_value)
            else:
                audio_features = self.model.encoder(input_ids[..., :(i + 1) * (self.enc_emb_gran * 2)], index=[0, (i + 1) * self.enc_emb_gran], mask=True)
            out = self.model.decoder(dec_input_ids, audio_features, dump_type="None")

            if step == "train":
                optimizer.zero_grad()

            # loss calc
            frame_labels = self._calc_labels(labels, endpoints, i, out if self.cfg.self_supervision else None)
            loss = self.loss_fn(out.view(-1, out.size(-1)), frame_labels.view(-1))

            # optimizer step if relevant.
            if step == "train":
                self.manual_backward(loss)
                optimizer.step() # might move optimizer step to out of the loop for faster training

        return {"out": out, "loss": loss}

    def _forward_step(self, batch, step):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # self.model.encoder.reset()
        audio_features = self.model.encoder(input_ids, mask=True) # use mask for fast learning, simulates stream mode
        out = self.model.decoder(dec_input_ids, audio_features, dump_type="None")

        # loss calc
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        if step == "train":
            return loss
        
        return out, loss

    def training_step(self, batch, batch_id):
        if batch_id == 0 and self.current_epoch == 0:
            print("\n=== FIRST TRAIN BATCH ===")
            for k, v in batch.items():
                if hasattr(v, "shape"):
                    print(f"{k}: shape={v.shape}")
                else:
                    print(f"{k}: {v}")
            labels_sample = batch["labels"][0]
            # remove ignore_index if present
            labels_sample = labels_sample[labels_sample != -100]

            try:
                labels_sample = labels_sample.detach().cpu().tolist()
                decoded = self.tokenizer.decode(labels_sample)
                print("DECODED LABEL:", decoded)
            except:
                print("Could not decode labels")

        if self.full_stream and self.cfg.stale_encoder_cache_train:
            result = self._forward_step_stale_cache(batch, "train")
            loss = result["loss"]
        elif self.full_stream:
            result = self._forward_step_stream(batch, "train")
            loss = result["loss"]
        else:
            loss = self._forward_step(batch, "train")

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_id):
        eval_labels = batch["labels"]
        if self.full_stream and self.cfg.stale_encoder_cache_train:
            result = self._forward_step_stale_cache(batch, "val")
            out, loss = result["out"], result["loss"]
            eval_labels = result.get("eval_labels", eval_labels)
        elif self.full_stream:
            result = self._forward_step_stream(batch, "val")
            out, loss = result["out"], result["loss"]
        else:
            out, loss = self._forward_step(batch, "val")

        wer = self.calc_wer_val(out, eval_labels)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/wer", wer, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        result = {
            "wer": wer,
            "loss": loss
        }

        if self.calc_rwer_arwer and self.full_stream and "endpoints" in batch:
            rwer, arwer = self._calc_streaming_rwer_arwer(batch)

            self.log("val/rwer", rwer, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("val/arwer", arwer, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

            result["rwer"] = rwer
            result["arwer"] = arwer

        return result

    def predict_step(self, batch, batch_id):
        wavs = batch["wav_path"]
        text = batch["text"]

        results, times = self.model.transcribe(wav_file=wavs[0],
                                            simulate_stream=True,
                                            beam_size=self.beam_size,
                                            language=self.language,
                                            use_ca_kv_cache=self.use_ca_kv_cache,
                                            use_sa_kv_cache=self.use_kv_cache,
                                            get_times=self.get_times)

        return [res.text for res in results], text, wavs[0], times

    def on_train_epoch_start(self):
        random.seed(self.current_epoch + self.cfg.seed)

    def get_dataset(self, ds_path, split):
        print(f"Stream simulation mode: {self.simulate_stream}")
        print(f"Using precomputed features: {self.cfg.precomputed_features}")
        max_audio_seconds = self._stale_cache_training_audio_seconds()
        if self.cfg.stale_encoder_cache_train:
            retained_seconds = self._stale_cache_context_frames() * 0.02
            print(
                "Stale encoder-cache training: "
                f"retained_context={retained_seconds:.2f}s, "
                f"max_audio={max_audio_seconds:.2f}s, "
                f"fresh_fraction={self.cfg.stale_cache_fresh_fraction}, "
                f"bucket_weights={self.stale_cache_bucket_weights}"
            )
        
        if self.full_stream:
            if self.cfg.precomputed_features:
                return PrecomputedAlignedDataset(
                    manifest_path=ds_path,
                    custom_len=self.cfg.custom_len
                )
            
            return AlignedTextGridDataset(
                ds_path=ds_path,
                get_streamed_mel=True,
                gran=self.enc_emb_gran,
                extra_gran_blocks=self.enc_context,
                n_mels=self.model.dims.n_mels,
                multilingual=self.cfg.multilingual,
                max_audio_seconds=max_audio_seconds,
            )
        
        return WAVsDataset(ds_path=ds_path, get_streamed_mel=self.simulate_stream)
    
    def configure_optimizers(self):
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.cfg.weight_decay,
                "lr": self.cfg.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if not p.requires_grad],
                "weight_decay": 0.0,
                "lr": 0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=2, factor=0.5
        )

        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "monitor": "val/loss"}]
    
    def train_dataloader(self):
        dataset = self.get_dataset(self.__train_dataset, "train")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.num_worker,
            pin_memory=True,
            persistent_workers=self.cfg.num_worker > 0,
            prefetch_factor=4 if self.cfg.num_worker > 0 else None,
            collate_fn=LoRAWhisperDataCollatorWithPadding(),
        )

    def val_dataloader(self):
        dataset = self.get_dataset(self.__eval_dataset, "val")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_worker,
            pin_memory=True,
            persistent_workers=self.cfg.num_worker > 0,
            prefetch_factor=2 if self.cfg.num_worker > 0 else None,
            collate_fn=LoRAWhisperDataCollatorWithPadding(),
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["dims"] = self.model.dims.__dict__

    def _decode_logits_to_texts(self, out: Tensor):
        pred_ids = torch.argmax(out, dim=-1)
        texts = []
        for seq in pred_ids:
            texts.append(self.tokenizer.decode(seq.detach().cpu().tolist()).strip().lower())
        return texts

    def _decode_labels_to_texts(self, labels: Tensor):
        labels = labels.clone()
        labels[labels == -100] = self.tokenizer.eot

        texts = []
        for seq in labels:
            texts.append(self.tokenizer.decode(seq.detach().cpu().tolist()).strip().lower())
        return texts

    def _calculate_idsc_batch(self, references, hypotheses):
        insertions = deletions = substitutions = hits = 0

        for ref, hyp in zip(references, hypotheses):
            if not ref and not hyp:
                continue
            if not ref:
                insertions += len(hyp.split())
                continue
            if not hyp:
                deletions += len(ref.split())
                continue

            out = jiwer.process_words(ref, hyp)
            insertions += out.insertions
            deletions += out.deletions
            substitutions += out.substitutions
            hits += out.hits

        return insertions, deletions, substitutions, hits

    def _calc_streaming_rwer_arwer(self, batch):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        endpoints = batch["endpoints"]

        sample_points = self._get_sample_points(endpoints)

        global_rwer_num, global_rwer_den = 0, 0
        global_arwer_num, global_arwer_den = 0, 0

        for i in sample_points:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()

            audio_features = self.model.encoder(
                input_ids[..., :(i + 1) * (self.enc_emb_gran * 2)],
                index=[0, (i + 1) * self.enc_emb_gran],
                mask=True
            )
            out = self.model.decoder(dec_input_ids, audio_features, dump_type="None")

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            processing_latency = time.perf_counter() - start_time

            hyp_texts = self._decode_logits_to_texts(out)

            # RWER: compare against prefix available at audio time rho
            frame_labels_rho = self._calc_labels(labels, endpoints, i)
            ref_texts_rho = self._decode_labels_to_texts(frame_labels_rho)

            i_r, d_r, s_r, c_r = self._calculate_idsc_batch(ref_texts_rho, hyp_texts)
            global_rwer_num += (i_r + d_r + s_r)
            global_rwer_den += (c_r + d_r + s_r)

            # ARWER: compare against prefix available at real time tau = rho + latency
            tau_seconds = ((i + 1) * self.enc_emb_gran * 0.02) + processing_latency
            tau_index = min(
                int(tau_seconds / (self.enc_emb_gran * 0.02)) - 1,
                self.num_frames - 1
            )
            tau_index = max(tau_index, self.enc_context)

            frame_labels_tau = self._calc_labels(labels, endpoints, tau_index)
            ref_texts_tau = self._decode_labels_to_texts(frame_labels_tau)

            i_a, d_a, s_a, c_a = self._calculate_idsc_batch(ref_texts_tau, hyp_texts)
            global_arwer_num += (i_a + d_a + s_a)
            global_arwer_den += (c_a + d_a + s_a)

        rwer = global_rwer_num / global_rwer_den if global_rwer_den > 0 else 0.0
        arwer = global_arwer_num / global_arwer_den if global_arwer_den > 0 else 0.0

        return rwer, arwer
