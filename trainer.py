
from transformers.utils import is_apex_available
from transformers import Seq2SeqTrainer
import os
import json
import re
from packaging import version
import torch.nn as nn
from collections import defaultdict
from metrics import CorefAllMetrics
from typing import Dict, Union, Any, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalLoopOutput
from data import get_document_predicts, parse_int_output_tokens, \
    parse_short_target_tokens, parse_nonint_output_tokens
from constants import SPECIAL_IDS, MARK_SPECIAL_IDS, NON_INT_SPECIAL_IDS, \
    MENTION_END_NON_INT_SPECIAL_IDS
from transformers.deepspeed import deepspeed_init
from eval_quotes import evaluate
from quotes import convert_to_quote_json

from transformers.utils import logging, is_torch_tpu_available, \
    is_sagemaker_mp_enabled, is_safetensors_available

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse(
        "1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_safetensors_available():
    import safetensors.torch
if is_apex_available():
    from apex import amp

from transformers import LogitsProcessorList
from logits_processor import ShortSeqProcessor, IntProcessor, NonIntProcessor
from transformers.trainer_seq2seq import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class CorefTrainer(Seq2SeqTrainer):

    def __init__(
        self,
        *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            compute_metrics=self.compute_metrics
        )

    def compute_metrics(self, pred: Any) -> Dict:
        
        predicts = pred.predictions
        doc_labels, samples, split, id_to_name = self.eval_info
        del self.eval_info
        print("deleted eval_info")
        if self.args.joint_train:
            data_names = self.args.joint_data_names.split(',')
            joint_threds = [
                int(t) for t in self.args.joint_min_num_mentions.split(',')]
            name_to_threds = {n: t for n, t in zip(data_names, joint_threds)}
        documents_to_chunk_data = defaultdict(list)
        documents_to_chunk_gold = defaultdict(list)
        predictions = {}
        golds = {}
        assert len(samples) <= len(predicts) # with batching and multi-GPU predicts may be longer
        out_sents = []
        last_doc_id = re.sub(r'_\d+$', '', samples[0]['doc_key'])
        for sample, predict in zip(samples, predicts):
            doc_key = sample['doc_key']
            doc_id = re.sub(r'_\d+$', '', doc_key)
            # require convert to ids first
            input_ids = sample['sentence']
            subtoken_map = sample['subtoken_map']
            offset = sample['offset']
            # remove bos
            predict_ids = predict[1:].tolist()
            gold_data = sample['seg_clusters']
            if self.args.joint_train:
                thred = name_to_threds[id_to_name[doc_id]]
            else:
                thred = self.args.min_num_mentions
            if self.args.seq2seq_type == "short_seq":
                special_ids = MARK_SPECIAL_IDS if self.args.mark_sentence \
                    else SPECIAL_IDS
                pred_data, aligned_input_ids, aligned_pred_ids = \
                    parse_short_target_tokens(input_ids, predict_ids,
                                              special_ids, subtoken_map,
                                              self.tokenizer,
                                              self.args.align_mode,
                                              thred,
                                              self.args.mark_sentence
                                              )
                predict_ids = [t for t in predict_ids if t != self.tokenizer.pad_token_id]
                pred_tokens = self.tokenizer.convert_ids_to_tokens(
                    predict_ids)
                out_predict = {
                    'doc_key': doc_key,
                    'pred_tokens': pred_tokens,
                    'pred_text': self.tokenizer.convert_tokens_to_string(
                        pred_tokens),
                    'pred_aligned_text': self.tokenizer.convert_ids_to_tokens(
                        aligned_pred_ids
                    ),
                    'predict_clusters': pred_data,
                    'gold_clusters': gold_data,
                    'input_aligned_text': self.tokenizer.convert_ids_to_tokens(
                        aligned_input_ids
                    )
                }
            else:
                is_tagging = (self.args.seq2seq_type == 'tagging')
                if self.args.action_type == 'integer':
                    pred_data, pred_token_mentions, predict_ids = \
                        parse_int_output_tokens(
                            input_ids,
                            predict_ids,
                            SPECIAL_IDS,
                            subtoken_map,
                            self.tokenizer,
                            thred, is_tagging)
                else:
                    special_ids = MENTION_END_NON_INT_SPECIAL_IDS if \
                        self.args.add_mention_end else NON_INT_SPECIAL_IDS
                    pred_data, pred_token_mentions, predict_ids = \
                        parse_nonint_output_tokens(
                            input_ids,
                            predict_ids,
                            special_ids,
                            subtoken_map,
                            self.tokenizer, self.args.add_mention_end,
                            thred)
                pred_token_mentions = [(m[0] + offset, m[1] + offset) for m in
                                       pred_token_mentions]
                predict_ids = [t for t in predict_ids if t != self.tokenizer.pad_token_id]
                pred_tokens = self.tokenizer.convert_ids_to_tokens(
                    predict_ids)
                out_predict = {'doc_key': doc_key,
                               'pred_tokens': pred_tokens,
                               'pred_text':
                                   self.tokenizer.convert_tokens_to_string(
                                       pred_tokens),
                               'predict_clusters': pred_data,
                               'gold_clusters': gold_data,
                               'predict_token_mentions': pred_token_mentions
                               }
            # list of (m1,m2)

            documents_to_chunk_data[doc_id].extend(pred_data)
            documents_to_chunk_gold[doc_id].extend(gold_data)

            out_sents.append(out_predict)
            if doc_id != last_doc_id:
                predictions[last_doc_id] = \
                    documents_to_chunk_data[
                        last_doc_id]
                golds[last_doc_id] = \
                    documents_to_chunk_gold[
                        last_doc_id]
            last_doc_id = doc_id
        # final one
        predictions[last_doc_id] = \
            documents_to_chunk_data[last_doc_id]
        
        golds[last_doc_id] = \
            documents_to_chunk_gold[last_doc_id]
        
        
        
        # print(predictions)
        if self.args.joint_train:
            predictions_list = defaultdict(list)
            labels_list = defaultdict(list)
            golds_list = defaultdict(list)
        else:
            predictions_list = []
            labels_list = []
            golds_list = []
        for document_id, doc_label in doc_labels.items():
            if self.args.joint_train:
                predictions_list[id_to_name[document_id]].append(
                    predictions[document_id])
                labels_list[id_to_name[document_id]].append(doc_label)
                golds_list[id_to_name[document_id]].append(golds[document_id])
            else:
                predictions_list.append(predictions[document_id])
                labels_list.append(doc_label)
                golds_list.append(golds[document_id])
    
        quote_gold_path = f"{self.eval_dataset.data_args.original_input_dir}/{split}.jsonlines"
        quote_predictions = convert_to_quote_json(quote_gold_path, predictions, self.args.seq2seq_type == "short_seq" and self.args.mark_sentence)

        if self.args.joint_train:
            label_results = {}
            gold_results = {}
            for dn in predictions_list.keys():
                metrics = CorefAllMetrics().get_all_metrics(
                    labels_list[dn],
                    predictions_list[dn])
                metrics_golds = CorefAllMetrics().get_all_metrics(
                    golds_list[dn],
                    predictions_list[dn])
                single_label_results = {
                    f'{dn}_{metric_name}_{x}': v
                    for metric_name, metric_values in metrics['micro'].items()
                    for x, v in metric_values.items()
                }
                single_gold_results = {
                    f'{dn}_gold_{metric_name}_{x}': v
                    for metric_name, metric_values in
                    metrics_golds['micro'].items()
                    for x, v in metric_values.items()
                }
                label_results.update(single_label_results)
                gold_results.update(single_gold_results)

        else:
            metrics = CorefAllMetrics().get_all_metrics(labels_list,
                                                        predictions_list)
            metrics_golds = CorefAllMetrics().get_all_metrics(golds_list,
                                                              predictions_list)
            quote_metrics = evaluate(quote_predictions, quote_gold_path)
            label_results = {
                f'{metric_name}_{x}': v
                for metric_name, metric_values in metrics['micro'].items()
                for x, v in metric_values.items()
            }
            gold_results = {
                f'gold_{metric_name}_{x}': v
                for metric_name, metric_values in metrics_golds['micro'].items()
                for x, v in metric_values.items()
            }
            quote_results = {
                f'quote_{k}': v for k, v in quote_metrics.items()
            }

        results = {**label_results, **gold_results, **quote_results}
        if self.args.joint_train:
            avg_f1s = [results[f"{dname}_average_f1"] for dname in
                       data_names]
            results["average_f1"] = sum(avg_f1s) / len(avg_f1s)
        if self.is_world_process_zero() and self.args.save_predicts:
            os.makedirs(self.args.save_dir, exist_ok=True)
            save_path = os.path.join(self.args.save_dir,
                                     f'{split}-predicts.jsonlines')
            results_path = os.path.join(self.args.save_dir,
                                        f'{split}-results.json')
            quote_path = os.path.join(self.args.save_dir,
                                        f'{split}-quote.jsonlines')
            with open(save_path, 'w', encoding="utf-8") as f:
                for p in out_sents:
                    json.dump(p, f, ensure_ascii=False)
                    # f.write(json.dumps(p, ensure_ascii=False))
                    f.write('\n')
            with open(results_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False)
            with open(quote_path, "w") as f:
                for quote in quote_predictions:
                    json.dump(quote, f, ensure_ascii=False)
                    f.write("\n")

        return results

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = False,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        eval_dataset = getattr(dataloader, "dataset", None)
        if hasattr(self, "eval_info"):
            raise ValueError("eval_info must not be present!")
        self.eval_info = (eval_dataset.doc_labels, eval_dataset.samples, eval_dataset.split, eval_dataset.id_to_name if self.args.joint_train else None)
        
        return super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys:
                list of ignore keys

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys
            )
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        gen_kwargs["max_length"] = (
            gen_kwargs["max_length"] if gen_kwargs.get(
                "max_length") is not None else self.model.config.max_length
        )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get(
                "num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get(
                "synced_gpus") is not None else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(self.model,
                   "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        #  add our logits_processor here
        if self.args.seq2seq_type != 'short_seq':
            if self.args.action_type == 'non_integer':
                special_ids = MENTION_END_NON_INT_SPECIAL_IDS if \
                    self.args.add_mention_end else NON_INT_SPECIAL_IDS
                gen_kwargs['logits_processor'] = LogitsProcessorList(
                    [NonIntProcessor(generation_inputs, special_ids,
                                     self.args.seq2seq_type,
                                     self.args.add_mention_end)])
            else:
                gen_kwargs['logits_processor'] = LogitsProcessorList(
                    [IntProcessor(generation_inputs, SPECIAL_IDS,
                                  self.args.seq2seq_type)])
        elif self.args.mark_sentence:
            gen_kwargs['logits_processor'] = LogitsProcessorList(
                [ShortSeqProcessor(generation_inputs, MARK_SPECIAL_IDS)])
        # if self.args.use_peft:
        #     gen_kwargs["input_ids"] = generation_inputs
        #     gen_kwargs["use_cache"] = True
        #     generated_tokens = self.model.generate(
        #         **gen_kwargs,
        #     )
        # else:
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens,
                                                            gen_kwargs[
                                                                "max_length"])

        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs,
                                               inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else
                            outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels,
                                                      gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
