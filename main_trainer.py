import logging
import os
import sys
from transformers import HfArgumentParser, set_seed
from transformers import AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoConfig, AutoTokenizer
from transformers.integrations import TensorBoardCallback
from arguments import DataArguments, ModelArguments, CorefTrainingArguments \
    as TrainingArguments
from data import CorefDataset, JointDataset
from constants import SPEAKER_START, SPEAKER_END, MENTION_START, MENTION_END, \
    COPY, CLUSTER_NEW, CLUSTERS, SENTENCE_START, SENTENCE_END, SPECIAL_IDS, \
    NON_INT_SPECIAL_IDS, MARK_SPECIAL_IDS, MENTION_END_NON_INT_SPECIAL_IDS, \
    MENTION_ENDS, REQUIRED_PARTS
from trainer import CorefTrainer
from data import ConstrainedDataCollator
from model import ConstrainedT5

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))


def main(args=None):
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    elif args is not None:
        model_args, data_args, training_args = parser.parse_dict(args)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1,
                                                           0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16 training: %s, bf16 training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
        training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("Data arguments %s", data_args)

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=4096)
    if training_args.action_type == "integer":
        num_new_tokens = tokenizer.add_tokens(REQUIRED_PARTS + [COPY])
    elif training_args.action_type == "non_integer":
        if training_args.add_mention_end:
            num_new_tokens = tokenizer.add_tokens(REQUIRED_PARTS + [COPY, CLUSTER_NEW] +CLUSTERS)
        else:
            num_new_tokens = tokenizer.add_tokens(REQUIRED_PARTS + [COPY, CLUSTER_NEW] +MENTION_ENDS)
    else:
        raise ValueError(f"wrong action type {training_args.action_type}")
    if training_args.seq2seq_type == 'short_seq' and \
            training_args.mark_sentence:
        num_new_tokens += tokenizer.add_tokens([SENTENCE_START, SENTENCE_END])
    # we  need to resize model token embeddings
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if training_args.gradient_checkpointing:
        # use_cache is False for training, True for evaluation
        config.use_cache = False
    if training_args.seq2seq_type == 'action' or training_args.seq2seq_type \
            == 'tagging' or training_args.seq2seq_type == 'input_feed':
        if training_args.action_type == "integer":
            special_ids = SPECIAL_IDS
        elif training_args.action_type == "non_integer":
            if training_args.add_mention_end:
                special_ids = MENTION_END_NON_INT_SPECIAL_IDS
            else:
                special_ids = NON_INT_SPECIAL_IDS
        else:
            raise ValueError(f"wrong action type {training_args.action_type}")
        model = ConstrainedT5.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            special_ids=special_ids,
            seq2seq_type=training_args.seq2seq_type,
            action_type=training_args.action_type,
            add_mention_end=training_args.add_mention_end
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path, config=config)

    if len(model.get_input_embeddings().weight) < len(tokenizer):
        logger.info('resize model input embeddings')
        model.resize_token_embeddings(len(tokenizer))

    if training_args.seq2seq_type == 'action' or training_args.seq2seq_type \
            == 'input_feed':
        collator = ConstrainedDataCollator(tokenizer, model=model)
    else:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # model.resize_token_embeddings(len(tokenizer))
    data_cls = JointDataset if training_args.joint_train else CorefDataset
    if data_args.predict_only:
        predict_set = data_cls(tokenizer, data_args, training_args, 'predict')
        train_set = None
        dev_set = predict_set
        test_set = predict_set
        tb_callback = None
    else:
        train_set = data_cls(tokenizer, data_args, training_args, 'train')
        dev_set = data_cls(tokenizer, data_args, training_args, 'dev')
        test_set = data_cls(tokenizer, data_args, training_args, 'test')
        tb_callback = TensorBoardCallback()
    if training_args.parallelize_model:
        model.parallelize()
    trainer = CorefTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=collator,
        callbacks=[] if tb_callback is None else [tb_callback]
    )
    if training_args.do_train:
        if training_args.resume_from_checkpoint is not None and training_args.resume_from_checkpoint != False and training_args.resume_from_checkpoint != "False":
            trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint)
        else:
            trainer.train()
        if trainer.is_world_process_zero():
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        test_results = trainer.evaluate(
            test_set,
            max_length=data_args.max_eval_len_out,
            num_beams=training_args.generation_num_beams)
        logger.info(f'test results: {test_results}')
        dev_results = trainer.evaluate(
            dev_set,
            max_length=data_args.max_eval_len_out,
            num_beams=training_args.generation_num_beams)
        logger.info(f'dev results: {dev_results}')
    if data_args.predict_only:
        trainer.predict(
            predict_set,
            max_length=data_args.max_eval_len_out,
            num_beams=training_args.generation_num_beams,
        )


if __name__ == "__main__":
    main()
