import sys
import os

max_len = 4096
predict_quote_filename = "predict-quote.jsonlines"


def main(input_folder, preprocessing_folder, model_path, output_folder):
    text2json(input_folder, preprocessing_folder, overwrite=True)
    preprocess(preprocessing_folder, overwrite=True)
    predict(preprocessing_folder, model_path, output_folder, overwrite=True)
    extract(output_folder)


def text2json(input_folder, preprocessing_folder, overwrite=True):
    output_path = os.path.join(preprocessing_folder, "predict.jsonlines")
    if overwrite == False and os.path.isfile(output_path):
        return

    import json
    import spacy

    nlp = spacy.load("de_dep_news_trf")

    with open(output_path, "w") as jsonlines:
        for filename in os.listdir(input_folder):
            with open(os.path.join(input_folder, filename)) as f:
                text = f.read()
                doc = _text2doc(nlp(text), filename, text)
                json.dump(doc, jsonlines, ensure_ascii=False)
                jsonlines.write("\n")


def _text2doc(inp, filename, text):
    sent_map = {s: i for i, s in enumerate(inp.sents)}
    doc = {"annotations": [], "documentName": filename, "originalText": text}
    doc["sentences"] = [
        {
            "begin": s.start,
            "charBegin": s.start_char,
            "charEnd": s.end_char,
            "end": s.end,
            "id": i,
            "text": s.text,
            "tokenIds": [t.i for t in s],
            "tokens": [t.text for t in s],
        }
        for i, s in enumerate(inp.sents)
    ]
    doc["tokens"] = [
        {
            "charBegin": t.idx,
            "charEnd": t.idx + len(t),
            "id": t.i,
            "sentence": sent_map[t.sent],
            "text": t.text,
            "word": _word_in_sentence(t.sent, t),
        }
        for t in inp
    ]
    return doc


def _word_in_sentence(sent, token):
    for i, t in enumerate(sent):
        if t == token:
            return i
    raise ValueError("token is not in sentence")


def preprocess(preprocessing_folder, overwrite=True):
    filename = f"predict.t5-small.german.{max_len}.jsonlines"
    if overwrite == False and os.path.isfile(
        os.path.join(preprocessing_folder, filename)
    ):
        return
    from preprocess_scripts.preprocess_data import (
        main as run_preprocessing,
        PreprocessingOptions,
    )

    options = PreprocessingOptions()
    options.dataset_name = "quote"
    options.input_dir = preprocessing_folder
    options.language = "german"
    options.mark_sentence = True
    options.output_dir = preprocessing_folder
    options.seg_lens = str(max_len)
    options.splits = "predict"
    run_preprocessing(options)


def predict(preprocessing_dir, model_name, output_dir, overwrite=True):
    if overwrite == False and os.path.isfile(
        os.path.join(output_dir, predict_quote_filename)
    ):
        return
    from main_trainer import main as run_predict

    options = {
        "output_dir": output_dir,
        "model_name_or_path": model_name,
        "original_input_dir": preprocessing_dir,
        "do_train": False,
        "data_dir": preprocessing_dir,
        "language": "german",
        "save_dir": output_dir,
        "per_device_eval_batch_size": 1,
        "overwrite_output_dir": True,
        "dataloader_num_workers": 0,
        "predict_with_generate": True,
        "max_eval_len": max_len,
        "max_eval_len_out": max_len,
        "generation_num_beams": 1,
        "generation_max_length": max_len,
        "save_predicts": True,
        "do_predict": False,
        "bf16": True,
        "seq2seq_type": "short_seq",
        "mark_sentence": True,
        "action_type": "integer",
        "align_mode": "l",
        "min_num_mentions": 1,
        "add_mention_end": False,
        "predict_only": True,
    }
    run_predict(options)


def extract(output_folder):
    import json
    import csv

    empty_dict = {}

    with open(os.path.join(output_folder, predict_quote_filename)) as lines, open(
        os.path.join(output_folder, "quotes.csv"), "w"
    ) as x:
        w = csv.DictWriter(
            x,
            fieldnames=[
                "file",
                "quote",
                "type",
                "speaker",
                "cue",
                "addressee",
                "frame",
                "quote_offsets",
                "cue_offsets",
                "speaker_offsets",
                "addressee_offsets",
                "frame_offsets",
            ],
        )
        w.writeheader()
        for line in lines:
            doc = json.loads(line)
            docname = doc["documentName"]
            for anno in doc["annotations"]:
                q = anno["quote"]
                s = anno.get("speaker", empty_dict)
                f = anno.get("frame", empty_dict)
                c = anno.get("cue", empty_dict)
                a = anno.get("addressee", empty_dict)
                w.writerow(
                    {
                        "file": docname,
                        "quote_offsets": ";".join(
                            f"{o['charBegin']}-{o['charEnd']}" for o in q["spans"]
                        ),
                        "quote": q["text"],
                        "type": anno["type"],
                        "speaker_offsets": ";".join(
                            f"{o['charBegin']}-{o['charEnd']}"
                            for o in s.get("spans", [])
                        ),
                        "speaker": s.get("text", ""),
                        "cue_offsets": ";".join(
                            f"{o['charBegin']}-{o['charEnd']}"
                            for o in c.get("spans", [])
                        ),
                        "cue": c.get("text", ""),
                        "frame_offsets": ";".join(
                            f"{o['charBegin']}-{o['charEnd']}"
                            for o in f.get("spans", [])
                        ),
                        "frame": f.get("text", ""),
                        "addressee_offsets": ";".join(
                            f"{o['charBegin']}-{o['charEnd']}"
                            for o in a.get("spans", [])
                        ),
                        "addressee": a.get("text", ""),
                    }
                )
            with open(
                os.path.join(output_folder, docname + ".json"), "w"
            ) as json_writer:
                json.dump(doc, json_writer, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python predict.py <input-dir> <temp-processing-dir> <model-dir> <output-dir>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
