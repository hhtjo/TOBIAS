import nltk
import evaluate
import numpy as np
from topic_score import TopicScore
from gensim.models import LdaModel
from gensim.corpora import Dictionary

rouge_metric = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

lda = LdaModel.load("../../lda_model/250/lda.model")
dictionary = Dictionary.load("../../lda_model/250/dictionary.dic")
topic_metric = TopicScore(lda, dictionary)

def calculate_topic_score(preds, dataset, clean_chars=True):
    for i, pred in enumerate(preds):
        topic_id = int(dataset[i]["tid"])
        if clean_chars:
            bad_chars = [";", ":", "!", "*", ".", ",", "'", '"']
            pred = pred.lower()
            for i in bad_chars:
                pred = pred.replace(i, "")
        topic_metric(pred, topic_id)

    result = topic_metric.compute()
    topic_metric.reset()

    return result

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def calculate_metrics(eval_preds, tokenizer, eval_dataset=None):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=False
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}

    stemmed_rouge = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    for k,v in stemmed_rouge.items():
        result[f"stemmed.{k}"] = round(v*100,4)

    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    result["bleu.precisions"] = np.mean(bleu_result["precisions"])
    result["bleu.bleu"] = bleu_result["bleu"]
    result["bleu.translation_length"] = bleu_result["translation_length"]

    bertscore_result = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        model_type="distilbert-base-uncased",
    )
    result["bertscore.precision"] = np.mean(bertscore_result["precision"])
    result["bertscore.recall"] = np.mean(bertscore_result["recall"])
    result["bertscore.f1"] = np.mean(bertscore_result["f1"])
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    if eval_dataset is not None:
        result["clean.topic_score"] = calculate_topic_score(decoded_preds, eval_dataset, clean_chars=True)
        result["topic_score"] = calculate_topic_score(decoded_preds, eval_dataset, clean_chars=False)
    return result