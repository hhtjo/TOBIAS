from torchmetrics import Metric
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np


# Based on https://github.com/ali-bahrainian/NEWTS/blob/main/topicscore.py
class TopicScore(Metric):
    higher_is_better = True

    def __init__(
        self, topic_model: LdaModel, dictionary: Dictionary, tokenizer=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.topic_model = topic_model
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.add_state("running_topic_score", default=[])

    def _tokenize(self, document: str) -> list[str]:
        if self.tokenizer is not None:
            return self.tokenizer(document)

        return document.split(" ")

    def _doc_topics(self, document: str) -> dict:
        """
        Helper function that returns dictionary of topic ids mapped to prevalence of
        that topic within the given document.
        """

        self.topic_model.minimum_phi_value = 0.01
        self.topic_model.per_word_topics = False

        tokenized = self._tokenize(document)
        vec_bow = self.dictionary.doc2bow(tokenized)

        temp = self.topic_model[vec_bow]
        #temp.sort(key=lambda x: x[1], reverse=True)
        return dict(temp)

    def _calculate_topic_score(self, document: str, topic_id: int) -> float:
        """
        Returns the prevalence of the given topic in a particular document. Can be applied
        to human-written summaries, machine-generated summaries, or the articles themselves.
        """
        prevalences = self._doc_topics(document)

        if topic_id not in prevalences.keys():
            return 0.0
        else:
            return prevalences[topic_id]

    def update(self, document: str, topic_id: int):
        topic_score = self._calculate_topic_score(document, topic_id)
        self.running_topic_score.append(topic_score)

    def compute(self):
        return np.mean(self.running_topic_score)