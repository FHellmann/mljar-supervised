import warnings
from typing import List, Callable, Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_serializer import AttributeSerializer


class TextTransformer(BaseTransformer, AttributeSerializer):
    def __init__(self):
        self._new_columns = []
        self._old_column = None
        self._max_features = 100
        self._vectorizer = None

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        column = kwargs["column"]
        self._old_column = column
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            lowercase=True,
            max_features=self._max_features,
        )

        x = X[column][~pd.isnull(X[column])]
        self._vectorizer.fit(x)
        for f in list(self._vectorizer.get_feature_names_out()):
            new_col = self._old_column + "_" + f
            self._new_columns += [new_col]

    def transform(self, X: DataFrame, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter(
                action="ignore", category=pd.errors.PerformanceWarning
            )
            ii = ~pd.isnull(X[self._old_column])
            x = X[self._old_column][ii]
            vect = self._vectorizer.transform(x)

            for f in self._new_columns:
                X[f] = 0.0

            X.loc[ii, self._new_columns] = vect.toarray()
            X.drop(self._old_column, axis=1, inplace=True)
        return X

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
        def vectorizer_encoder(vectorizer):
            return {
                "vocabulary": self._vectorizer.vocabulary_,
                "fixed_vocabulary": {key: int(value) for key, value in vectorizer.vocabulary_.items()},
                "idf": list(self._vectorizer.idf_),
            }

        return super().to_dict(exclude_callables_nones, exclude_attributes,
                               _vectorizer=vectorizer_encoder, **attribute_encoders)

    def from_dict(self, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        def vectorizer_decoder(data):
            vectorizer = TfidfVectorizer(
                analyzer="word",
                stop_words="english",
                lowercase=True,
                max_features=self._max_features,
            )
            vectorizer.vocabulary_ = data.get("vocabulary")
            vectorizer.fixed_vocabulary_ = data.get("fixed_vocabulary")
            vectorizer.idf_ = np.array(data.get("idf"))

        super().from_dict(params, **attribute_decoders)