import logging
from decimal import Decimal

import numpy as np
from pandas import DataFrame
from sklearn import preprocessing as sk_preproc

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_serializer import AttributeSerializer
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class LabelEncoder(BaseTransformer, AttributeSerializer):
    def __init__(self, try_to_fit_numeric=False):
        super(LabelEncoder, self).__init__("label_encoder")
        self.lbl = sk_preproc.LabelEncoder()
        self._try_to_fit_numeric = try_to_fit_numeric

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        self.lbl.fit(X)  # list(x.values))
        if self._try_to_fit_numeric:
            logger.debug("Try to fit numeric in LabelEncoder")
            try:
                arr = {Decimal(c): c for c in self.lbl.classes_}
                sorted_arr = dict(sorted(arr.items()))
                self.lbl.classes_ = np.array(
                    list(sorted_arr.values()), dtype=self.lbl.classes_.dtype
                )
            except Exception as e:
                pass

    def transform(self, X: DataFrame, **kwargs) -> DataFrame:
        try:
            return self.lbl.transform(X)
        except ValueError as ve:
            # rescue
            classes = np.unique(X)
            diff = np.setdiff1d(classes, self.lbl.classes_)
            self.lbl.classes_ = np.concatenate((self.lbl.classes_, diff))
            return self.lbl.transform(X)

    def inverse_transform(self, X: DataFrame, **kwargs) -> DataFrame:
        return self.lbl.inverse_transform(X)
