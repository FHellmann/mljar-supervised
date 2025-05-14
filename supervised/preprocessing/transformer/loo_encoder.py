import logging
import warnings

from category_encoders.leave_one_out import LeaveOneOutEncoder
from pandas import DataFrame

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class LooEncoder(BaseTransformer):
    def __init__(self, cols=None):
        self.enc = LeaveOneOutEncoder(
            cols=cols,
            verbose=1,
            drop_invariant=False,
            return_df=True,
            handle_unknown="value",
            handle_missing="value",
            random_state=1,
            sigma=0,
        )

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.enc.fit(X, y)

    def transform(self, X: DataFrame, **kwargs):
        return self.enc.transform(X)
