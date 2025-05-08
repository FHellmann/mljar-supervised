import json
import logging
import warnings
from typing import List, Callable, Any, Dict

import pandas as pd
from category_encoders.leave_one_out import LeaveOneOutEncoder
from pandas import DataFrame

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_serializer import AttributeSerializer
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class LooEncoder(BaseTransformer, AttributeSerializer):
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

    def to_dict(self, exclude_callables_nones: bool = False, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
        def enc_encoder(loo_enc):
            data_json = {
                "cols": loo_enc.cols,
                "dim": loo_enc._dim,
                "mean": float(loo_enc._mean),
                "feature_names": loo_enc.get_feature_names_out(),
                "mapping": {},
            }
            for k, v in loo_enc.mapping.items():
                data_json["mapping"][k] = v.to_json()
            return data_json

        return super().to_dict(exclude_callables_nones=exclude_callables_nones, exclude_attributes=exclude_attributes,
                        enc=enc_encoder, **attribute_encoders)

    def from_dict(self, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        def enc_decoder(loo_enc):
            loo_enc.cols = params.get("cols")
            loo_enc._dim = params.get("dim")
            loo_enc._mean = params.get("mean")
            loo_enc.feature_names = params.get("feature_names")
            loo_enc.mapping = {}
            for k, v in params.get("mapping", {}).items():
                loo_enc.mapping[k] = pd.DataFrame(json.loads(v))

        super().from_dict(params, enc=enc_decoder, **attribute_decoders)
