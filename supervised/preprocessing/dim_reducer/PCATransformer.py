from typing import Optional

from pandas import DataFrame

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_storage import AttributeStorage


class PCATransformer(BaseTransformer, AttributeStorage):
    def __init__(self, results_path: Optional[str] = None):
        super().__init__('pca', results_path)

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs) -> None:
        pass

    def transform(self, X: DataFrame, **kwargs) -> DataFrame:
        pass
