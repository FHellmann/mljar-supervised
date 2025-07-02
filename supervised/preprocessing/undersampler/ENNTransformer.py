from typing import Optional, Union, Tuple

from imblearn.under_sampling import EditedNearestNeighbours
from pandas import DataFrame, Series

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_storage import AttributeStorage


class ENNTransformer(BaseTransformer, AttributeStorage):

    def __init__(self, results_path: Optional[str] = None):
        super().__init__("enn", results_path)
        self.results_path = results_path
        self._error = None
        self.enn = None
        self.X_resampled = None
        self.y_resampled = None

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs) -> None:
        if y is None:
            raise ValueError("y must be provided to fit EditedNearestNeighboursTransformer.")
        self.enn = EditedNearestNeighbours()

    def transform(
        self, X: DataFrame, y: Optional[Union[DataFrame, Series]] = None, **kwargs
    ) -> Tuple[DataFrame, Union[DataFrame, Series]]:
        if y is None:
            raise ValueError("y must be provided to transform EditedNearestNeighboursTransformer.")

        print(
            "DEBUG (EditedNearestNeighboursTransformer.py; transform): Shapes before undersampling: ",
            X.shape,
            y.shape,
        )
        X_res, y_res = self.enn.fit_resample(X, y)
        print(
            "DEBUG (EditedNearestNeighboursTransformer.py; transform): Resampled shapes: ",
            X_res.shape,
            y_res.shape,
        )

        self.X_resampled = DataFrame(X_res, columns=X.columns)

        if isinstance(y, Series):
            self.y_resampled = Series(y_res, name=y.name)
        elif isinstance(y, DataFrame):
            self.y_resampled = DataFrame(y_res, columns=y.columns)
        else:
            raise TypeError("Unsupported type for y.")

        return self.X_resampled, self.y_resampled

    def fit_transform(
        self, X: DataFrame, y: Optional[Union[DataFrame, Series]] = None, **kwargs
    ) -> Tuple[DataFrame, Union[DataFrame, Series]]:
        self.fit(X, y, **kwargs)
        return self.transform(X, y, **kwargs)

