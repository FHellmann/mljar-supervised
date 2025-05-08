from abc import abstractmethod
from typing import Optional

from pandas import DataFrame


class BaseTransformer(object):
    """
    Abstract base class for data transformers in a machine learning pipeline.

    This class defines the interface and basic functionality required for any
    transformer, including methods for fitting, transforming, serialization,
    and saving transformer state to disk.
    """

    @abstractmethod
    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs) -> None:
        """
        Fit the transformer to the training data.

        Args:
            X (DataFrame): Training features.
            y (Optional[DataFrame]): Training labels (optional).
            **fit_params: Additional fit parameters.

        Returns:
            None
        """
        pass

    def fit_transform(self, X: DataFrame, y: Optional[DataFrame] = None, **kwargs) -> tuple[DataFrame, DataFrame]:
        """
        Fit the transformer to the data, then transform it.

        This method first fits the transformer to the input data `X` (and optionally `y`),
        then applies the transformation to `X`. It is a convenience method that combines
        `fit` and `transform` into a single call.

        Args:
            X (DataFrame): Input features to fit and transform.
            y (Optional[DataFrame]): Optional target values for supervised transforms.
            **fit_params: Additional parameters to pass to the fit method.

        Returns:
            Any: The transformed version of `X`, as returned by the `transform` method.
        """
        self.fit(X, **kwargs)
        X_transformed = self.transform(X, **kwargs)
        y_transformed = self.transform(y, **kwargs)
        return X_transformed, y_transformed

    @abstractmethod
    def transform(self, X: DataFrame, **kwargs) -> DataFrame:
        """
        Transform the input data using the fitted transformer.

        Args:
            X (DataFrame): Input data.

        Returns:
            DataFrame: Transformed data.
        """
        pass

    def inverse_transform(self, X: DataFrame, **kwargs) -> DataFrame:
        """
        Inverse transform the input data using the fitted transformer.

        Args:
            X (DataFrame): Transformed data.

        Returns:
            DataFrame: Inverse data.
        """
        raise NotImplementedError('Inverse transformation not implemented.')
