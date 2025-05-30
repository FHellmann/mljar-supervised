import unittest

import pandas as pd

from supervised.preprocessing.transformer.categorical_transformer import CategoricalTransformer

import warnings


class CategoricalIntegersTest(unittest.TestCase):
    def test_constructor_preprocessing_categorical(self):
        """
        Check if PreprocessingCategorical object is properly initialized
        """
        categorical = CategoricalTransformer(
            [], CategoricalTransformer.CONVERT_INTEGER
        )
        self.assertEqual(
            categorical._convert_method, CategoricalTransformer.CONVERT_INTEGER
        )
        self.assertEqual(categorical._convert_params, {})

    def test_fit_integers(self):
        # training data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        categorical = CategoricalTransformer(
            df.columns, CategoricalTransformer.CONVERT_INTEGER
        )
        categorical.fit(df)

        self.assertTrue("col2" in categorical._convert_params)
        self.assertTrue("col4" in categorical._convert_params)
        self.assertTrue("a" in categorical._convert_params["col2"])
        self.assertTrue("c" in categorical._convert_params["col2"])
        self.assertTrue("b" not in categorical._convert_params["col2"])
        self.assertTrue("a" in categorical._convert_params["col4"])
        self.assertTrue("b" in categorical._convert_params["col4"])
        self.assertTrue("c" in categorical._convert_params["col4"])

    def test_fit_transform_integers(self):
        # training data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        categorical = CategoricalTransformer(
            df.columns, CategoricalTransformer.CONVERT_INTEGER
        )
        categorical.fit(df)
        df = categorical.transform(df)
        for col in ["col1", "col2", "col3", "col4"]:
            self.assertTrue(col in df.columns)
        self.assertEqual(df["col2"][0], 0)
        self.assertEqual(df["col2"][1], 0)
        self.assertEqual(df["col2"][2], 1)
        self.assertEqual(df["col4"][0], 0)
        self.assertEqual(df["col4"][1], 1)
        self.assertEqual(df["col4"][2], 2)

    def test_future_warning_pandas_transform(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            # training data
            d = {
                "col1": [False, True, True],
                "col2": [False, False, True],
                "col3": [True, False, True],
            }
            df = pd.DataFrame(data=d)
            categorical = CategoricalTransformer(
                df.columns, CategoricalTransformer.CONVERT_INTEGER
            )
            categorical.fit(df)

            df = categorical.transform(df).astype(int)

    def test_future_warning_pandas_inverse_transform(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            # training data
            d = {
                "col1": [False, True, True],
                "col2": [False, False, True],
                "col3": [True, False, True],
            }
            df = pd.DataFrame(data=d)
            categorical = CategoricalTransformer(
                df.columns, CategoricalTransformer.CONVERT_INTEGER
            )
            categorical.fit(df)

            df = categorical.transform(df).astype(int)
            df = categorical.inverse_transform(df)

    def test_fit_transform_inverse_transform_integers(self):
        # training data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        categorical = CategoricalTransformer(
            df.columns, CategoricalTransformer.CONVERT_INTEGER
        )
        categorical.fit(df)
        df_transform = categorical.transform(df).astype(int)
        df_inverse = categorical.inverse_transform(df_transform)
        for col in ["col1", "col2", "col3", "col4"]:
            self.assertTrue(col in df_inverse.columns)
        self.assertEqual(d["col2"][0], df_inverse["col2"][0])
        self.assertEqual(d["col2"][1], df_inverse["col2"][1])
        self.assertEqual(d["col2"][2], df_inverse["col2"][2])
        self.assertEqual(d["col4"][0], df_inverse["col4"][0])
        self.assertEqual(d["col4"][1], df_inverse["col4"][1])
        self.assertEqual(d["col4"][2], df_inverse["col4"][2])

    def test_fit_transform_integers_with_new_values(self):
        # training data
        d_train = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df_train = pd.DataFrame(data=d_train)
        categorical = CategoricalTransformer(
            df_train.columns, CategoricalTransformer.CONVERT_INTEGER
        )
        categorical.fit(df_train)
        # testing data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "d", "f"],
            "col3": [1, 1, 3],
            "col4": ["e", "b", "z"],
        }
        df = pd.DataFrame(data=d)
        df = categorical.transform(df)
        for col in ["col1", "col2", "col3", "col4"]:
            self.assertTrue(col in df.columns)
        self.assertEqual(df["col2"][0], 0)
        self.assertEqual(df["col2"][1], 2)  # new values get higher indexes
        self.assertEqual(df["col2"][2], 3)  # new values get higher indexes
        self.assertEqual(df["col4"][0], 3)  # new values get higher indexes
        self.assertEqual(df["col4"][1], 1)
        self.assertEqual(df["col4"][2], 4)  # new values get higher indexes

    def test_to_and_from_json_convert_integers(self):
        # training data
        d = {
            "col1": [1, 2, 3],
            "col2": ["a", "a", "c"],
            "col3": [1, 1, 3],
            "col4": ["a", "b", "c"],
        }
        df = pd.DataFrame(data=d)
        cat1 = CategoricalTransformer(
            df.columns, CategoricalTransformer.CONVERT_INTEGER
        )
        cat1.fit(df)

        cat2 = CategoricalTransformer(
            df.columns, CategoricalTransformer.CONVERT_INTEGER
        )
        cat2.from_json(cat1.to_json())
        df = cat2.transform(df)
        for col in ["col1", "col2", "col3", "col4"]:
            self.assertTrue(col in df.columns)
        self.assertEqual(df["col2"][0], 0)
        self.assertEqual(df["col2"][1], 0)
        self.assertEqual(df["col2"][2], 1)
        self.assertEqual(df["col4"][0], 0)
        self.assertEqual(df["col4"][1], 1)
        self.assertEqual(df["col4"][2], 2)


if __name__ == "__main__":
    unittest.main()
