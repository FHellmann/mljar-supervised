from supervised.preprocessing.dim_reducer.PCATransformer import PCATransformer
from supervised.preprocessing.dim_reducer.SVDTransformer import SVDTransformer
from supervised.preprocessing.oversampler.SMOTETransformer import SMOTETransformer

class PreprocessingRegistry:
    registry = {}

    @staticmethod
    def add(name, preprocessing_class, default_params):
        PreprocessingRegistry.registry[name] = {
            "class": preprocessing_class,
            "default_params": default_params,
        }

    @staticmethod
    def get_class(name):
        return PreprocessingRegistry.registry[name]["class"]

    @staticmethod
    def get_default_params(name):
        return PreprocessingRegistry.registry[name]["default_params"]


PreprocessingRegistry.add("pca", PCATransformer, {"variance_threshold": 0.9})
PreprocessingRegistry.add("svd", SVDTransformer, {"n_components": 2})
PreprocessingRegistry.add("smote", SMOTETransformer, {"random_state": 42})
