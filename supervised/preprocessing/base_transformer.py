

class Preprocessor(object):
    def __init__(self):
        pass

    def fit_and_transform(self, X_train, y_train, sample_weight=None):
        pass

    def transform(self, X_validation, y_validation, sample_weight_validation=None):
        pass


    def inverse_scale_target(self, y):
        pass

    def inverse_categorical_target(self, y):
        pass

    def get_target_class_names(self):
        pass

    def prepare_target_labels(self, y):
        pass

    def to_json(self):
        pass

    def from_json(self, data_json, results_path):
        pass