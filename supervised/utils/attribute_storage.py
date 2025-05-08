import json
import os
from typing import Optional

import joblib

from supervised.utils.attribute_serializer import AttributeSerializer
from supervised.utils import MLJSONEncoder


class AttributeStorage(AttributeSerializer):
    """
    Class for serializing and deserializing object attributes to disk.

    This class extends AttributeSerializer and provides methods to save and load
    object state either as a JSON file (for non-callable, non-None attributes)
    or as a joblib file (if any attribute is callable).

    Attributes:
        _result_file (str): The base name of the file to save results.
        _result_path (Optional[str]): The full path to the result file (if provided).
    """

    def __init__(self, transformer_name: str, results_path: Optional[str] = None):
        """
        Initialize the AttributeStorage.

        Args:
            transformer_name (str): Name of the transformer (used for result file naming).
            results_path (Optional[str]): Directory path to save the result file.
        """
        self._result_file = transformer_name
        if results_path is not None:
            self._result_path = os.path.join(results_path, self._result_file)

    def save(self) -> None:
        """
        Save the object's attributes to disk.

        If the object has any callable attributes, the state is saved as a compressed
        joblib file. Otherwise, non-callable and non-None attributes are saved as a JSON file.

        Raises:
            AttributeError: If _result_path is not set.
        """
        if len(self.get_callable_attrs()) > 0:
            # Save everything as joblib
            joblib_file = f"{self._result_path}.joblib"
            joblib.dump(
                self.to_dict(),
                joblib_file,
                compress=True,
            )
        else:
            # Save all non-callables and non-None values as json
            json_file = f"{self._result_path}.json"
            with open(json_file, "w") as result_file:
                json.dump(self.to_dict(exclude_callables_nones=True), result_file, indent=4, cls=MLJSONEncoder)

    def load(self) -> bool:
        """
        Load the object's attributes from disk, using either a joblib or JSON file.

        This method attempts to load the object's state from disk by first checking for a
        joblib file (which can store all attributes, including callables). If the joblib
        file does not exist, it then checks for a JSON file (which stores only non-callable
        and non-None attributes). The loaded data is used to update the object's attributes
        via the `from_dict` method.

        Returns:
            bool: True if loading was successful (from either file), False otherwise.
        """
        joblib_file = f"{self._result_path}.joblib"
        json_file = f"{self._result_path}.json"
        if os.path.exists(joblib_file):
            # Load everything as joblib
            data = joblib.load(joblib_file)
            self.from_dict(data)
            return True
        elif os.path.exists(json_file):
            # Load all non-callables and non-None values as json
            with open(json_file, "r") as result_file:
                self.from_dict(json.load(result_file))
                return True
        return False
