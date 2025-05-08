from typing import Dict, Any, List, Callable


class AttributeSerializer(object):
    """
    Mixin class providing serialization and deserialization of object attributes.

    Methods:
        to_dict: Serializes the object's attributes to a dictionary, with options to exclude callables, None values, or specific attributes.
        from_dict: Deserializes a dictionary into the object's attributes, matching variable names with a pre-underscore.
        _has_callable_attr: Checks if the object has any callable attributes.
    """

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any]:
        """
        Serialize the object's attributes to a dictionary.

        Args:
            exclude_callables_nones (bool): If True, exclude callables and None values from the output dictionary.
            exclude_attributes (Optional[List[str]]): List of attribute names to explicitly exclude.
            **attribute_encoders: Callable[[Any], Any] instances for custom attribute encoding.

        Returns:
            Dict[str, Any]: Dictionary of the object's attributes with leading underscores removed.
        """
        params = self.__dict__.copy()
        # Exclude values
        if exclude_callables_nones:
            exclude = []
            for key, value in params.items():
                if not callable(value) or value is None or exclude_attributes is not None and key in exclude_attributes:
                    exclude.append(key)
            for key in exclude:
                params.pop(key, None)
        # Encode custom values
        for key, value in params.items():
            for enc_key, enc_value in attribute_encoders.items():
                if enc_key == key:
                    params[key] = enc_value(value)
        # Remove underscore for all attributes
        return {key[1:]: value for key, value in params.items()}

    def from_dict(self, data_json: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        """
        Load object attributes from a dictionary, matching variable names with a pre-underscore.

        For list-type attributes, assigns an empty list if the value is None.
        Custom decoders can be applied to specific attributes.

        Args:
            data_json (Dict[str, Any]): Dictionary containing attribute values to load.
            **attribute_decoders: Callable[[Any], Any] instances for custom attribute decoding.

        Returns:
            None
        """
        for key, value in data_json.items():
            attr_name = f"_{key}"  # Add pre-underscore to match class variable
            if hasattr(self, attr_name):
                current_value = getattr(self, attr_name)
                # Decode custom values
                for enc_key, enc_value in attribute_decoders.items():
                    if enc_value == attr_name:
                        value = enc_value(value)
                if isinstance(current_value, list):
                    setattr(self, attr_name, value if value is not None else [])
                else:
                    setattr(self, attr_name, value)

    def _has_callable_attr(self):
        """
        Check if the object has any callable attributes.

        Returns:
            bool: True if any attribute is callable, False otherwise.
        """
        params = self.__dict__.copy()
        return any([callable(value) for value in params.values()])
