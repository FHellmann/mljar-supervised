from typing import Dict, Any, List, Callable, Type


class AttributeSerializer(object):
    """
    Mixin class providing serialization and deserialization of object attributes.
    """

    def __repr__(self):
        return self.to_dict()

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
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
            exclude = list(self.get_callable_attrs().keys()) + list(self.get_none_attrs().keys())
            for key in exclude:
                params.pop(key, None)
        # Encode custom values
        for key, value in params.items():
            for enc_key, enc_value in attribute_encoders.items():
                if enc_key == key:
                    params[key] = enc_value(value)
        # Remove underscore for all attributes
        return {key[1:]: value for key, value in params.items()}

    def from_dict(self, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        """
        Load object attributes from a dictionary, matching variable names with a pre-underscore.

        For list-type attributes, assigns an empty list if the value is None.
        Custom decoders can be applied to specific attributes.

        Args:
            params (Dict[str, Any]): Dictionary containing attribute values to load.
            **attribute_decoders: Callable[[Any], Any] instances for custom attribute decoding.

        Returns:
            Self
        """
        for key, value in params.items():
            attr_name = f"_{key}"  # Add pre-underscore to match class variable
            if hasattr(self, attr_name):
                current_value = getattr(self, attr_name)
                # Decode custom values
                for dec_key, dec_value in attribute_decoders.items():
                    if dec_key == attr_name:
                        value = dec_value(value)
                if isinstance(current_value, list):
                    setattr(self, attr_name, value if value is not None else [])
                if isinstance(current_value, dict):
                    setattr(self, attr_name, value if value is not None else {})
                else:
                    setattr(self, attr_name, value)

    def get_callable_attrs(self) -> Dict[str, Callable]:
        """
        Get all callable attributes.

        Returns:
            List: List of callable attributes.
        """
        params = self.__dict__.copy()
        return {key: value for key, value in params.items() if callable(value)}

    def get_none_attrs(self) -> Dict[str, None]:
        """
        Get all None attributes.

        Returns:
            List: List of None attributes.
        """
        params = self.__dict__.copy()
        return {key: value for key, value in params.items() if value is None}

    @staticmethod
    def create_from_dict(cls: Type, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> object:
        """
        Factory method to create and initialize an object from serialized parameters.

        Args:
            cls (Type): Class type to instantiate (must subclass AttributeSerializer)
            params (Dict[str, Any]): Dictionary of serialized parameters
            **attribute_decoders: Custom decoders for specific attributes

        Returns:
            object: Initialized instance of the specified class

        Example:
            >>> obj = AttributeSerializer.create_from_dict(
            >>>     MyCustomSerializer,
            >>>     {"param1": 42},
            >>>     _param1=lambda v: v * 2
            >>> )
        """
        object_params = {}
        for key, value in params.items():
            attr_name = f"_{key}"  # Add pre-underscore to match class variable
            # Decode custom values
            for dec_key, dec_value in attribute_decoders.items():
                if dec_key == attr_name:
                    value = dec_value(value)
            object_params[key] = value
        return cls(**object_params)
