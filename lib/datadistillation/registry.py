"""
Method registry for dataset distillation methods.

This module implements a factory pattern that allows easy registration
and creation of different distillation methods. New methods can be added
simply by decorating the class with @DistillationMethodRegistry.register().
"""

from typing import Dict, Type, Optional
from .base import BaseDistillationMethod


class DistillationMethodRegistry:
    """
    Registry for distillation methods using the factory pattern.

    This class maintains a registry of all available distillation methods
    and provides a factory method to create instances.

    Example usage:
        # Register a method
        @DistillationMethodRegistry.register('my_method')
        class MyMethod(BaseDistillationMethod):
            ...

        # Create an instance
        method = DistillationMethodRegistry.create('my_method', **config)

        # List all available methods
        methods = DistillationMethodRegistry.list_methods()
    """

    _methods: Dict[str, Type[BaseDistillationMethod]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a distillation method.

        Args:
            name: Unique identifier for the method (e.g., 'frepo', 'mtt', 'kip')

        Returns:
            Decorator function

        Example:
            @DistillationMethodRegistry.register('frepo')
            class FRePoMethod(BaseDistillationMethod):
                pass
        """
        def decorator(method_class: Type[BaseDistillationMethod]):
            if name in cls._methods:
                raise ValueError(
                    f"Method '{name}' is already registered. "
                    f"Existing: {cls._methods[name].__name__}, "
                    f"New: {method_class.__name__}"
                )

            if not issubclass(method_class, BaseDistillationMethod):
                raise TypeError(
                    f"Method class {method_class.__name__} must inherit "
                    f"from BaseDistillationMethod"
                )

            cls._methods[name] = method_class
            return method_class

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> BaseDistillationMethod:
        """
        Factory method to create a distillation method instance.

        Args:
            name: Name of the registered method
            **kwargs: Configuration parameters passed to the method's __init__

        Returns:
            Instance of the requested method

        Raises:
            ValueError: If method name is not registered

        Example:
            method = DistillationMethodRegistry.create(
                'frepo',
                num_nn_state=10,
                max_online_updates=100
            )
        """
        if name not in cls._methods:
            available = ', '.join(cls.list_methods())
            raise ValueError(
                f"Unknown method: '{name}'. "
                f"Available methods: {available}"
            )

        method_class = cls._methods[name]
        return method_class(**kwargs)

    @classmethod
    def get_method_class(cls, name: str) -> Type[BaseDistillationMethod]:
        """
        Get the class of a registered method without instantiating it.

        Args:
            name: Name of the registered method

        Returns:
            Method class

        Raises:
            ValueError: If method name is not registered
        """
        if name not in cls._methods:
            available = ', '.join(cls.list_methods())
            raise ValueError(
                f"Unknown method: '{name}'. "
                f"Available methods: {available}"
            )

        return cls._methods[name]

    @classmethod
    def list_methods(cls) -> list:
        """
        List all registered method names.

        Returns:
            List of method names (strings)

        Example:
            >>> DistillationMethodRegistry.list_methods()
            ['frepo', 'mtt', 'kip', 'dc', 'dm']
        """
        return sorted(list(cls._methods.keys()))

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a method is registered.

        Args:
            name: Method name to check

        Returns:
            True if method is registered, False otherwise
        """
        return name in cls._methods

    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a method (primarily for testing).

        Args:
            name: Name of the method to unregister

        Raises:
            ValueError: If method name is not registered
        """
        if name not in cls._methods:
            raise ValueError(f"Method '{name}' is not registered")

        del cls._methods[name]

    @classmethod
    def get_method_info(cls, name: str) -> Dict[str, str]:
        """
        Get information about a registered method.

        Args:
            name: Name of the method

        Returns:
            Dictionary with method information (name, class, docstring)

        Raises:
            ValueError: If method name is not registered
        """
        if name not in cls._methods:
            raise ValueError(f"Unknown method: '{name}'")

        method_class = cls._methods[name]
        return {
            'name': name,
            'class': method_class.__name__,
            'module': method_class.__module__,
            'docstring': method_class.__doc__ or 'No documentation available',
        }

    @classmethod
    def print_registry(cls):
        """
        Print all registered methods with their information.

        Useful for debugging and seeing what methods are available.
        """
        methods = cls.list_methods()
        if not methods:
            print("No methods registered.")
            return

        print(f"Registered Distillation Methods ({len(methods)}):")
        print("=" * 60)

        for name in methods:
            info = cls.get_method_info(name)
            print(f"\n  Name: {name}")
            print(f"  Class: {info['class']}")
            print(f"  Module: {info['module']}")
            doc = info['docstring'].strip().split('\n')[0]
            print(f"  Description: {doc}")

        print("\n" + "=" * 60)


# Convenience function for registering methods
def register_method(name: str):
    """
    Convenience wrapper for DistillationMethodRegistry.register().

    Args:
        name: Method name

    Returns:
        Decorator function

    Example:
        @register_method('my_method')
        class MyMethod(BaseDistillationMethod):
            pass
    """
    return DistillationMethodRegistry.register(name)
