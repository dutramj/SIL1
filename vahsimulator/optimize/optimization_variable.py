# Python standard libraries
from __future__ import annotations

# 3rd party libraries
import numpy as np
from pydantic import BaseModel, model_validator, ValidationInfo, PrivateAttr
from typing_extensions import Self


class OptimizationVariable(BaseModel):

    model_config = {"arbitrary_types_allowed": True}

    lower_bound: np.float64
    upper_bound: np.float64
    value: np.float64 | None = None
    _register_index: int | None = PrivateAttr(None)

    @model_validator(mode='after')
    def validate_bounds(self) -> Self:

        if self.upper_bound < self.lower_bound:
            raise ValueError('Upper bound must be greater than lower bound!')

        if self.value is None:
            return self

        if self.value < self.lower_bound:
            raise ValueError(f'Value is smaller than lower bound! value: {self.value} | lower_bound: {self.lower_bound}')
        
        if self.value > self.upper_bound:
            raise ValueError(f'Value is greater than upper bound! value: {self.value} | upper_bound: {self.upper_bound}')
        
        return self
    
    @model_validator(mode='after')
    def register_opt_variable(self, info: ValidationInfo) -> Self:

        if info.context and 'opt_variables_registry' in info.context:
            registry = info.context['opt_variables_registry']

            if self._register_index is None:
                registry.append(self)
                self._register_index = len(registry) - 1
            else:
                registry[self._register_index] = self

        return self
    
    @classmethod
    def from_list(cls, value_list) -> Self:

        if len(value_list) != 2:
            raise ValueError(f"'value_list' list must be [lower, upper]! 'value_list' provided: {value_list}")
        
        return cls(
            lower_bound=np.float64(value_list[0]),
            upper_bound=np.float64(value_list[1]),
        )
    
    def __array__(self, dtype=None, copy=None):
        return np.array(self.value, dtype=dtype, copy=copy)
    
    @staticmethod
    def lower_bounds(vars: list[OptimizationVariable]):
        xl = np.array([v.lower_bound for v in vars])
        return xl

    @staticmethod
    def upper_bounds(vars: list[OptimizationVariable]):
        xu = np.array([v.upper_bound for v in vars])
        return xu

    @staticmethod
    def set_values(vars: list[OptimizationVariable], x: np.ndarray):

        if len(vars) != len(x):
            raise ValueError("Vector length mismatch")

        for v, val in zip(vars, x):
            v.value = val
