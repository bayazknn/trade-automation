"""
Config Loader Module

Loads and parses technical_indicators_config.json to extract
indicator definitions and optimizable parameters.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ParamConfig:
    """Configuration for a single parameter."""
    name: str
    default: Union[int, float]
    range: Tuple[Union[int, float], Union[int, float]]
    param_type: str  # 'int' or 'float'

    @classmethod
    def from_dict(cls, name: str, config: Dict) -> "ParamConfig":
        return cls(
            name=name,
            default=config.get("default"),
            range=tuple(config.get("range", [config.get("default"), config.get("default")])),
            param_type=config.get("type", "int")
        )


@dataclass
class SignalConfig:
    """Configuration for entry or exit signal."""
    signal_type: str  # 'threshold' or 'crossover'
    left: str
    operator: str
    right: str
    constant: Optional[ParamConfig] = None
    factor: Optional[ParamConfig] = None

    @classmethod
    def from_dict(cls, config: Dict, signal_name: str) -> Optional["SignalConfig"]:
        if not config:
            return None

        condition = config.get("condition", {})

        # Parse constant if present
        constant = None
        if "constant" in config and isinstance(config["constant"], dict):
            const_config = config["constant"]
            if "range" in const_config:
                constant = ParamConfig(
                    name=f"{signal_name}_constant",
                    default=const_config.get("default"),
                    range=tuple(const_config.get("range")),
                    param_type="float"
                )

        # Parse factor if present
        factor = None
        if "factor" in config and isinstance(config["factor"], dict):
            factor_config = config["factor"]
            if "range" in factor_config:
                factor = ParamConfig(
                    name=f"{signal_name}_factor",
                    default=factor_config.get("default"),
                    range=tuple(factor_config.get("range")),
                    param_type="float"
                )

        return cls(
            signal_type=config.get("signal_type", "threshold"),
            left=condition.get("left", ""),
            operator=condition.get("operator", ""),
            right=condition.get("right", ""),
            constant=constant,
            factor=factor
        )


@dataclass
class IndicatorConfig:
    """Complete configuration for an indicator."""
    name: str
    full_name: str
    category: str
    talib_function: str
    params: Dict[str, ParamConfig] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    entry: Optional[SignalConfig] = None
    exit: Optional[SignalConfig] = None
    sma_period: Optional[int] = None
    dual_ma: Optional[Dict[str, int]] = None
    usage: Optional[str] = None
    requires: List[str] = field(default_factory=list)

    def get_all_optimizable_params(self) -> Dict[str, ParamConfig]:
        """Get all parameters that can be optimized."""
        all_params = dict(self.params)

        # Add entry constant/factor
        if self.entry:
            if self.entry.constant:
                all_params[self.entry.constant.name] = self.entry.constant
            if self.entry.factor:
                all_params[self.entry.factor.name] = self.entry.factor

        # Add exit constant/factor
        if self.exit:
            if self.exit.constant:
                all_params[self.exit.constant.name] = self.exit.constant
            if self.exit.factor:
                all_params[self.exit.factor.name] = self.exit.factor

        # Add dual_ma parameters if present
        if self.dual_ma:
            all_params["fast_period"] = ParamConfig(
                name="fast_period",
                default=self.dual_ma.get("fast_period", 9),
                range=(3, 50),
                param_type="int"
            )
            all_params["slow_period"] = ParamConfig(
                name="slow_period",
                default=self.dual_ma.get("slow_period", 21),
                range=(10, 200),
                param_type="int"
            )

        # Add sma_period if present
        if self.sma_period:
            all_params["sma_period"] = ParamConfig(
                name="sma_period",
                default=self.sma_period,
                range=(5, 50),
                param_type="int"
            )

        return all_params


class ConfigLoader:
    """Loads and manages technical indicators configuration."""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self._raw_config: Dict = {}
        self._indicators: Dict[str, IndicatorConfig] = {}
        self._operators: Dict[str, str] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file."""
        with open(self.config_path, "r") as f:
            self._raw_config = json.load(f)

        self._operators = self._raw_config.get("operators", {})
        self._parse_indicators()

    def _parse_indicators(self):
        """Parse all indicator configurations."""
        indicators_raw = self._raw_config.get("indicators", {})

        for name, config in indicators_raw.items():
            self._indicators[name] = self._parse_indicator(name, config)

    def _parse_indicator(self, name: str, config: Dict) -> IndicatorConfig:
        """Parse a single indicator configuration."""
        # Parse params
        params = {}
        for param_name, param_config in config.get("params", {}).items():
            if isinstance(param_config, dict) and "range" in param_config:
                params[param_name] = ParamConfig.from_dict(param_name, param_config)

        # Parse entry/exit signals
        entry = SignalConfig.from_dict(config.get("entry", {}), "entry")
        exit_signal = SignalConfig.from_dict(config.get("exit", {}), "exit")

        return IndicatorConfig(
            name=name,
            full_name=config.get("name", name),
            category=config.get("category", ""),
            talib_function=config.get("talib_function", ""),
            params=params,
            outputs=config.get("outputs", []),
            entry=entry,
            exit=exit_signal,
            sma_period=config.get("sma_period"),
            dual_ma=config.get("dual_ma"),
            usage=config.get("usage"),
            requires=config.get("requires", [])
        )

    def get_indicator(self, name: str) -> IndicatorConfig:
        """Get configuration for a specific indicator."""
        if name not in self._indicators:
            raise ValueError(f"Unknown indicator: {name}")
        return self._indicators[name]

    def get_all_indicators(self) -> Dict[str, IndicatorConfig]:
        """Get all indicator configurations."""
        return self._indicators.copy()

    def get_indicators_by_category(self, category: str) -> Dict[str, IndicatorConfig]:
        """Get indicators filtered by category."""
        return {
            name: config
            for name, config in self._indicators.items()
            if config.category == category
        }

    def get_operator_symbol(self, operator_name: str) -> str:
        """Get the symbol for an operator name."""
        return self._operators.get(operator_name, operator_name)

    def list_indicator_names(self) -> List[str]:
        """List all available indicator names."""
        return list(self._indicators.keys())

    def list_categories(self) -> List[str]:
        """List all available categories."""
        return list(set(c.category for c in self._indicators.values()))
