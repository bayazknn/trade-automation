"""
Data loading utilities for vectorbt optimizer.

Handles loading feather files from data/binance directory and whitelist parsing.
"""
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


class DataLoader:
    """Load OHLCV data from feather files."""

    def __init__(self, data_dir: Path, config_path: Optional[Path] = None):
        """
        Initialize DataLoader.

        Parameters
        ----------
        data_dir : Path
            Directory containing feather files (e.g., data/binance)
        config_path : Path, optional
            Path to config.json for loading whitelist
        """
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path) if config_path else None

    def load_feather(self, symbol: str, timeframe: str = "1h") -> pd.DataFrame:
        """
        Load feather file for a symbol.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol (e.g., "BTC", "ETH")
        timeframe : str
            Timeframe (default "1h")

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame with columns: date, open, high, low, close, volume

        Raises
        ------
        FileNotFoundError
            If feather file doesn't exist
        """
        # Normalize symbol (remove /USDT if present)
        symbol = symbol.replace("/USDT", "").replace("_USDT", "").upper()

        filename = f"{symbol}_USDT-{timeframe}.feather"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_feather(filepath)

        # Ensure standard column names
        df.columns = [c.lower() for c in df.columns]

        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
            df = df.drop(columns=["timestamp"])

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def list_available_symbols(self, timeframe: str = "1h") -> List[str]:
        """
        List all available symbols in data directory.

        Parameters
        ----------
        timeframe : str
            Timeframe to filter by (default "1h")

        Returns
        -------
        List[str]
            List of available symbols (e.g., ["BTC", "ETH", "SOL"])
        """
        pattern = f"*_USDT-{timeframe}.feather"
        files = list(self.data_dir.glob(pattern))

        symbols = []
        for f in files:
            # Extract symbol from filename: BTC_USDT-1h.feather -> BTC
            symbol = f.stem.replace(f"_USDT-{timeframe}", "")
            symbols.append(symbol)

        return sorted(symbols)

    def load_whitelist(self) -> List[str]:
        """
        Load pair whitelist from config.json.

        Returns
        -------
        List[str]
            List of symbols from whitelist (e.g., ["BTC", "ETH", "SOL"])

        Raises
        ------
        FileNotFoundError
            If config.json doesn't exist
        ValueError
            If pair_whitelist not found in config
        """
        if self.config_path is None:
            raise ValueError("config_path not provided to DataLoader")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = json.load(f)

        # pair_whitelist is in exchange section
        whitelist = config.get("exchange", {}).get("pair_whitelist", [])

        if not whitelist:
            raise ValueError("pair_whitelist not found in config.json")

        # Convert "BTC/USDT" -> "BTC"
        symbols = [pair.split("/")[0] for pair in whitelist]

        return symbols

    def resolve_symbols(self, symbols: str | List[str]) -> List[str]:
        """
        Resolve symbol specification to list of symbols.

        Parameters
        ----------
        symbols : str or List[str]
            - "all": All available symbols in data directory
            - "whitelist": Symbols from config.json pair_whitelist
            - List[str]: Explicit list of symbols

        Returns
        -------
        List[str]
            Resolved list of symbols
        """
        if isinstance(symbols, list):
            return symbols

        if symbols == "all":
            return self.list_available_symbols()

        if symbols == "whitelist":
            return self.load_whitelist()

        # Single symbol as string
        return [symbols]


def load_feather(symbol: str, data_dir: str = "data/binance") -> pd.DataFrame:
    """
    Convenience function to load feather file.

    Parameters
    ----------
    symbol : str
        Cryptocurrency symbol
    data_dir : str
        Data directory path

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame
    """
    loader = DataLoader(Path(data_dir))
    return loader.load_feather(symbol)


def list_available_symbols(data_dir: str = "data/binance") -> List[str]:
    """Convenience function to list available symbols."""
    loader = DataLoader(Path(data_dir))
    return loader.list_available_symbols()


def load_whitelist(config_path: str = "config.json") -> List[str]:
    """Convenience function to load whitelist from config."""
    loader = DataLoader(Path("."), Path(config_path))
    return loader.load_whitelist()
