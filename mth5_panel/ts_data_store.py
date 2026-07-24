from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import xarray as xr

from mth5 import CHANNEL_DTYPE, RUN_SUMMARY_DTYPE
from mth5.mth5 import MTH5

try:
    import dask.array as da  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    da = None


@dataclass(frozen=True)
class DataStoreConfig:
    time_chunk_size: int = 500_000
    enable_dask_chunks: bool = True


class MTDataStore:
    """Loads and caches MTH5 summaries and run datasets."""

    def __init__(self, config: DataStoreConfig | None = None):
        self.config = config or DataStoreConfig()
        self.channel_summary = pd.DataFrame(columns=CHANNEL_DTYPE.names)
        self.run_summary = pd.DataFrame(columns=RUN_SUMMARY_DTYPE.names)
        self._run_cache: Dict[Tuple[str, str, bool], xr.Dataset] = {}

    @staticmethod
    def _normalize_paths(file_paths: Iterable[str]) -> list[str]:
        return [str(Path(path)) for path in file_paths]

    def load_summaries(
        self, file_paths: Iterable[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        full_df_channels = pd.DataFrame()
        full_df_runs = pd.DataFrame()

        for file_path in self._normalize_paths(file_paths):
            with MTH5() as m:
                m = m.open_mth5(file_path, mode="r")

                run_df = m.run_summary
                run_df["hdf5_reference"] = run_df["run_hdf5_reference"].apply(
                    lambda ref: m.get_reference_path(ref)
                )
                run_df["file"] = file_path
                run_df.drop(columns=["station_hdf5_reference"], inplace=True)

                channel_df = m.channel_summary.to_dataframe()
                channel_df["hdf5_reference"] = channel_df["hdf5_reference"].apply(
                    lambda ref: m.get_reference_path(ref)
                )
                channel_df["file"] = file_path
                channel_df.drop(
                    columns=["run_hdf5_reference", "station_hdf5_reference"],
                    inplace=True,
                )

            full_df_channels = pd.concat([full_df_channels, channel_df])
            full_df_runs = pd.concat([full_df_runs, run_df])

        self.channel_summary = full_df_channels.reset_index(drop=True)
        self.run_summary = full_df_runs.reset_index(drop=True)
        return self.channel_summary, self.run_summary

    def _maybe_chunk_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        if not self.config.enable_dask_chunks or da is None:
            return dataset
        if "time" not in dataset.dims:
            return dataset
        return dataset.chunk({"time": self.config.time_chunk_size})

    def get_run_dataset(
        self, mth5_file: str, run_ref: str, calibrate: bool = True
    ) -> xr.Dataset:
        cache_key = (mth5_file, run_ref, bool(calibrate))
        cached = self._run_cache.get(cache_key)
        if cached is not None:
            return cached

        with MTH5() as m:
            m.open_mth5(mth5_file, mode="r")
            run = m.from_reference(run_ref)
            run_ts = run.to_runts()
            if calibrate:
                run_ts.calibrate()
            ds = self._maybe_chunk_dataset(run_ts.dataset)

        self._run_cache[cache_key] = ds
        return ds

    def load_selected_runs(
        self, selected_runs: Dict[str, list[str]], calibrate: bool = True
    ) -> Dict[str, xr.Dataset]:
        out: Dict[str, xr.Dataset] = {}
        for mth5_file, refs in selected_runs.items():
            for run_ref in refs:
                ds = self.get_run_dataset(mth5_file, run_ref, calibrate=calibrate)
                with MTH5() as m:
                    m.open_mth5(mth5_file, mode="r")
                    run = m.from_reference(run_ref)
                    run_key = (
                        f"{run.survey_metadata.id}."
                        f"{run.station_metadata.id}."
                        f"{run.metadata.id}"
                    )
                out[run_key] = ds
        return out

    def clear_cache(self):
        self._run_cache.clear()
