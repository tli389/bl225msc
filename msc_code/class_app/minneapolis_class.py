"""Fit-once projection adapter for the public Minneapolis CLASS model.

The Federal Reserve Bank of Minneapolis publishes the R program, estimation
data, fitted coefficients and example outputs used by its 2020 public stress
test tool:

https://www.minneapolisfed.org/banking/financial-studies-and-community-banking/covid-19-stress-test-tool

This module does not re-estimate or alter that model. It loads the archive's
fitted coefficients and 2019Q4 FR Y-9C state once, then evaluates alternative
eight-variable macroeconomic paths. This is useful for comparing path banks:
every generator is passed through the same fixed satellite. The implementation
was independently transcribed from the published model equations and is tested
against the archive's supplied ``u_extreme`` projection.

The archive has no explicit software licence. Keep it as an external research
input and consult the Minneapolis Fed terms before redistributing its contents.
"""

from __future__ import annotations

import argparse
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL_MODEL_URL = (
    "https://www.minneapolisfed.org/banking/financial-studies-and-community-"
    "banking/covid-19-stress-test-tool"
)

RAW_MACRO_COLUMNS = (
    "date",
    "growth",
    "cpr_ur",
    "cpr_t3m",
    "cpr_t10y",
    "bbb_corp",
    "cppi",
    "hpi",
    "djtsm",
)

MODEL_NAMES = (
    "nim",
    "nintrat",
    "tradrat",
    "afs_return",
    "nie_comp",
    "nie_fass",
    "nie_allother",
    "nco_firstlien",
    "nco_jrlien",
    "nco_heloc",
    "nco_const",
    "nco_multi",
    "nco_nfnr",
    "nco_othre",
    "nco_ci",
    "nco_cc",
    "nco_othcons",
    "nco_agr",
    "nco_lease",
    "nco_oth",
)

LOAN_CATEGORIES = (
    "agr",
    "cc",
    "ci",
    "const",
    "firstlien",
    "heloc",
    "jrlien",
    "lease",
    "multi",
    "nfnr",
    "oth",
    "othcons",
    "othre",
)

BANKS = pd.DataFrame(
    {
        "entity": (
            1037003,
            1039502,
            1068025,
            1068191,
            1069778,
            1070345,
            1073757,
            1074156,
            1111435,
            1119794,
            1120754,
            1199611,
            1275216,
            1562859,
            1951350,
            2162966,
            2277860,
            2380443,
            3242838,
            3587146,
            3846375,
        ),
        "tkr": (
            "MT",
            "JPM",
            "KEY",
            "HBAN",
            "PNC",
            "FITB",
            "BAC",
            "TFC",
            "STT",
            "USB",
            "WFC",
            "NTRS",
            "AXP",
            "ALLY",
            "C",
            "MS",
            "COF",
            "GS",
            "RF",
            "BK",
            "DFS",
        ),
    }
)

GSIB_SURCHARGE = {
    1039502: 2.5,
    1951350: 2.0,
    1073757: 1.5,
    1120754: 1.5,
    2380443: 1.5,
    1111435: 1.0,
    2162966: 1.0,
    3587146: 1.0,
}

GLOBAL_MARKET_SHOCK_2020 = {
    1039502: 21_800_000.0,
    1951350: 6_000_000.0,
    2380443: 18_400_000.0,
    2162966: 10_100_000.0,
    1073757: 10_500_000.0,
    1120754: 900_000.0,
    1111435: 600_000.0,
    3587146: 800_000.0,
}

OPERATIONAL_RISK_2020 = 0.874301107 * 144.0 * 1_000_000.0


@dataclass(frozen=True)
class ClassProjection:
    """Outputs for one macroeconomic scenario.

    ``aggregate``, ``firm`` and ``breaches`` follow the official CSV schemas.
    ``detail`` retains projected income, charge-off and provision components so
    downstream work can report the mechanism behind capital changes.
    """

    aggregate: pd.DataFrame
    firm: pd.DataFrame
    breaches: pd.DataFrame
    detail: pd.DataFrame


@dataclass(frozen=True)
class ClassBatchProjection:
    """Stacked outputs for a named collection of macroeconomic paths."""

    aggregate: pd.DataFrame
    firm: pd.DataFrame
    breaches: pd.DataFrame
    detail: pd.DataFrame


def _date_to_quarter_number(values: pd.Series) -> pd.Series:
    dates = pd.to_numeric(values, errors="raise").astype(np.int64)
    years = dates // 10_000
    months = (dates // 100) % 100
    if not months.isin((3, 6, 9, 12)).all():
        bad = dates.loc[~months.isin((3, 6, 9, 12))].tolist()
        raise ValueError(f"Dates must be quarter ends in YYYYMMDD form; got {bad[:5]}")
    return years * 4 + months // 3


def prepare_macro_path(
    macro: pd.DataFrame,
    *,
    date_q0: int = 20191231,
    future_horizon: int = 13,
) -> pd.DataFrame:
    """Validate raw CLASS inputs and construct its macro regressors.

    The input must contain the start quarter, at least its previous three
    quarters, and ``future_horizon`` consecutive future quarters. The leading
    history is needed for four-quarter HPI and commercial-property growth.
    """

    missing = sorted(set(RAW_MACRO_COLUMNS).difference(macro.columns))
    if missing:
        raise ValueError(f"Macro path is missing required columns: {missing}")

    out = macro.loc[:, RAW_MACRO_COLUMNS].copy()
    out["date"] = pd.to_numeric(out["date"], errors="raise").astype(np.int64)
    if out["date"].duplicated().any():
        raise ValueError("Macro path contains duplicate quarter dates")
    out = out.sort_values("date").reset_index(drop=True)
    out["quarter_number"] = _date_to_quarter_number(out["date"])
    if not np.all(np.diff(out["quarter_number"].to_numpy()) == 1):
        raise ValueError("Macro path dates must form one consecutive quarterly sequence")

    q0_number = int(_date_to_quarter_number(pd.Series([date_q0])).iloc[0])
    required = set(range(q0_number - 3, q0_number + future_horizon + 1))
    available = set(out["quarter_number"].astype(int))
    absent = sorted(required.difference(available))
    if absent:
        raise ValueError(
            "Macro path must include the three quarters before q0, q0, and "
            f"{future_horizon} future quarters; missing quarter indices {absent}"
        )

    numeric = [name for name in RAW_MACRO_COLUMNS if name != "date"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    if out[numeric].isna().any().any():
        cols = out[numeric].columns[out[numeric].isna().any()].tolist()
        raise ValueError(f"Macro path has missing/non-numeric inputs in: {cols}")
    if (out[["hpi", "cppi", "djtsm"]] <= 0).any().any():
        raise ValueError("HPI, commercial-property and stock index levels must be positive")

    out["spread"] = out["cpr_t10y"] - out["cpr_t3m"]
    out["d10y"] = out["cpr_t10y"].diff()
    out["stockgrowth"] = np.log(out["djtsm"] / out["djtsm"].shift(1)) * 100.0
    out["unchange"] = out["cpr_ur"].diff() * 4.0
    out["yyhpi"] = np.log(out["hpi"] / out["hpi"].shift(4)) * 100.0
    out["yyhpi_zero"] = np.where(out["yyhpi"] <= 0.0, out["yyhpi"], 0.0)
    out["yycppi"] = np.log(out["cppi"] / out["cppi"].shift(4)) * 100.0
    out["yycppi_zero"] = np.where(out["yycppi"] <= 0.0, out["yycppi"], 0.0)
    out["bspread"] = out["bbb_corp"] - out["cpr_t10y"]
    out["dbspread"] = out["bspread"].diff()
    out["dbspread_pos"] = np.where(out["dbspread"] >= 0.0, out["dbspread"], 0.0)

    keep = out.loc[
        out["quarter_number"].between(q0_number, q0_number + future_horizon)
    ].copy()
    derived = (
        "spread",
        "d10y",
        "stockgrowth",
        "unchange",
        "yyhpi",
        "yyhpi_zero",
        "yycppi",
        "yycppi_zero",
        "bspread",
        "dbspread",
        "dbspread_pos",
    )
    if keep.loc[keep["quarter_number"] > q0_number, list(derived)].isna().any().any():
        raise ValueError("Macro transformations are incomplete over the projection horizon")
    return keep.set_index("quarter_number", drop=False)


class MinneapolisClassProjector:
    """Frozen 2019Q4 Minneapolis CLASS satellite for one or many macro paths."""

    def __init__(
        self,
        coefficients: pd.DataFrame,
        y9c: pd.DataFrame,
        *,
        date_q0: int = 20191231,
        include_special_losses: bool = True,
        exclude_dividends: bool = False,
        tax_rate: float = 0.21,
        reserve_trueup_horizon: int = 8,
        projection_horizon: int = 13,
        report_horizon: int = 9,
    ) -> None:
        self.date_q0 = int(date_q0)
        self.include_special_losses = bool(include_special_losses)
        self.exclude_dividends = bool(exclude_dividends)
        self.tax_rate = float(tax_rate)
        self.reserve_trueup_horizon = int(reserve_trueup_horizon)
        self.projection_horizon = int(projection_horizon)
        self.report_horizon = int(report_horizon)
        if self.projection_horizon < self.report_horizon + 4:
            raise ValueError("Projection horizon must cover report horizon plus four quarters")
        self._coefficients = self._validate_coefficients(coefficients)
        self._q0 = self._prepare_start_state(y9c)

    @classmethod
    def from_files(
        cls,
        *,
        coefficients_path: str | Path,
        y9c_path: str | Path,
        **kwargs: object,
    ) -> MinneapolisClassProjector:
        """Load the two official archive inputs required by the frozen model."""

        coefficients = pd.read_csv(Path(coefficients_path))
        y9c = pd.read_csv(Path(y9c_path))
        return cls(coefficients, y9c, **kwargs)

    @staticmethod
    def _validate_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
        required = {"mdl", "vars", "coef"}
        missing = sorted(required.difference(coefficients.columns))
        if missing:
            raise ValueError(f"Coefficient file is missing columns: {missing}")
        out = coefficients.copy()
        out["coef"] = pd.to_numeric(out["coef"], errors="raise")
        absent = sorted(set(MODEL_NAMES).difference(out["mdl"].unique()))
        if absent:
            raise ValueError(f"Coefficient file is missing equations: {absent}")
        duplicated = out.duplicated(["mdl", "vars"])
        if duplicated.any():
            raise ValueError("Coefficient file has duplicate equation-variable rows")
        return out

    def _prepare_start_state(self, y9c: pd.DataFrame) -> pd.DataFrame:
        required = {
            "entity",
            "dt",
            "cl_assets",
            "cl_intearn_ass",
            "cl_afssec",
            "qprefdividend",
            "qcommdividend",
            "unsafe_ratio",
            "cet1",
            "rwa",
        }
        missing = sorted(required.difference(y9c.columns))
        if missing:
            raise ValueError(f"Y-9C file is missing columns: {missing}")

        data = y9c.copy()
        data["date"] = pd.to_numeric(data["dt"], errors="raise").astype(np.int64)
        dates = data["date"]
        months = (dates // 100) % 100
        data["ddt"] = dates // 10_000 + months / 12.0
        data["time"] = data["ddt"] - data["ddt"].min()

        real = data["entity"] != 9_999_999
        industry = ~real
        for mask in (real, industry):
            idx = data.index[mask]
            grouped = data.loc[idx].groupby("date", sort=False)
            data.loc[idx, "total_assets"] = grouped["cl_assets"].transform("sum").to_numpy()
            data.loc[idx, "total_intearn_assets"] = grouped[
                "cl_intearn_ass"
            ].transform("sum").to_numpy()
            data.loc[idx, "total_afssec"] = grouped["cl_afssec"].transform("sum").to_numpy()
        data["shr_assets"] = 100.0 * data["cl_assets"] / data["total_assets"]
        data["shr_intearn_assets"] = (
            100.0 * data["cl_intearn_ass"] / data["total_intearn_assets"]
        )
        data["shr_afssec"] = 100.0 * data["cl_afssec"] / data["total_afssec"]

        q0 = data.loc[(data["date"] == self.date_q0) & data["entity"].isin(BANKS.entity)].copy()
        q0 = BANKS.merge(q0, on="entity", how="left", validate="one_to_one")
        if q0["date"].isna().any():
            absent = q0.loc[q0["date"].isna(), "entity"].astype(int).tolist()
            raise ValueError(f"Y-9C file has no q0 observation for entities: {absent}")
        q0["date"] = q0["date"].astype(np.int64)
        q0["cl_dividend"] = (
            0.0 if self.exclude_dividends else q0["qprefdividend"] + q0["qcommdividend"]
        )
        if self.include_special_losses:
            q0["loss_ops"] = (
                q0["cl_assets"] / q0["cl_assets"].sum() * OPERATIONAL_RISK_2020 / 9.0
            )
            q0["loss_gms"] = q0["entity"].map(GLOBAL_MARKET_SHOCK_2020).fillna(0.0)
        else:
            q0["loss_ops"] = 0.0
            q0["loss_gms"] = 0.0
        q0["charge"] = q0["entity"].map(GSIB_SURCHARGE).fillna(0.0)
        q0["req"] = np.where(q0["charge"] == 0.0, 0.045, 0.045 + q0["charge"] / 100.0)
        return q0

    def project_path(self, macro_path: pd.DataFrame) -> ClassProjection:
        """Project one raw eight-variable macro path through the frozen model."""

        macro = prepare_macro_path(
            macro_path,
            date_q0=self.date_q0,
            future_horizon=self.projection_horizon,
        )
        q0_number = int(_date_to_quarter_number(pd.Series([self.date_q0])).iloc[0])
        state = self._q0.copy()
        state["quarter_number"] = q0_number
        all_rows = [state.copy()]

        for step in range(1, self.projection_horizon + 1):
            future = state.copy()
            future["quarter_number"] = q0_number + step
            macro_row = macro.loc[q0_number + step]
            for name, value in macro_row.items():
                if name not in {"date", "quarter_number"}:
                    future[name] = value
            future["constant"] = 1.0
            future["unsafe_dbspread"] = future["unsafe_ratio"] * future["dbspread"]
            future["unsafe_dbspread_pos"] = np.where(
                future["dbspread"] > 0.0,
                future["unsafe_dbspread"],
                0.0,
            )

            for model in MODEL_NAMES:
                lag_name = f"cl_{model}_lag"
                target = f"cl_{model}"
                future[lag_name] = future[target]
                beta = self._coefficients.loc[
                    self._coefficients["mdl"] == model, ["vars", "coef"]
                ]
                missing = sorted(set(beta["vars"]).difference(future.columns))
                if missing:
                    raise ValueError(f"Path/state lacks regressors for {model}: {missing}")
                future[target] = (
                    future[beta["vars"].tolist()].to_numpy(dtype=float)
                    @ beta["coef"].to_numpy(dtype=float)
                )
                future.drop(columns=lag_name, inplace=True)

            future["qint_inc_net"] = future["cl_nim"] / 400.0 * future["cl_intearn_ass"]
            future["qnonint_inc"] = future["cl_nintrat"] / 400.0 * future["cl_assets"]
            future["qtradrev_inc"] = future["cl_tradrat"] / 400.0 * future["cl_trading_ass"]
            future["qnonint_exp_comp"] = future["cl_nie_comp"] / 400.0 * future["cl_assets"]
            future["qnonint_exp_fass"] = future["cl_nie_fass"] / 400.0 * future["cl_assets"]
            future["qnonint_exp_allother"] = (
                future["cl_nie_allother"] / 400.0 * future["cl_assets"]
            )
            future["qafssecur_inc"] = future["cl_afs_return"] / 400.0 * future["cl_afssec"]
            future["qhtmsecur_inc"] = 0.0
            for category in LOAN_CATEGORIES:
                future[f"qnetxoff_{category}"] = (
                    future[f"cl_nco_{category}"] / 400.0 * future[f"ln_{category}"]
                )
            all_rows.append(future.copy())
            state = future

        projected = pd.concat(all_rows, ignore_index=True)
        return self._capital_projection(projected, macro, q0_number)

    def project_paths(
        self,
        paths: Mapping[Hashable, pd.DataFrame],
    ) -> ClassBatchProjection:
        """Project a named path bank while retaining a ``path_id`` key."""

        if not paths:
            raise ValueError("At least one macro path is required")
        outputs = {path_id: self.project_path(path) for path_id, path in paths.items()}

        def stack(attribute: str) -> pd.DataFrame:
            frames = []
            for path_id, result in outputs.items():
                frame = getattr(result, attribute).copy()
                frame.insert(0, "path_id", path_id)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True)

        return ClassBatchProjection(
            aggregate=stack("aggregate"),
            firm=stack("firm"),
            breaches=stack("breaches"),
            detail=stack("detail"),
        )

    def _capital_projection(
        self,
        projected: pd.DataFrame,
        macro: pd.DataFrame,
        q0_number: int,
    ) -> ClassProjection:
        work = projected.sort_values(["entity", "quarter_number"]).reset_index(drop=True)
        work["loss_gms"] = np.where(
            work["quarter_number"] == q0_number + 1,
            work["loss_gms"],
            0.0,
        )
        work["extra_items"] = 0.0

        # Preserve the category pairing in the published implementation so the
        # numerical adapter remains a strict reproduction of its reserve rule.
        reserve_source = {"heloc": "jrlien", "jrlien": "heloc"}
        for category in LOAN_CATEGORIES:
            source = reserve_source.get(category, category)
            work[f"q4_{category}"] = work.groupby("entity", sort=False)[
                f"qnetxoff_{source}"
            ].transform(self._forward_four_quarter_sum)
        work["q4_total"] = work[[f"q4_{name}" for name in LOAN_CATEGORIES]].sum(axis=1)
        work["qnetxoff_total"] = work[
            [f"qnetxoff_{name}" for name in LOAN_CATEGORIES]
        ].sum(axis=1)

        start = work.loc[
            work["quarter_number"] == q0_number, ["entity", "llres", "q4_total"]
        ].copy()
        start["trueup"] = start["q4_total"] - start["llres"]
        work = work.merge(start[["entity", "trueup"]], on="entity", how="left")
        work = work.sort_values(["entity", "quarter_number"]).reset_index(drop=True)
        work["stepN"] = np.maximum(
            1.0
            - (work["quarter_number"] - q0_number) / self.reserve_trueup_horizon,
            0.0,
        )
        work["b_lower"] = work["q4_total"] - work["trueup"] * work["stepN"]
        work["b_upper"] = 2.5 * work["b_lower"]
        work["llresX"] = np.nan
        for _, indices in work.groupby("entity", sort=False).groups.items():
            idx = list(indices)
            reserve: list[float] = []
            for position, row_index in enumerate(idx):
                row = work.loc[row_index]
                if position == 0:
                    value = float(row["llres"])
                else:
                    adjustment = (
                        float(row["trueup"]) / self.reserve_trueup_horizon
                        if position <= self.reserve_trueup_horizon
                        else 0.0
                    )
                    value = reserve[-1] + adjustment
                    value = min(value, float(row["b_upper"]))
                    value = max(value, float(row["b_lower"]))
                reserve.append(value)
            work.loc[idx, "llresX"] = reserve

        work["qprov"] = (
            work["qnetxoff_total"]
            + work["llresX"]
            - work.groupby("entity", sort=False)["llresX"].shift(1)
        )
        work["cl_dividend"] = work["cl_dividend"].clip(lower=0.0)
        work.loc[work["quarter_number"] == q0_number, "cl_dividend"] = 0.0
        work["ppnr"] = (
            work["qint_inc_net"]
            + work["qnonint_inc"]
            + work["qtradrev_inc"]
            - work["qnonint_exp_comp"]
            - work["qnonint_exp_fass"]
            - work["qnonint_exp_allother"]
        )
        work["pretax"] = (
            work["ppnr"]
            + work["qafssecur_inc"]
            + work["qhtmsecur_inc"]
            - work["qprov"]
            - work["loss_ops"]
            - work["loss_gms"]
        )
        work["taxes"] = work["pretax"] * self.tax_rate
        work["netinc"] = (
            work["pretax"] - work["taxes"] + work["qminor_int"] + work["extra_items"]
        ).fillna(0.0)
        work["netincX"] = work.groupby("entity", sort=False)["netinc"].cumsum()
        work["dividendX"] = work.groupby("entity", sort=False)["cl_dividend"].cumsum()
        work["taxesX"] = np.where(work["taxes"] > 0.0, 0.0, work["taxes"])
        work.loc[work["quarter_number"] == q0_number, "taxesX"] = 0.0
        work["taxesX2"] = work.groupby("entity", sort=False)["taxesX"].cumsum()
        work["cet1X"] = work["cet1"] + work["netincX"] - work["dividendX"] + work["taxesX2"]
        work["req_cet1"] = work["req"] * work["rwa"]

        report = work.loc[
            work["quarter_number"] <= q0_number + self.report_horizon
        ].copy()
        date_map = macro.set_index("quarter_number")["date"].astype(np.int64)
        report["date"] = report["quarter_number"].map(date_map)
        report["cet1_out"] = report["cet1X"] / 1_000_000.0
        report["rwa_out"] = report["rwa"] / 1_000_000.0
        report["avg_assets_out"] = report["avg_assets"] / 1_000_000.0
        report["breach"] = (report["cet1X"] < report["req_cet1"]).astype(int)
        q0_cet1 = report.loc[
            report["quarter_number"] == q0_number, ["entity", "cet1_out"]
        ].rename(columns={"cet1_out": "cet1_q0"})
        report = report.merge(q0_cet1, on="entity", how="left")
        report["cap_ratio"] = report["cet1_out"] / report["rwa_out"] * 100.0
        report["lev_ratio"] = report["cet1_out"] / report["avg_assets_out"] * 100.0
        report["cet1_drop"] = report["cet1_out"] - report["cet1_q0"]
        report = report.merge(BANKS, on="entity", how="left", suffixes=("", "_bank"))
        if "tkr_bank" in report:
            report["tkr"] = report["tkr"].fillna(report["tkr_bank"])

        firm = pd.DataFrame(
            {
                "date": report["date"].astype(np.int64),
                "entity": report["entity"].astype(np.int64),
                "tkr": report["tkr"],
                "cet1": report["cet1_out"],
                "rwa": report["rwa_out"],
                "avg_assets": report["avg_assets_out"],
                "cap_ratio": report["cap_ratio"],
                "lev_ratio": report["lev_ratio"],
                "cet1_drop": report["cet1_drop"],
                "breach": report["breach"].astype(int),
            }
        ).sort_values(["entity", "date"], ignore_index=True)

        breaches = (
            firm.groupby(["entity", "tkr"], as_index=False, sort=False)["breach"]
            .max()
            .sort_values("entity", ignore_index=True)
        )
        aggregate = (
            firm.groupby("date", as_index=False, sort=True)[["cet1", "rwa", "avg_assets"]]
            .sum()
            .sort_values("date", ignore_index=True)
        )
        aggregate["cap_ratio"] = aggregate["cet1"] / aggregate["rwa"] * 100.0
        aggregate["lev_ratio"] = aggregate["cet1"] / aggregate["avg_assets"] * 100.0
        aggregate["cet1_drop"] = aggregate["cet1"] - aggregate["cet1"].iloc[0]

        detail_columns = [
            "entity",
            "tkr",
            "quarter_number",
            "qnetxoff_total",
            "qprov",
            "ppnr",
            "pretax",
            "netinc",
            "cet1X",
            "loss_ops",
            "loss_gms",
        ]
        detail = work.merge(BANKS, on="entity", how="left", suffixes=("", "_bank"))
        if "tkr_bank" in detail:
            detail["tkr"] = detail["tkr"].fillna(detail["tkr_bank"])
        detail["date"] = detail["quarter_number"].map(date_map)
        detail = detail[["date", *detail_columns]].sort_values(
            ["entity", "quarter_number"], ignore_index=True
        )
        return ClassProjection(
            aggregate=aggregate,
            firm=firm,
            breaches=breaches,
            detail=detail,
        )

    @staticmethod
    def _forward_four_quarter_sum(values: pd.Series) -> pd.Series:
        return values.shift(-1).rolling(4, min_periods=4).sum().shift(-3).fillna(0.0)


def _read_macro_file(path: Path, scenario: str | None) -> pd.DataFrame:
    macro = pd.read_csv(path)
    if "type" in macro.columns:
        if not scenario:
            choices = sorted(macro["type"].dropna().astype(str).unique())
            raise ValueError(f"--scenario is required for this macro file; choose from {choices}")
        macro = macro.loc[macro["type"].astype(str) == scenario].copy()
        if macro.empty:
            raise ValueError(f"Scenario {scenario!r} is not present in {path}")
    elif scenario:
        raise ValueError("--scenario was supplied but the macro file has no 'type' column")
    return macro


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path, required=True)
    parser.add_argument("--y9c", type=Path, required=True)
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--scenario")
    parser.add_argument("--official-aggregate", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--exclude-special-losses",
        action="store_true",
        help="Remove the fixed DFAST 2020 operational-risk and market-shock overlays.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    projector = MinneapolisClassProjector.from_files(
        coefficients_path=args.coefficients,
        y9c_path=args.y9c,
        include_special_losses=not args.exclude_special_losses,
    )
    macro = _read_macro_file(args.macro, args.scenario)
    result = projector.project_path(macro)

    minimum = result.aggregate.loc[result.aggregate["cap_ratio"].idxmin()]
    print(f"minimum aggregate CET1 ratio: {minimum['cap_ratio']:.6f}%")
    print(f"minimum-ratio date: {int(minimum['date'])}")
    print(f"minimum aggregate CET1 change: {result.aggregate['cet1_drop'].min():.6f} bn")
    print(f"firms breaching threshold: {int(result.breaches['breach'].sum())}")

    if args.official_aggregate:
        expected = pd.read_csv(args.official_aggregate)
        columns = ["cet1", "rwa", "avg_assets", "cap_ratio", "lev_ratio", "cet1_drop"]
        if len(expected) != len(result.aggregate):
            raise AssertionError(
                f"Official output has {len(expected)} rows; projection has {len(result.aggregate)}"
            )
        difference = np.max(
            np.abs(
                expected[columns].to_numpy(dtype=float)
                - result.aggregate[columns].to_numpy(dtype=float)
            )
        )
        print(f"maximum absolute aggregate fixture difference: {difference:.3e}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result.aggregate.to_csv(args.output_dir / "aggregate_projections.csv", index=False)
        result.firm.to_csv(args.output_dir / "firm_projections.csv", index=False)
        result.breaches.to_csv(args.output_dir / "firm_breaches.csv", index=False)
        result.detail.to_csv(args.output_dir / "projection_detail.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
