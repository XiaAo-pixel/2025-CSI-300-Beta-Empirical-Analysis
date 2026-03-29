import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")
RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(exist_ok=True)


def load_index_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df = df.loc[:, ["交易日期_TrdDt", "指数日收益率_IdxDRet"]].rename(
        columns={"交易日期_TrdDt": "date", "指数日收益率_IdxDRet": "index_return"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date")
    df["index_return_lag"] = df["index_return"].shift(1)
    return df


def load_stock_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df["date"] = pd.to_datetime(df["日期_Date"])
    if "日收益率_Dret" in df.columns:
        df = df[["date", "日收益率_Dret"]].rename(
            columns={"日收益率_Dret": "stock_return"}
        )
    else:
        # 如果没有直接提供收益率，则用收盘价计算日收益率
        close_col = None
        for candidate in ["收盘价(元)_Clpr", "收盘价(元)_ClPr", "收盘价_ClPr", "收盘价"]:
            if candidate in df.columns:
                close_col = candidate
                break
        if close_col is None:
            raise KeyError(f"无法在文件中找到收益率列或收盘价列: {path.name}")
        df = df[["date", close_col]].rename(columns={close_col: "close_price"})
        df = df.sort_values("date").drop_duplicates(subset="date")
        df["stock_return"] = df["close_price"].pct_change()
        df = df[["date", "stock_return"]]

    df = df.sort_values("date").drop_duplicates(subset="date")
    return df


def align_returns(stock_df: pd.DataFrame, index_df: pd.DataFrame, method: str = "contemporaneous") -> pd.DataFrame:
    if method == "contemporaneous":
        index_col = "index_return"
    elif method == "lagged":
        index_col = "index_return_lag"
    else:
        raise ValueError("method must be 'contemporaneous' or 'lagged'")

    merged = stock_df.merge(index_df[["date", index_col]], on="date", how="inner")
    merged = merged.rename(columns={index_col: "index_return"})
    merged = merged.dropna(subset=["stock_return", "index_return"])
    return merged


def estimate_beta(df: pd.DataFrame) -> dict[str, float] | None:
    if len(df) < 10:
        return None
    x = sm.add_constant(df["index_return"])
    y = df["stock_return"]
    model = sm.OLS(y, x)
    result = model.fit()
    return {
        "alpha": float(result.params["const"]),
        "beta": float(result.params["index_return"]),
        "r2": float(result.rsquared),
        "nobs": int(result.nobs),
    }


def compute_monthly_betas(stock_df: pd.DataFrame, index_df: pd.DataFrame, method: str = "contemporaneous") -> pd.DataFrame:
    df = align_returns(stock_df, index_df, method)
    if df.empty:
        return pd.DataFrame()
    df["year_month"] = df["date"].dt.to_period("M")
    records = []
    for period, group in df.groupby("year_month"):
        stats = estimate_beta(group)
        if stats is None:
            continue
        stats.update({"year_month": period.to_timestamp(), "observations": len(group)})
        records.append(stats)
    return pd.DataFrame(records)


def compute_stock_summary(stock_path: Path, index_df: pd.DataFrame, method: str = "contemporaneous") -> pd.DataFrame:
    stock_df = load_stock_data(stock_path)
    merged = align_returns(stock_df, index_df, method)
    summary = estimate_beta(merged)
    if summary is None:
        return pd.DataFrame()
    summary.update({"stock": stock_path.stem, "method": method, "observations": len(merged)})
    monthly = compute_monthly_betas(stock_df, index_df, method)
    if not monthly.empty:
        monthly["stock"] = stock_path.stem
        monthly["method"] = method
    return pd.DataFrame([summary]), monthly


def process_all_stocks(index_df: pd.DataFrame, method: str = "contemporaneous") -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = DATA_ROOT / "2025沪深300成分股日收益率"
    summaries = []
    monthly_records = []
    for xlsx in sorted(folder.glob("*.xlsx")):
        try:
            summary_df, monthly_df = compute_stock_summary(xlsx, index_df, method)
            if not summary_df.empty:
                summaries.append(summary_df)
            if not monthly_df.empty:
                monthly_records.append(monthly_df)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", xlsx.name, exc)
    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    monthly_all = pd.concat(monthly_records, ignore_index=True) if monthly_records else pd.DataFrame()
    return summary_all, monthly_all


def plot_histogram(df: pd.DataFrame, column: str, filename: Path, title: str) -> None:
    plt.figure(figsize=(9, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_monthly_beta(df: pd.DataFrame, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    summary = df.groupby(["year_month", "method"])["beta"].median().reset_index()
    for method, group in summary.groupby("method"):
        ax.plot(group["year_month"], group["beta"], marker="o", label=method)
    ax.set_title("2025 年沪深300成分股月度 Beta 中位数")
    ax.set_xlabel("月份")
    ax.set_ylabel("Beta")
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close()


def save_results(summary: pd.DataFrame, monthly: pd.DataFrame, method: str) -> None:
    summary_file = RESULTS_ROOT / f"beta_summary_{method}.csv"
    monthly_file = RESULTS_ROOT / f"beta_monthly_{method}.csv"
    summary.to_csv(summary_file, index=False)
    monthly.to_csv(monthly_file, index=False)
    plot_histogram(summary, "beta", RESULTS_ROOT / f"beta_distribution_{method}.png", f"{method} Beta 分布")
    if not monthly.empty:
        plot_monthly_beta(monthly, RESULTS_ROOT / f"beta_monthly_trend_{method}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="2025 沪深300 Beta 计算与分析")
    parser.add_argument("--method", choices=["contemporaneous", "lagged"], default="contemporaneous")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    index_df = load_index_data(DATA_ROOT / "2025沪深300指数日收益率.xlsx")
    summary, monthly = process_all_stocks(index_df, method=args.method)

    if summary.empty:
        logger.error("没有找到有效的 Beta 结果。请检查数据格式。")
        return

    logger.info("保存 %s Beta 汇总结果，共 %d 只股票", args.method, len(summary))
    save_results(summary, monthly, args.method)
    print("结果已保存到 results/ 目录。请打开 analysis_report.ipynb 查看更多分析。")


if __name__ == "__main__":
    main()
