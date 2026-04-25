#!/usr/bin/env python3
from pathlib import Path
import pandas as pd


# close-loop with reactive agents
PARQUET_PATHS = [
    "/home/ryan/nuplan/exp/simulation/closed_loop_reactive_agents/test14-hard/planMAE/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2026.04.24.11.59.53.parquet",
    "/home/ryan/nuplan/exp/simulation/closed_loop_reactive_agents/test14-hard/planTF/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2026.04.21.20.37.52.parquet",
]

# close-loop with non-reactive agents
PARQUET_PATHS = [
    "/home/ryan/nuplan/exp/simulation/closed_loop_nonreactive_agents/test14-hard/planMAE/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2026.04.24.11.59.53.parquet",
    "/home/ryan/nuplan/exp/simulation/closed_loop_nonreactive_agents/test14-hard/planTF/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2026.04.21.19.49.05.parquet",
]

# open-loop metrics
PARQUET_PATHS = [
    "/home/ryan/nuplan/exp/simulation/open_loop_boxes/test14-hard/planMAE/aggregator_metric/open_loop_boxes_weighted_average_metrics_2026.04.24.11.59.53.parquet",
    "/home/ryan/nuplan/exp/simulation/open_loop_boxes/test14-hard/planTF/aggregator_metric/open_loop_boxes_weighted_average_metrics_2026.04.21.21.41.00.parquet",
]


def infer_meta_from_path(path: str) -> dict:
    parts = Path(path).parts
    meta = {
        "source_path": path,
        "filename": Path(path).name,
        "challenge": "",
        "split": "",
        "planner": "",
    }

    if "simulation" in parts:
        idx = parts.index("simulation")
        if len(parts) > idx + 3:
            meta["challenge"] = parts[idx + 1]
            meta["split"] = parts[idx + 2]
            meta["planner"] = parts[idx + 3]

    return meta


def infer_output_excel_path(parquet_paths: list[str]) -> Path:
    first_path = Path(parquet_paths[0])
    parts = first_path.parts

    if "simulation" not in parts:
        raise ValueError(f"Cannot infer output path: {first_path}")

    idx = parts.index("simulation")
    simulation_root = Path(*parts[: idx + 1])
    challenge = parts[idx + 1]
    split = parts[idx + 2]

    return simulation_root / challenge / f"{split}_metrics_summary.xlsx"


def make_safe_sheet_name(name: str) -> str:
    for ch in ["\\", "/", "*", "?", ":", "[", "]"]:
        name = name.replace(ch, "_")
    return name[:31]


def make_sheet_name(path: str, run_id: int) -> str:
    meta = infer_meta_from_path(path)

    # 文件名末尾一般是时间戳：
    # closed_loop_nonreactive_agents_weighted_average_metrics_2026.04.21.10.40.23.parquet
    timestamp = Path(path).stem.split("metrics_")[-1]

    name = f"{run_id}_{meta['planner']}_{timestamp}"
    return make_safe_sheet_name(name)


def read_one_parquet(path: str, run_id: int) -> pd.DataFrame:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path_obj)
    meta = infer_meta_from_path(path)

    print("=" * 120)
    print(f"Run ID: {run_id}")
    print(f"File: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    if len(df) > 0:
        print("\nLast row, usually score:")
        print(df.tail(1).to_string(index=False))
    else:
        print("\nEmpty table, no score row.")

    print("=" * 120)
    print()

    df = df.copy()

    # 加元信息，方便在每个 sheet 里追踪来源
    df.insert(0, "run_id", run_id)
    df.insert(1, "planner", meta["planner"])
    df.insert(2, "split", meta["split"])
    df.insert(3, "challenge", meta["challenge"])
    df.insert(4, "filename", meta["filename"])
    df.insert(5, "source_path", meta["source_path"])

    return df


def main():
    output_path = infer_output_excel_path(PARQUET_PATHS)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_dfs = []

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for run_id, path in enumerate(PARQUET_PATHS):
            df = read_one_parquet(path, run_id)

            sheet_name = make_sheet_name(path, run_id)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            if len(df) > 0:
                score_dfs.append(df.tail(1).copy())

        if score_dfs:
            score_df = pd.concat(score_dfs, ignore_index=True)
            score_df.to_excel(writer, sheet_name="score_rows", index=False)

    print("\nSaved Excel:")
    print(output_path.resolve())

    if score_dfs:
        print("\nAll score rows:")
        print(score_df.to_string(index=False))


if __name__ == "__main__":
    main()