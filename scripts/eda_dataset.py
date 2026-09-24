from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "dataset" / "train"
TEST_DIR = ROOT / "data" / "dataset" / "test"

OUTPUT_DIR = ROOT / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def get_file_info(path):
    size_gb = path.stat().st_size / (1024 ** 3)
    return {
        "file": path.name,
        "size_gb": round(size_gb, 3)
    }


def analyze_source_file(path):
    """
    Analyze a source TSV without loading the entire file.
    """

    print(f"\n{'=' * 70}")
    print(f"Analyzing: {path}")
    print(f"{'=' * 70}")

    total_rows = 0

    missing = Counter()
    empty = Counter()

    country_counts = Counter()

    unique_ids = set()
    duplicate_ids = 0

    name_lengths = []
    address_lengths = []

    name_counter = Counter()
    address_counter = Counter()

    first_chunk = None

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False
    ):

        if first_chunk is None:
            first_chunk = chunk.copy()

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Missing / empty
        # ----------------------------------------------------

        for col in chunk.columns:

            empty_count = (chunk[col].str.strip() == "").sum()

            empty[col] += int(empty_count)

        # ----------------------------------------------------
        # Country
        # ----------------------------------------------------

        if "country" in chunk.columns:
            country_counts.update(
                chunk["country"].str.strip().str.upper()
            )

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        if "entity_id" in chunk.columns:

            ids = chunk["entity_id"]

            for entity_id in ids:
                if entity_id in unique_ids:
                    duplicate_ids += 1
                else:
                    unique_ids.add(entity_id)

        # ----------------------------------------------------
        # Name statistics
        # ----------------------------------------------------

        if "business_name" in chunk.columns:

            names = chunk["business_name"].fillna("")

            name_lengths.extend(
                names.str.len().tolist()
            )

            name_counter.update(
                names.str.strip().str.lower()
            )

        # ----------------------------------------------------
        # Address statistics
        # ----------------------------------------------------

        if "business_address" in chunk.columns:

            addresses = chunk["business_address"].fillna("")

            address_lengths.extend(
                addresses.str.len().tolist()
            )

            address_counter.update(
                addresses.str.strip().str.lower()
            )

        print(
            f"\rRows processed: {total_rows:,}",
            end=""
        )

    print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    result = {
        "file": path.name,
        "rows": total_rows,
        "columns": list(first_chunk.columns),
        "duplicate_entity_ids": duplicate_ids,

        "name_length_mean": (
            np.mean(name_lengths)
            if name_lengths else None
        ),

        "name_length_median": (
            np.median(name_lengths)
            if name_lengths else None
        ),

        "address_length_mean": (
            np.mean(address_lengths)
            if address_lengths else None
        ),

        "address_length_median": (
            np.median(address_lengths)
            if address_lengths else None
        ),

        "unique_names": len(name_counter),
        "unique_addresses": len(address_counter),

        "countries": dict(country_counts),
        "empty_values": dict(empty),

        "top_names": name_counter.most_common(20),
        "top_addresses": address_counter.most_common(20),

        "sample": first_chunk.head(10)
    }

    return result


# ============================================================
# GROUND TRUTH
# ============================================================

def analyze_ground_truth(path):

    print(f"\n{'=' * 70}")
    print("Analyzing ground truth")
    print(f"{'=' * 70}")

    total_rows = 0
    empty_matches = 0

    match_count_distribution = Counter()

    total_matches = 0

    matched_entity_counter = Counter()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False
    ):

        total_rows += len(chunk)

        match_column = "matched_entity_ids"

        for value in chunk[match_column]:

            value = value.strip()

            if not value:
                empty_matches += 1
                match_count_distribution[0] += 1
                continue

            matches = [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]

            count = len(matches)

            match_count_distribution[count] += 1

            total_matches += count

            matched_entity_counter.update(matches)

        print(
            f"\rRows processed: {total_rows:,}",
            end=""
        )

    print()

    return {
        "total_s1_entities": total_rows,
        "zero_match_entities": empty_matches,
        "total_match_links": total_matches,
        "match_count_distribution":
            dict(sorted(match_count_distribution.items())),
        "unique_matched_entity_ids":
            len(matched_entity_counter)
    }


# ============================================================
# SAVE REPORT
# ============================================================

def write_source_report(results, split):

    output = OUTPUT_DIR / f"{split}_source_statistics.md"

    with open(output, "w", encoding="utf-8") as f:

        f.write(f"# {split.upper()} Source EDA\n\n")

        for result in results:

            f.write(f"## {result['file']}\n\n")

            f.write(
                f"- Rows: **{result['rows']:,}**\n"
            )

            f.write(
                f"- Duplicate entity IDs: "
                f"**{result['duplicate_entity_ids']:,}**\n"
            )

            f.write(
                f"- Unique business names: "
                f"**{result['unique_names']:,}**\n"
            )

            f.write(
                f"- Unique business addresses: "
                f"**{result['unique_addresses']:,}**\n"
            )

            f.write(
                f"- Mean name length: "
                f"**{result['name_length_mean']:.2f}**\n"
            )

            f.write(
                f"- Median name length: "
                f"**{result['name_length_median']:.2f}**\n"
            )

            f.write(
                f"- Mean address length: "
                f"**{result['address_length_mean']:.2f}**\n"
            )

            f.write(
                f"- Median address length: "
                f"**{result['address_length_median']:.2f}**\n"
            )

            f.write("\n### Country distribution\n\n")

            for country, count in sorted(
                result["countries"].items(),
                key=lambda x: -x[1]
            ):

                f.write(
                    f"- `{country}`: {count:,}\n"
                )

            f.write("\n### Empty values\n\n")

            for col, count in result["empty_values"].items():

                f.write(
                    f"- `{col}`: {count:,}\n"
                )

            f.write("\n### Top repeated business names\n\n")

            for name, count in result["top_names"]:

                if name:
                    f.write(
                        f"- `{name}`: {count:,}\n"
                    )

            f.write("\n### Sample\n\n")

            f.write(
                result["sample"].to_markdown(index=False)
            )

            f.write("\n\n---\n\n")


def write_ground_truth_report(result):

    output = OUTPUT_DIR / "ground_truth_statistics.md"

    with open(output, "w", encoding="utf-8") as f:

        f.write("# Ground Truth EDA\n\n")

        f.write(
            f"- Total S1 entities: "
            f"**{result['total_s1_entities']:,}**\n"
        )

        f.write(
            f"- Zero-match entities: "
            f"**{result['zero_match_entities']:,}**\n"
        )

        f.write(
            f"- Total match links: "
            f"**{result['total_match_links']:,}**\n"
        )

        f.write(
            f"- Unique matched entity IDs: "
            f"**{result['unique_matched_entity_ids']:,}**\n"
        )

        f.write("\n## Match-count distribution\n\n")

        for count, entities in result[
            "match_count_distribution"
        ].items():

            f.write(
                f"- {count} matches: "
                f"**{entities:,} S1 entities**\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("AMAZON ML CHALLENGE — DATASET EDA")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN SOURCES
    # --------------------------------------------------------

    train_files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
    ]

    train_results = []

    for path in train_files:

        if not path.exists():
            print(f"\nWARNING: Missing {path}")
            continue

        train_results.append(
            analyze_source_file(path)
        )

    write_source_report(
        train_results,
        "train"
    )

    # --------------------------------------------------------
    # TEST SOURCES
    # --------------------------------------------------------

    test_files = [
        TEST_DIR / "test_source1.tsv",
        TEST_DIR / "test_source2.tsv",
        TEST_DIR / "test_source3.tsv",
    ]

    test_results = []

    for path in test_files:

        if not path.exists():
            print(f"\nWARNING: Missing {path}")
            continue

        test_results.append(
            analyze_source_file(path)
        )

    write_source_report(
        test_results,
        "test"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    ground_truth = TRAIN_DIR / "train_ground_truth.tsv"

    if ground_truth.exists():

        result = analyze_ground_truth(
            ground_truth
        )

        write_ground_truth_report(result)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(f"\nReports written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()