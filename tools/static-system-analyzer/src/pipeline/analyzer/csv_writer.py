"""
CSV writing utilities for topic entries.
"""

import csv
from pathlib import Path
from typing import List

from .models import TopicEntry, expand_pubsub_entries
from common.logger import log_info, log_error

CSV_HEADERS = ["folder", "topic_name", "role"]


def write_topic_csv(
    output_path: Path,
    entries: List[TopicEntry],
    expand_pubsub: bool = True
) -> bool:
    """
    Write topic entries to a CSV file.

    Args:
        output_path: Path for the output CSV file
        entries: List of TopicEntry objects
        expand_pubsub: If True, expand pubsub entries into pub and sub

    Returns:
        True if successful, False otherwise
    """
    try:
        # Expand pubsub entries if requested
        if expand_pubsub:
            entries = expand_pubsub_entries(entries)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write data rows (no header)
            for entry in entries:
                writer.writerow([
                    entry.source_folder,
                    entry.name,
                    entry.role
                ])

        log_info(f"Successfully wrote {len(entries)} entries to {output_path}")
        return True

    except IOError as e:
        log_error(f"Error writing CSV file {output_path}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error writing CSV: {e}")
        return False


def write_topic_csv_simple(
    output_path: Path,
    entries: List[TopicEntry]
) -> bool:
    """
    Write topic entries to CSV without header row.

    Args:
        output_path: Path for the output CSV file
        entries: List of TopicEntry objects (already expanded)

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(f"{entry.to_csv_row()}\n")

        log_info(f"Successfully wrote {len(entries)} entries to {output_path}")
        return True

    except IOError as e:
        log_error(f"Error writing CSV file {output_path}: {e}")
        return False
