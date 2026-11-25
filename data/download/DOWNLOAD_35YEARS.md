# NDVI 35-Year Download System

## Overview

Downloads 35 years of seasonal NDVI data (1990-2024) from Google Earth Engine for SSL4EO-L locations. Each location produces a GeoTIFF with 140 bands (4 seasons x 35 years).

## Script: `download_ndvi_35years.py`

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--project` | GCP project ID for Earth Engine |
| `--service-account` | Path to service account JSON key |
| `--locations` | Path to locations.json file |
| `--output` | Output directory for GeoTIFF files |
| `--workers` | Number of concurrent download threads (default: 40) |
| `--skip-first` | Skip first N locations (applied BEFORE limit) |
| `--limit` | Maximum locations to process (applied AFTER skip-first) |
| `--reverse` | Process locations in reverse order within the range |
| `--worker-id`, `--num-workers` | Round-robin partition of the selected range (worker N takes every `num-workers`-th item); progress goes to `_progress_worker{N}.json`. Must be used together |
| `--test N` | Test mode: download only N locations |
| `--log-file` | Redirect output to a log file (line-buffered, for background runs) |
| `--report-interval` | Print progress every N downloads (default: 1 = tqdm bar) |

Run `python data/download/download_ndvi_35years.py --help` for the full list.

### Range Calculation

The order of operations is:
1. Load all locations from JSON
2. Apply `--skip-first` to define starting position
3. Apply `--limit` to define range size
4. Filter out already-completed items (for resume capability)

This ensures ranges are calculated from the **full list**, not filtered by existing files.

**Example:**
```
--skip-first 50000 --limit 50000
```
Always processes locations [50,000 - 99,999] regardless of what files exist on disk.

## Parallel Downloads

Earth Engine applies per-project quotas, so large downloads go faster when split
into non-overlapping ranges, each run with its own GCP project and service
account. For example, five workers covering the 250,000 entries in
`data/locations.json`:

```bash
python data/download/download_ndvi_35years.py --project PROJECT_A --service-account keys/project_a.json \
    --locations data/locations.json --output OUTPUT_DIR --skip-first 0      --limit 50000
python data/download/download_ndvi_35years.py --project PROJECT_B --service-account keys/project_b.json \
    --locations data/locations.json --output OUTPUT_DIR --skip-first 50000  --limit 50000
# ... and so on for 100000, 150000 and 200000
```

The ranges come from the full location list (see "Range Calculation"), so the
workers never overlap and can share one output directory. The released 200,000
locations (`data/locations_200k.json`) are what remains of these 250,000 after
the completeness filter.

Alternatively, `--worker-id N --num-workers K` splits a range
round-robin across K processes, each with its own `_progress_worker{N}.json`.

## Service Accounts

Each worker needs a GCP service account registered for Earth Engine, with its
JSON key stored locally (e.g. under `keys/`, which is gitignored). Pass the
project ID with `--project` and the key with `--service-account`.

## Resume Capability

- Progress is saved to `_progress_35years.json` in the output directory (`_progress_worker{N}.json` with `--worker-id`)
- Already-downloaded files are skipped automatically
- Failed downloads are tracked and can be retried by re-running the script
