# Profiler Timeline Visualization

Per-chunk **timeline** figures for NCCL collectives from the profiler's raw `.out` dumps.
Each figure lays out a rank's send stream chunk-by-chunk, drawing the measured phases of
every chunk (post → data-ready → transmit → done, plus per-rail transmission and the
theoretical wire time) so you can see the pipeline behavior and where time goes.

This is visualization only — it reads raw profiler `.out` files and produces PNGs. It does
not fit or export model parameters (that lives in `realistic_model/`).

## Layout

```
profiler_timeline_visualization/
├── common/                    # shared, algorithm-/platform-independent building blocks (a package)
│   ├── parsing.py             #   pure text parsing of the .out format (no matplotlib)
│   ├── plotting.py            #   generic matplotlib timeline helpers
│   └── __init__.py
└── p5en/                      # one folder per platform
    ├── ring_allreduce_timeline.ipynb    # Ring AllReduce timeline
    └── *.png                            # generated figures
```

Future platforms get their own sibling folder next to `p5en/` (e.g. `p6-b200/`).

## Responsibility boundary: `common/` vs the notebooks

The split is deliberate — **generic parsing/plotting is shared, everything algorithm- or
platform-specific stays in the notebook.**

`common/` holds only what is the same across algorithms and platforms:

* `common.parsing` (no matplotlib dependency — reusable headless):
  * `pick` / `STAT` — select a per-chunk statistic from a per-iteration summary
    (`mean=.. med=.. p95=..`).
  * `split_test_cases`, `parse_msg_header`, `parse_info_fields`, `parse_nchunks`,
    `parse_t_total` — the file/test-case header parsing every `.out` shares.
  * `split_sections(tc, header_regex)` — split a test case on any 2-group section header.
  * `findall_metric(section, pattern)` — pull `(chunk_id, value)` pairs for one bracket-tag
    metric; the caller passes the tag pattern, so it works for any algorithm's tags.
  * `extract_mode` — the NCCL_TESTS_SPLIT_MASK / mode suffix from a filename.
* `common.plotting` (matplotlib):
  * `fmt_bytes`, `TimelineStyle` (fonts/colors/size, overridable), `draw_group_background`,
    `draw_chunk_label`, `draw_separator`, `finalize_timeline_axes`, `save_and_show`.

Each notebook keeps what is **specific to its algorithm/platform**:

* the section layout and topology of its `.out` (ring: send/recv/rail/ctrl + the ring
  successor `(rank+1)%n`; tree: reduce/broadcast ops + the tree root),
* the bracket-tag names it feeds to `findall_metric`,
* the bar-drawing body (ring: dual-rail lanes + T_wire; tree: 3 sub-rows + BW estimate),
* its chunk-size formula and per-platform NIC constants.

So `common` is mostly the **parsing skeleton** plus a few plot primitives; the visually rich
plot body is intentionally left in each notebook.

## Data flow (paths)

Inputs and outputs are separate directories:

* **`DATA_DIRS`** — per-mask map of where the raw `.out` files are read from (each
  NCCL_TESTS_SPLIT_MASK lives in its own model-notebook folder).
* **`OUT_DIR`** — where PNGs are written (the notebook's own folder, `'.'`).

Current wiring:

| Notebook | reads (`DATA_DIRS`) | writes (`OUT_DIR`) |
|---|---|---|
| `ring_allreduce_timeline.ipynb` | `0x0` → `../../../p5en-allreduce-0x0-notebook/`, `0x7` → `../../../p5en-allreduce-0x7-notebook/` (`nccl_allreduce_ring_inter_r-{0..3}-{mask}.out`; rank 0 required, 1–3 optional. The paired 0x0-vs-0x7 figure needs both masks present) | `.` (this folder) |

## How to add a new algorithm or platform

1. **New platform, existing algorithm** — copy the closest notebook into a new platform
   folder (e.g. `p6-b200/ring_allreduce_timeline.ipynb`), point `DATA_DIR` at that
   platform's `.out`, and adjust the per-platform constants (NIC bandwidth, rail count, …).
   The `common` imports stay as-is.
2. **New algorithm** — start from an existing notebook and change only the algorithm-specific
   parts: the bracket-tag patterns passed to `findall_metric`, the section split regexes, the
   topology, and the plot body. Reuse `common.parsing` for the header/test-case parsing and
   `common.plotting` for axes/labels/save.
3. Keep the boundary: if you find yourself writing the same parsing/axis/save code in two
   notebooks, lift it into `common/` instead.

Notebooks import `common` via a `sys.path` insert of the `profiler_timeline_visualization/`
directory (see the first cell), then `from common import parsing as P, plotting as PL`.

## `.out` format

The profiler emits per-chunk metrics as bracket-tagged lines with per-iteration summaries,
e.g. `[POST_TO_DR] chunk 3 posted → data ready: mean=.. med=.. p95=.. ms`. `common.parsing`
targets this format; each notebook passes its own tag names to `findall_metric`.
