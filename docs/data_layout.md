# Data layout

No data ship with this repository. This page lists what the pipeline expects, so you can
arrange your own recordings (or adapt the path templates in `config.yaml`).

## Raw data

Paths are templates in `paths:`; the defaults expect one folder per subject and session:

```
<raw_root>/
└── <sub>/
    └── session_<ses>/
        ├── nirs/<sub>_s<ses>.snirf        continuous-wave fNIRS/HD-DOT, 2 wavelengths
        ├── lsl/<sub>_s<ses>.xdf           LSL recording with the marker and sync streams
        ├── ratings/<sub>_s<ses>.csv       PsychoPy output with the interest ratings
        └── events_override.csv            optional, see below
```

`config/sessions.csv` lists the sessions to process (`subject,session[,include]`). Subject
codes are used as folder names; pick codes that identify nobody.

### SNIRF

- One recording per file, amplitude data in volts, probe geometry with landmarks
  (Nz, Iz, LPA, RPA, Cz) so the montage can be registered to the atlas head.
- An aux channel (`lsl.nirs_sync_aux`, default `digital`) that records the hardware sync
  pulses also sent to LSL.
- Two wavelengths per channel. The co-located-source correction of the original study
  (sources emitting at a third wavelength) is not part of this release.

### XDF

- A marker stream (`lsl.marker_stream`) with string markers.
- A stream (`lsl.sync_stream`) whose timestamps are the LSL times of the same hardware pulses.

Markers are mapped onto the fNIRS clock by matching the two pulse trains on their
inter-pulse intervals and fitting a line through the matched pairs (offset + clock drift).

### Event markers

| marker | meaning |
|---|---|
| `video_<i>` | clip *i* of the session, *i* = 1 … `n_video_blocks × videos_per_block` |
| `slider_<i>` | rating screen after clip *i* |
| `post_rating_baseline_<i>` | fixation after the rating of clip *i* (ends a video block) |
| `rest_<b>` | rest block *b* |
| `end` | end of the session; data after it are discarded |

Clip *i* belongs to video block `(i − 1) // videos_per_block + 1`. Other markers (instructions,
questionnaires, other tasks) are ignored.

### `events_override.csv`

Columns `onset, duration, value, trial_type`, onsets in seconds on the fNIRS clock. If the file
exists it replaces XDF synchronisation. Use it for sessions that need manual repair (e.g.
recordings split by an interruption): build the table once, check it, and keep it next to the
raw data so the repair is reproducible.

### PsychoPy ratings

One row per presented clip with:

| column (config key) | default | content |
|---|---|---|
| `rating_column` | `circular_slider.response` | the rating |
| `position_column` | `video_block.thisTrialN` | 0-based position within the video block |
| `block_columns` | `main.thisN`, `main.thisRepN` | 0-based index of the video block (outer loop) |

Unrated clips keep their slot; block membership is never inferred from rated rows only.

### Videos (optional, stimulus features)

```
<videos>/session_<s>/block_<b>/video_<n>.mp4      n = position within the block
```

Every participant must have seen the same clip at the same session/block/position; then
`video_uid = (s − 1) × videos_per_session + (b − 1) × videos_per_block + n` identifies a clip
across participants.

## Derivatives

Everything the pipeline writes goes under `paths.derivatives`:

```
<derivatives>/
├── forward_model/forward_model.pkl             step 1
├── sub-<sub>/ses-<ses>/
│   ├── events.csv                              step 2
│   ├── parcel_sensitivity.csv                  step 2
│   ├── preproc.pkl                             step 2 (large; can be deleted after step 3)
│   ├── qc.json                                 steps 2-3
│   ├── parcels/<block>.nc, global_<block>.nc   step 3 (netCDF, HbO/HbR per parcel, µM)
│   ├── glm_auc.csv.gz, glm_trials.csv          step 4
└── group/
    ├── ratings/          step 0
    ├── features/         step 5
    ├── stats/            step 6
    ├── decoding/         step 7
    └── video_features/   video steps
```

Per-session derivatives (time series) are participant data; keep them where your data
governance allows. The group tables in `group/` are trial-level features and labels.
