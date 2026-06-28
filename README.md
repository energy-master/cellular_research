# cellular

Cellular-automata research. Each top-level folder is a **self-contained
experiment** — its own data sources, scripts, and notes — sharing the common
`brahma_cellular` CA toolkit and the `identdynamics` SDK.

## Experiments

| Folder | Description |
|--------|-------------|
| [`hp_games/`](hp_games/) | `hp_ca` — first CA run wiring `brahma_cellular` into the IDent Dynamics SDK; runs one CA pass over a single stream/saved file and records it to ident db. |

## Conventions

- **One folder per experiment.** New experiment → new top-level folder; keep it
  self-contained.
- **No data in git.** Audio (`*.wav`, …) and array dumps are `.gitignore`d;
  inputs are pulled from the IDent Dynamics stream or saved files at run time.
- **No secrets in git.** Keep API tokens out of committed code — prefer
  environment variables or a `.gitignore`d local config.

## Environment

Shared virtualenv at `~/dev/ident_games` with:

```bash
pip install "brahma_cellular[identdynamics] @ git+https://github.com/energy-master/brahma_cellular"
```
