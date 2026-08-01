# mth5-panel

- **Current Version**: 0.0.3
- **Home URL**: [IAGA-DVI-DataStandards/mth5-panel: Panel application to build and view MTH5 files](https://github.com/IAGA-DVI-DataStandards/mth5-panel)

Includes `panel` applications for building and viewing a MTH5 file.

`mth5-panel` provides interactive user interfaces built with `panel` for common
MTH5 workflows. The applications help you:

- Create MTH5 files from supported input data.
- Inspect MTH5 contents in a browser-based interface.
- Explore time-series data and metadata without writing custom plotting code.

## Installation

Install from source with either `pip` or `uv`.

Standalone executables are published on GitHub Releases for users who do not want to install Python. Those builds will be available at [Releases](https://github.com/IAGA-DVI-DataStandards/mth5-panel/releases).

### pip

```bash
pip install mth5-panel
```

### uv

```bash
uv pip install mth5-panel
```

### Development Mode

```bash
git clone https://github.com/IAGA-DVI-DataStandards/mth5-panel.git
pip install -e .
```

This project is not currently available through `conda` channels.

## Contributing

Contributions are welcome.

### Raise an issue

1. Search existing issues to avoid duplicates.
2. Open a new issue: https://github.com/IAGA-DVI-DataStandards/mth5-panel/issues/new
3. Review existing issues: https://github.com/IAGA-DVI-DataStandards/mth5-panel/issues
4. Use a clear title, expected behavior, observed behavior, and reproducible steps.

### Create a pull request

1. Fork the repository and create a feature branch.
2. Make focused changes with tests where possible.
3. Run tests locally before submitting.
4. Open a pull request: https://github.com/IAGA-DVI-DataStandards/mth5-panel/pulls
5. Link the related issue and explain what changed and why.

## Install Package

```bash
pip install mth5-panel
```

## Serve

You can run the Panel apps in multiple ways.  First install the package:



### 1) Script

Install the package, then run the app launcher:

```bash
mth5-panel-app --show
```

This launches the unified MTH5 create/view application through the installable script.

### 2) Command line

Use `panel serve` directly against an app script, best if `mth5-panel` is installed into a local directory using `pip intall -e .`:

```bash
panel serve mth5_panel/mth5_panel_app.py --show
```

### 3) Jupyter notebook

In a notebook, enable the Panel extension and render a view inline:

```python
import panel as pn

pn.extension()
```

Then import or build a Panel object from one of the app modules and display it in a notebook cell.

### 4) VS Code

1. Open this repository in VS Code.
2. Open a terminal in the project root.
3. Run one of the launcher commands, for example:

```bash
panel serve mth5_panel/mth5_panel_app.py --autoreload --show
```

VS Code will show the local server URL in the terminal output; open it in your browser to use the application.



