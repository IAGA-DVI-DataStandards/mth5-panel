# mth5-panel

- **Current Version**: 0.0.2
- **Home URL**: [IAGA-DVI-DataStandards/mth5-panel: Panel application to build and view MTH5 files](https://github.com/IAGA-DVI-DataStandards/mth5-panel)

Includes `panel` applications for building and viewing a MTH5 file.

`mth5-panel` provides interactive user interfaces built with `panel` for common
MTH5 workflows. The applications help you:

- Create MTH5 files from supported input data.
- Inspect MTH5 contents in a browser-based interface.
- Explore time-series data and metadata without writing custom plotting code.

## Installation

Install from source with either `pip` or `uv`.

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

## Serve

You can run the Panel apps in multiple ways.

### 1) Command line

Use `panel serve` directly against an app script:

```bash
panel serve mth5_panel/mth5_viewer.py --show
```

You can also serve other app entry files in this package, for example:

```bash
panel serve mth5_panel/make_mth5.py --show
```

### 2) Jupyter notebook

In a notebook, enable the Panel extension and render a view inline:

```python
import panel as pn

pn.extension()
```

Then import or build a Panel object from one of the app modules and display it
in a notebook cell.

### 3) VS Code

1. Open this repository in VS Code.
2. Open a terminal in the project root.
3. Run a `panel serve` command, for example:

```bash
panel serve mth5_panel/mth5_viewer.py --autoreload --show
```

VS Code will show the local server URL in the terminal output; open it in your
browser to use the application.



