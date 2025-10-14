# Datalyzer

Datalyzer is a toolkit for data analysis, visualization, and automated reporting.

## Features

- Statistical diagnostics
- Outlier detection
- Feature importance
- Correlation analysis
- QQ plots, boxplots, violin plots, swarm plots
- Automated report generation

## Installation

Recommended (modern):

```bash
pip install build
python -m build
pip install dist/datalyzer-0.1.0-py3-none-any.whl
```

Or legacy:

```bash
pip install .
```

## Author

Mehmood Ul Haq (<mehmoodulhaq1040@gmail.com>)

## License

MIT License (see LICENSE)

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Security

See [SECURITY.md](SECURITY.md)

## Usage

```bash
python -m datalyzer.cli <input_file> <target_column>
```

## Requirements

See `requirements.txt` for dependencies.

## License

MIT
