[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/mehmoodulhaq570/smart-datalyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/)
[![Issues](https://img.shields.io/github/issues/mehmoodulhaq570/smart-datalyzer)](https://github.com/mehmoodulhaq570/smart-datalyzer/issues)
[![Size](https://img.shields.io/github/repo-size/mehmoodulhaq570/smart-datalyzer.svg)](https://github.com/mehmoodulhaq570/smart-datalyzer)
[![Downloads](https://img.shields.io/github/downloads/mehmoodulhaq570/smart-datalyzer/total.svg)](https://github.com/mehmoodulhaq570/smart-datalyzer/releases)

# Smart Datalyzer

**Smart Datalyzer** is an intelligent, automated toolkit for comprehensive data analysis, visualization, and reporting. It provides ML readiness scoring, advanced statistical diagnostics, and publication-quality visualizations with minimal effort.

## 🚀 Key Features

### 📊 Data Quality & Profiling

- **Smart Dataset Loading**: Automatic detection of CSV/XLSX files with type inference
- **Duplicate Detection**: Identify and report duplicate rows
- **Mixed Type Detection**: Find columns with inconsistent data types
- **Auto Type Conversion**: Intelligent conversion of string columns to numeric
- **Missing Value Analysis**: Detection and imputation suggestions
- **Constant Column Detection**: Identify features with zero variance
- **Scaling Issue Detection**: Flag features with extreme value ranges

### 🎯 Target-Aware Analysis (Multiple Targets Support)

- **Target Leakage Detection**: Identify features that leak target information (>95% accuracy)
- **Class Imbalance Analysis**: Compute imbalance ratios and distribution statistics
- **Feature-Target Association**: Statistical tests (ANOVA, Kruskal-Wallis, Chi-square)
- **Sensitivity Analysis**: Permutation importance for feature ranking
- **Model Suggestion**: Automatic recommendation (Regression vs Classification)

### 📈 Statistical Diagnostics

- **Normality Testing**: Shapiro-Wilk, D'Agostino, Kolmogorov-Smirnov tests with QQ plots
- **Outlier Detection**: Z-score based detection with percentage reporting
- **Correlation Analysis**: Pearson, Spearman, Kendall correlation matrices
- **VIF Computation**: Variance Inflation Factor for multicollinearity detection
- **Mutual Information**: Feature importance via mutual information scores
- **Covariance Matrix**: Full covariance analysis with CSV export
- **High Correlation Flagging**: Automatic detection of correlated pairs (>0.9)

### 📉 Visualization Suite

- **Distribution Plots**: Histograms with KDE overlays
- **Box Plots**: Outlier visualization with quartile analysis
- **Violin Plots**: Distribution density visualization
- **Swarm Plots**: Individual data point overlay on boxplots
- **QQ Plots**: Quantile-quantile plots for normality assessment
- **Correlation Heatmaps**: Multiple correlation methods with annotations
- **Feature Importance Charts**: RandomForest-based importance ranking
- **PCA Variance Plots**: Principal component analysis visualization
- **t-SNE Scatter Plots**: 2D dimensionality reduction visualization

### 📝 Reporting & Export

- **Interactive HTML Reports**: Comprehensive analysis with embedded visualizations
- **JSON Export**: Machine-readable summary statistics
- **PDF Generation**: Publication-ready reports (optional)
- **Plot Export**: High-resolution PNG plots (300 DPI)
- **Caching System**: Smart caching for faster re-analysis

### 🤖 Smart Auto Mode

- Automatic feature engineering recommendations
- ML readiness scoring (0-100)
- Actionable improvement suggestions
- Complete pipeline execution with single flag

## 📦 Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/mehmoodulhaq570/smart-datalyzer.git
cd smart-datalyzer

# Install build tools
pip install build

# Build the package
python -m build

# Install
pip install dist/smart_datalyzer-0.1.1-py3-none-any.whl
```

### Development Install

```bash
pip install -e .
```

## 🎮 Usage

### Basic Usage (Single Target)

```bash
python -m smart-datalyzer data.xlsx "target_column"
```

Or using the installed command:

```bash
smart-datalyzer data.xlsx "target_column"
```

### Multiple Target Columns

```bash
python -m smart-datalyzer data.csv "target1" "target2" "target3"
```

### Command Line Arguments

```bash
python -m smart-datalyzer <file> <target> [OPTIONS]
# or
smart-datalyzer <file> <target> [OPTIONS]

Arguments:
  file                    Path to dataset (CSV or XLSX)
  target                  Target column name(s) - space separated for multiple

Options:
  --stats                 Run detailed statistical analysis
  --outliers             Detect and report outliers
  --leakage              Detect target leakage features
  --imbalance            Check class imbalance
  --plots                Generate all visualization plots
  --report               Generate interactive HTML/JSON report
  --auto                 Run full automatic analysis (recommended)
  --max_rows N           Limit rows to read (default: 100000)
  --output_dir DIR       Output directory (default: "reports")
```

### Examples

**Quick Analysis:**

```bash
python -m smart-datalyzer sales.xlsx "Revenue" --auto
```

**Detailed Statistical Report:**

```bash
smart-datalyzer customers.csv "Churn" --stats --plots --report
```

**Multiple Targets with Custom Output:**

```bash
smart-datalyzer experiment.xlsx "Outcome1" "Outcome2" --auto --output_dir results
```

**Outlier & Leakage Detection:**

```bash
python -m smart-datalyzer medical.csv "Disease" --outliers --leakage
```

## 📊 Output Structure

```
reports/
├── plots/
│   ├── *_distribution.png      # Distribution histograms
│   ├── *_boxplot.png           # Box plots
│   ├── *_violinplot.png        # Violin plots
│   ├── *_swarmplot.png         # Swarm plots
│   ├── *_qqplot.png            # QQ plots
│   ├── correlation_*.png       # Correlation heatmaps
│   ├── feature_importance.png  # Feature importance chart
│   ├── pca_variance.png        # PCA analysis
│   └── tsne_scatter.png        # t-SNE visualization
├── report.html                 # Interactive HTML report
├── summary.json                # JSON summary statistics
├── covariance_matrix.csv       # Covariance matrix
└── .cache/                     # Analysis cache
```

## 🧰 Python API Usage

```python
from datalyzer.utils import load_dataset
from datalyzer.stats import feature_statistics, detect_outliers
from datalyzer.plots import plot_distributions, plot_correlation

# Load data
df = load_dataset("data.csv")

# Get statistics
stats, readiness, suggestions = feature_statistics(df)
print(f"ML Readiness Score: {readiness}/100")

# Detect outliers
outliers = detect_outliers(df, df.select_dtypes(include=['float64', 'int64']).columns)

# Generate plots
plot_paths = plot_distributions(df, plots_dir="reports/plots")
correlation_paths = plot_correlation(df, plots_dir="reports/plots")
```

## 🔧 Dependencies

### Core Requirements

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Statistical functions
- `statsmodels` - Advanced statistics
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Plotting backend
- `seaborn` - Statistical visualizations
- `rich` - Terminal formatting

See `requirements.txt` for complete list.

## 🎨 Features in Detail

### ML Readiness Score

Smart Datalyzer computes an ML readiness score (0-100) based on:

- Missing value percentage
- Constant features
- Numeric vs categorical balance
- Duplicate rows
- Data quality issues

### Caching System

Automatically caches analysis results using SHA256 hashing for:

- Faster re-analysis of same datasets
- Incremental updates
- Reduced computation time

### Smart Type Inference

Automatically detects and suggests:

- Numeric columns stored as strings
- Categorical features with high cardinality
- Date/time columns
- Mixed-type columns

## 👨‍💻 Author

**Mehmood Ul Haq**  
Email: mehmoodulhaq1040@gmail.com  
GitHub: [@mehmoodulhaq570](https://github.com/mehmoodulhaq570)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

## 🔒 Security

For security issues, please see [SECURITY.md](SECURITY.md).

## 📝 Changelog

### v0.1.1 (Current)

- Fixed swarm plot performance issues with large datasets (added sampling limit of 2000 points)
- Fixed filename sanitization for plots with special characters
- Improved visualization generation speed
- Skip class imbalance check for targets with >10 unique values

### v0.1.0

- Initial release
- Multiple target column support
- Comprehensive statistical analysis
- Advanced visualization suite
- Smart auto-analysis mode
- Caching system
- Interactive HTML reports

## 🙏 Acknowledgments

Built with modern Python data science stack and best practices for automated data analysis.
