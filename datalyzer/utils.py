import pandas as pd
import numpy as np
from rich.console import Console
console = Console()

def load_dataset(file_path, max_rows=100000):
	try:
		if file_path.endswith('.csv'):
			df = pd.read_csv(file_path, nrows=max_rows)
		elif file_path.endswith('.xlsx'):
			df = pd.read_excel(file_path, nrows=max_rows)
		else:
			raise ValueError("Unsupported file format! Use CSV or XLSX.")
		console.print(f"[bold cyan]Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns[/bold cyan]")
		numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
		cat_cols = df.select_dtypes(include=['object', 'category']).columns
		date_cols = df.select_dtypes(include=['datetime64']).columns
		console.print(f"[cyan]Numeric columns:[/cyan] {len(numeric_cols)}")
		console.print(f"[cyan]Categorical columns:[/cyan] {len(cat_cols)}")
		if len(date_cols) > 0:
			console.print(f"[cyan]Datetime columns:[/cyan] {len(date_cols)}")
		constant_cols = [col for col in df.columns if df[col].nunique() == 1]
		if constant_cols:
			console.print(f"[yellow]Warning: Constant columns detected: {constant_cols}[/yellow]")
		missing_cols = [col for col in df.columns if df[col].isna().sum() > 0]
		if missing_cols:
			console.print(f"[yellow]Warning: Columns with missing values: {missing_cols}[/yellow]")
		inconsistent_cols = [col for col in df.columns if df[col].apply(lambda x: type(x)).nunique() > 1]
		if inconsistent_cols:
			console.print(f"[yellow]Warning: Type inconsistencies detected: {inconsistent_cols}[/yellow]")
		return df
	except Exception as e:
		console.print(f"[red]Error loading dataset: {e}[/red]")
		return None

def detect_duplicates(df):
	dup_count = df.duplicated().sum()
	if dup_count > 0:
		console.print(f"[yellow]⚠ {dup_count} duplicate rows detected.[/yellow]")
	return dup_count

def suggest_imputations(df):
	impute_suggestions = {}
	for col in df.columns:
		if df[col].isna().sum() > 0:
			if df[col].dtype in ['float64', 'int64']:
				method = "median" if abs(df[col].skew()) > 1 else "mean"
			else:
				method = "mode"
			impute_suggestions[col] = method
	return impute_suggestions

def detect_mixed_types(df):
	mixed_cols = []
	for col in df.columns:
		types = df[col].apply(lambda x: type(x)).value_counts()
		if len(types) > 1:
			mixed_cols.append(col)
	if mixed_cols:
		console.print(f"[yellow]⚠ Mixed-type columns detected: {mixed_cols}[/yellow]")
	return mixed_cols

def auto_convert_types(df):
	for col in df.columns:
		if df[col].dtype == 'object':
			try:
				df[col] = pd.to_numeric(df[col])
				console.print(f"[green]Converted {col} to numeric[/green]")
			except Exception:
				continue
	return df

def detect_scaling_issues(df):
	scale_issues = {}
	for col in df.select_dtypes(include=['int64', 'float64']).columns:
		rng = df[col].max() - df[col].min()
		if rng > 1e6:
			scale_issues[col] = rng
	if scale_issues:
		console.print("[yellow]⚠ Features with large value ranges detected:[/yellow]")
		for col, rng in scale_issues.items():
			console.print(f"  {col}: range={rng:.2e}")
	return scale_issues

def flag_high_correlations(df, threshold=0.9):
	corr = df.corr().abs()
	upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
	high_corr = [(col, row, val) for col in upper.columns for row, val in upper[col].dropna().items() if val > threshold]
	if high_corr:
		console.print("[yellow]⚠ Highly correlated pairs (>0.9):[/yellow]")
		for a, b, v in high_corr:
			console.print(f"  {a} ↔ {b} ({v:.2f})")
	return high_corr
"""
Helpers, caching, type conversion, imputation, scaling, correlation, etc.
"""

# Functions: load_dataset, detect_duplicates, suggest_imputations, detect_mixed_types, auto_convert_types, detect_scaling_issues, flag_high_correlations
