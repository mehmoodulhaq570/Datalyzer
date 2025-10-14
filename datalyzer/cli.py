
import argparse
import hashlib
import pickle
import os
import time
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from sklearn.metrics import accuracy_score

# Import package modules
from datalyzer.utils import load_dataset, detect_duplicates, detect_mixed_types, auto_convert_types, suggest_imputations, detect_scaling_issues, flag_high_correlations
from datalyzer.stats import feature_statistics, detect_outliers, detect_target_leakage, detect_imbalance, run_statistical_diagnostics, feature_target_association, class_imbalance_ratio, sensitivity_analysis, check_normality, compute_vif, compute_mutual_info, compute_covariance_matrix, suggest_model
from datalyzer.plots import plot_distributions, plot_correlation, compute_feature_importance, dimensionality_estimation
from datalyzer.report import generate_interactive_report
from datalyzer.config import console

def main():
	parser = argparse.ArgumentParser(description="Enhanced Smart Dataset Analyzer v3")
	parser.add_argument("file", help="Path to dataset (CSV/XLSX)")
	parser.add_argument("target", help="Target column name")
	parser.add_argument("--stats", action="store_true", help="Run feature statistics")
	parser.add_argument("--outliers", action="store_true", help="Detect outliers")
	parser.add_argument("--leakage", action="store_true", help="Detect target leakage")
	parser.add_argument("--imbalance", action="store_true", help="Check imbalance")
	parser.add_argument("--plots", action="store_true", help="Generate plots")
	parser.add_argument("--report", action="store_true", help="Generate interactive HTML/PDF/JSON report")
	parser.add_argument("--max_rows", type=int, default=100000, help="Max rows to read for large datasets")
	parser.add_argument("--output_dir", default="reports", help="Directory to save reports")
	parser.add_argument("--auto", action="store_true", help="Run full automatic Smart Analysis mode")
	args = parser.parse_args()

	cache_dir = os.path.join(args.output_dir, '.cache')
	os.makedirs(cache_dir, exist_ok=True)

	def get_cache_key(file_path, args_dict):
		hasher = hashlib.sha256()
		with open(file_path, 'rb') as f:
			hasher.update(f.read())
		for k, v in sorted(args_dict.items()):
			hasher.update(str(k).encode())
			hasher.update(str(v).encode())
		return hasher.hexdigest()

	def load_cache(key):
		cache_path = os.path.join(cache_dir, f'{key}.pkl')
		if os.path.exists(cache_path):
			with open(cache_path, 'rb') as f:
				return pickle.load(f)
		return None

	def save_cache(key, data):
		cache_path = os.path.join(cache_dir, f'{key}.pkl')
		with open(cache_path, 'wb') as f:
			pickle.dump(data, f)

	cache_key = get_cache_key(args.file, vars(args))

	# Suppress sklearn regression/classification warning
	import warnings
	warnings.filterwarnings(
		"ignore",
		message="The number of unique classes is greater than 50% of the number of samples. `y` could represent a regression problem, not a classification problem.",
		category=UserWarning
	)

	start_time = time.time()
	console.rule("[bold blue]📦 Dataset Loading[bold blue]")
	df = load_dataset(args.file, args.max_rows)

	if not args.target or args.target not in df.columns:
		possible_targets = [c for c in df.columns if df[c].nunique() < 20 and df[c].dtype != 'float64']
		args.target = possible_targets[-1] if possible_targets else df.columns[-1]
		console.print(f"[yellow]⚠ No valid target provided. Automatically using: [bold]{args.target}[/bold][yellow]")

	if df is None:
		console.print("[red]❌ Dataset could not be loaded. Exiting.[/red]")
		return

	cat_cols = df.select_dtypes(include=['object', 'category']).columns
	if len(cat_cols) > 0:
		console.print(f"[yellow]⚠ {len(cat_cols)} categorical columns detected. Consider encoding before modeling.[yellow]")

	mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
	console.print(f"[cyan]💾 Memory usage:[/cyan] [bold]{mem_usage:.2f} MB[/bold]")

	console.rule("[bold green]🔍 Dataset Overview[bold green]")
	console.print(f"[bold]📂 File:[/bold] [blue]{args.file}[/blue]")
	console.print(f"[bold]📏 Shape:[/bold] [magenta]{df.shape[0]}[/magenta] rows × [magenta]{df.shape[1]}[/magenta] columns")

	numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

	if not any([args.stats, args.outliers, args.leakage, args.imbalance, args.plots, args.report, args.auto]):
		console.print("[bold yellow]⚠ No flags provided. Running all analyses by default...[bold yellow]")
		args.stats = args.outliers = args.leakage = args.imbalance = args.plots = args.report = True

	if args.auto or args.stats:
		console.rule("[bold magenta]🤖 Smart Auto Analysis[bold magenta]")
		detect_duplicates(df)
		detect_mixed_types(df)
		df = auto_convert_types(df)
		suggest_imputations(df)
		detect_scaling_issues(df)
		flag_high_correlations(df)
		high_corr = flag_high_correlations(df)
		if high_corr:
			to_drop = list({b for _, b, _ in high_corr})
			console.print(f"[cyan]💡 Suggested to drop correlated columns: [bold]{to_drop}[/bold][cyan]")
		console.print("[green]✅ Smart Data Quality Checks completed.[green]")

	cache_data = load_cache(cache_key)

	if cache_data:
		stats_df = cache_data.get('stats_df', None)
		readiness_score = cache_data.get('readiness_score', 0)
		suggestions = cache_data.get('suggestions', [])
		outliers = cache_data.get('outliers', {})
		leakage = cache_data.get('leakage', [])
		imbalance = cache_data.get('imbalance', {})
		diagnostics = cache_data.get('diagnostics', {})
		console.print('[green]Loaded cached analysis results.[green]')
	else:
		if args.stats:
			stats_df, readiness_score, suggestions = feature_statistics(df)
		else:
			stats_df, readiness_score, suggestions = None, 0, []

		outliers = detect_outliers(df, numeric_cols) if args.outliers else {}
		leakage = detect_target_leakage(df, args.target) if args.leakage else []
		imbalance = detect_imbalance(df, args.target) if args.imbalance else {}

		if args.auto or args.stats:
			diagnostics = run_statistical_diagnostics(df, args.target)
		else:
			diagnostics = {}

		save_cache(cache_key, {
			'stats_df': stats_df,
			'readiness_score': readiness_score,
			'suggestions': suggestions,
			'outliers': outliers,
			'leakage': leakage,
			'imbalance': imbalance,
			'diagnostics': diagnostics
		})

		if outliers:
			outlier_table = Table(title="Outlier Detection", show_header=True, header_style="bold magenta")
			outlier_table.add_column("Feature")
			outlier_table.add_column("Count", justify="right")
			outlier_table.add_column("Percent", justify="right")
			for col, info in outliers.items():
				outlier_table.add_row(col, str(info['count']), f"{info['percent']}%")
			console.print(outlier_table)

		if leakage:
			leakage_table = Table(title="Feature Leakage Scores", show_header=True, header_style="bold yellow")
			leakage_table.add_column("Feature")
			leakage_table.add_column("Score", justify="right")
			for col in leakage:
				score = None
				try:
					from sklearn.ensemble import RandomForestClassifier
					clf = RandomForestClassifier(n_estimators=50, random_state=42)
					clf.fit(df[[col]].fillna(0), df[df.columns[-1]])
					y = df[df.columns[-1]]
					y_pred = clf.predict(df[[col]].fillna(0))
					if pd.api.types.is_numeric_dtype(y) and y.nunique() > len(y) * 0.5:
						from sklearn.metrics import r2_score
						score = r2_score(y, y_pred)
					else:
						score = accuracy_score(y, y_pred)
				except Exception:
					score = 0
				leakage_table.add_row(col, f"{score:.2f}")
			console.print(leakage_table)

		if imbalance:
			dist_table = Table(title="Target Distribution", show_header=True, header_style="bold blue")
			dist_table.add_column("Class")
			dist_table.add_column("Percentage", justify="right")
			for cls, pct in imbalance.items():
				dist_table.add_row(str(cls), f"{pct*100:.2f}%")
			console.print(dist_table)

	if args.auto or args.stats:
		diagnostics = run_statistical_diagnostics(df, args.target)

	if args.stats:
		console.rule("[bold magenta]🧠 Advanced Statistical & Dimensional Analyses[bold magenta]")
		assoc_results = feature_target_association(df, args.target)
		if assoc_results:
			assoc_table = Table(title="Feature–Target Associations", show_header=True, header_style="bold cyan")
			assoc_table.add_column("Feature")
			assoc_table.add_column("Test")
			assoc_table.add_column("Stat", justify="right")
			assoc_table.add_column("p-value", justify="right")
			for feat, res in assoc_results.items():
				if isinstance(res, dict):
					assoc_table.add_row(
						str(feat), str(res.get('test', '')), str(res.get('stat', '')), str(res.get('p_value', ''))
					)
			console.print(assoc_table)

		pca_path, tsne_path = dimensionality_estimation(df, args.output_dir)
		dim_table = Table(title="Dimensionality Reduction Plots", show_header=True, header_style="bold yellow")
		dim_table.add_column("Method")
		dim_table.add_column("Plot Path")
		dim_table.add_row("PCA Variance", str(pca_path))
		dim_table.add_row("t-SNE Scatter", str(tsne_path))
		console.print(dim_table)

		imbalance_info = class_imbalance_ratio(df, args.target)
		if imbalance_info:
			cc_table = Table(title="Class Counts", show_header=True, header_style="bold blue")
			cc_table.add_column("Class")
			cc_table.add_column("Count", justify="right")
			for cls, cnt in imbalance_info['counts'].items():
				cc_table.add_row(str(cls), str(cnt))
			console.print(cc_table)
			ir_table = Table(title="Imbalance Ratio", show_header=True, header_style="bold red")
			ir_table.add_column("Imbalance Ratio (IR)", justify="right")
			ir_table.add_row(str(imbalance_info['imbalance_ratio']))
			console.print(ir_table)

	sensitivity = sensitivity_analysis(df, args.target)
	if sensitivity is not None and not sensitivity.empty:
			sens_table = Table(title="Top 5 Sensitive Features", show_header=True, header_style="bold magenta")
			sens_table.add_column("Feature")
			sens_table.add_column("Sensitivity", justify="right")
			for feat, val in list(sensitivity.items())[:5]:
				sens_table.add_row(str(feat), str(val))
			console.print(sens_table)

	if args.stats:
		console.rule("[bold cyan]📈 Statistical Diagnostics[bold cyan]")
		normality_results = check_normality(df)
		if normality_results:
			norm_table = Table(title="Normality Test Results", show_header=True, header_style="bold green")
			norm_table.add_column("Feature")
			norm_table.add_column("Skew", justify="right")
			norm_table.add_column("Kurtosis", justify="right")
			norm_table.add_column("Shapiro p", justify="right")
			norm_table.add_column("DAgostino p", justify="right")
			norm_table.add_column("KS p", justify="right")
			norm_table.add_column("p-value", justify="right")
			norm_table.add_column("Normal?", justify="center")
			for info in normality_results.values():
				if isinstance(info, dict):
					norm_table.add_row(
						str(info.get('feature', '')), str(info.get('skew', '')), str(info.get('kurtosis', '')),
						str(info.get('Shapiro_p', '')), str(info.get('DAgostino_p', '')), str(info.get('KS_p', '')),
						str(info.get('p_value', '')), "✅" if info.get('is_normal', False) else "❌"
					)
			console.print(norm_table)
			num_normal = sum(1 for v in normality_results.values() if v.get("is_normal"))
			console.print(f"[yellow]🟢 Features with normal distribution:[/yellow] [bold]{num_normal}[/bold]")

		vif_df = compute_vif(df)
		if not vif_df.empty:
			vif_table = Table(title="High VIF Features", show_header=True, header_style="bold cyan")
			vif_table.add_column("Feature")
			vif_table.add_column("VIF", justify="right")
			for _, row in vif_df.iterrows():
				vif_table.add_row(str(row['Feature']), str(row['VIF']))
			console.print(vif_table)

		mi_df = compute_mutual_info(df, args.target)
		if not mi_df.empty:
			mi_table = Table(title="Mutual Information (Top 5)", show_header=True, header_style="bold green")
			mi_table.add_column("Feature")
			mi_table.add_column("Mutual_Info", justify="right")
			for _, row in mi_df.head(5).iterrows():
				mi_table.add_row(str(row['Feature']), str(row['Mutual_Info']))
			console.print(mi_table)

		cov_path = compute_covariance_matrix(df, args.output_dir)
		if cov_path:
			console.print(f"[cyan]📊 Covariance matrix saved to:[/cyan] [bold]{cov_path}[/bold]")
		else:
			console.print("[red]❌ No numeric columns found for covariance computation.[red]")

	os.makedirs(args.output_dir, exist_ok=True)
	plots_dir = os.path.join(args.output_dir, "plots")
	os.makedirs(plots_dir, exist_ok=True)

	plot_paths = {}
	if args.plots:
		console.rule("[bold cyan]📊 Generating Visualizations[bold cyan]")
		with Progress() as progress:
			task = progress.add_task("Generating visualizations...", total=5)
			plot_paths = plot_distributions(df, plots_dir=plots_dir)
			progress.update(task, advance=1)
			corr_paths = plot_correlation(df, plots_dir=plots_dir)
			progress.update(task, advance=1)
			if isinstance(plot_paths, dict):
				plot_paths["correlation"] = corr_paths
			else:
				plot_paths = {"all": plot_paths, "correlation": corr_paths}
			importance_path, importance_data = compute_feature_importance(df, args.target, plots_dir)
			if importance_path:
				plot_paths["feature_importance"] = importance_path
			progress.update(task, advance=1)
			pca_path, tsne_path = dimensionality_estimation(df, args.output_dir)
			if pca_path:
				plot_paths["pca"] = pca_path
			if tsne_path:
				plot_paths["tsne"] = tsne_path
			progress.update(task, advance=1)
		if importance_path:
			console.rule("[bold cyan]🌲 Feature Importance[bold cyan]")
			console.print(f"[green]🌲 Feature importance plot saved to [bold]{importance_path}[/bold][green]")

	model_type = suggest_model(df, args.target)

	if args.report:
		console.rule("[bold cyan]🧾 Generating Report[bold cyan]")
		normality_dict = check_normality(df)
		normality_results = list(normality_dict.values())
		formatted_leakage = {}
		for col in leakage:
			try:
				from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
				y = df[df.columns[-1]]
				is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > len(y) * 0.5
				if is_regression:
					model = RandomForestRegressor(n_estimators=50, random_state=42)
					model.fit(df[[col]].fillna(0), y)
					y_pred = model.predict(df[[col]].fillna(0))
					from sklearn.metrics import r2_score
					acc = r2_score(y, y_pred)
				else:
					model = RandomForestClassifier(n_estimators=50, random_state=42)
					model.fit(df[[col]].fillna(0), y)
					y_pred = model.predict(df[[col]].fillna(0))
					acc = accuracy_score(y, y_pred)
			except Exception:
				acc = 0
			formatted_leakage[col] = acc
		import jinja2
		template = jinja2.Environment(loader=jinja2.FileSystemLoader('.')).get_template('template.html')
		html_file = os.path.join(args.output_dir, 'report.html')
		json_file = os.path.join(args.output_dir, 'summary.json')
		timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		with Progress() as progress:
			task = progress.add_task("Generating report...", total=1)
			generate_interactive_report(
				df=df,
				stats_df=stats_df,
				readiness_score=readiness_score,
				suggestions=suggestions,
				outliers=outliers,
				leakage=formatted_leakage,
				imbalance=imbalance,
				plots_dir=plots_dir,
				output_dir=args.output_dir,
				plot_paths=plot_paths,
				normality=normality_results,
				template=template,
				html_file=html_file,
				json_file=json_file,
				timestamp=timestamp
			)
			progress.update(task, advance=1)

	console.rule("[bold blue]📘 SUMMARY[bold blue]")
	console.print(f"[bold]📊 Rows:[/bold] [magenta]{df.shape[0]}[/magenta], [bold]Columns:[/bold] [magenta]{df.shape[1]}[/magenta]")
	console.print(f"[bold]🤖 ML Readiness Score:[/bold] [green]{readiness_score}/100[/green]")
	console.print(f"[bold]⚠️ Outliers detected:[/bold] [red]{sum(o['count'] for o in outliers.values()) if outliers else 0}[/red]")
	console.print(f"[bold]💡 Suggested Model Type:[/bold] [cyan]{model_type}[/cyan]")
	console.rule("[bold green]✅ Analysis Complete[bold green]")
	runtime = time.time() - start_time
	console.print(f"[green]⏱ Total runtime: [bold]{runtime:.2f}[/bold] seconds[green]")

if __name__ == "__main__":
	main()
