import os
import json
import pdfkit
import numpy as np
from scipy.stats import zscore

def generate_interactive_report(
	df,
	stats_df,
	outliers,
	normality,
	readiness_score,
	suggestions,
	leakage,
	imbalance,
	plot_paths,
	plots_dir,
	output_dir,
	template,
	html_file,
	json_file,
	timestamp
):
	# 🧮 Format Outlier Examples
	formatted_outliers = {}
	for col, info in outliers.items():
		try:
			numeric_values = df[col].dropna()
			z_scores = zscore(numeric_values)
			idx = np.where(abs(z_scores) > 3)[0]
			examples = numeric_values.iloc[idx].tolist()[:5]
		except Exception:
			examples = []
		formatted_outliers[col] = {"count": info['count'], "examples": examples}

	# Group plots by type for sectioned galleries
	plot_groups = {
		'QQ Plots': [],
		'Distribution Plots': [],
		'Boxplots': [],
		'Violin Plots': [],
		'Swarm Plots': [],
		'Feature Importance': [],
		'PCA': [],
		't-SNE': [],
		'Correlation': []
	}
	if os.path.exists(plots_dir):
		for f in sorted(os.listdir(plots_dir)):
			rel_path = os.path.relpath(os.path.join(plots_dir, f), start=output_dir)
			if f.startswith("qqplot_"):
				plot_groups['QQ Plots'].append(rel_path)
			if f.endswith("_distribution.png"):
				plot_groups['Distribution Plots'].append(rel_path)
			if f.endswith("_boxplot.png"):
				plot_groups['Boxplots'].append(rel_path)
			if f.endswith("_violinplot.png"):
				plot_groups['Violin Plots'].append(rel_path)
			if f.endswith("_swarmplot.png"):
				plot_groups['Swarm Plots'].append(rel_path)
			if f == "feature_importance.png":
				if rel_path not in plot_groups['Feature Importance']:
					plot_groups['Feature Importance'].append(rel_path)
			if f == "pca_variance.png":
				if rel_path not in plot_groups['PCA']:
					plot_groups['PCA'].append(rel_path)
			if f == "tsne_scatter.png":
				if rel_path not in plot_groups['t-SNE']:
					plot_groups['t-SNE'].append(rel_path)
	# Ensure only one PCA and t-SNE plot is shown
	if len(plot_groups['PCA']) > 1:
		plot_groups['PCA'] = [plot_groups['PCA'][0]]
	if len(plot_groups['t-SNE']) > 1:
		plot_groups['t-SNE'] = [plot_groups['t-SNE'][0]]
	if len(plot_groups['Feature Importance']) > 1:
		plot_groups['Feature Importance'] = [plot_groups['Feature Importance'][0]]
	# Add any file containing 'correlation' in its name to Correlation group
	if "correlation" in f.lower() and rel_path not in plot_groups['Correlation']:
		plot_groups['Correlation'].append(rel_path)
	# Add correlation plots from plot_paths if present
	corr_plot = None
	if plot_paths:
		if isinstance(plot_paths, dict):
			if "correlation" in plot_paths and isinstance(plot_paths["correlation"], list):
				for p in plot_paths["correlation"]:
					rel_path = os.path.relpath(p, start=output_dir) if os.path.exists(p) else p
					plot_groups['Correlation'].append(rel_path)
				if plot_paths["correlation"]:
					corr_plot = os.path.relpath(plot_paths["correlation"][0], start=output_dir) if os.path.exists(plot_paths["correlation"][0]) else plot_paths["correlation"][0]
		elif isinstance(plot_paths, list):
			for p in plot_paths:
				rel_path = os.path.relpath(p, start=output_dir) if os.path.exists(p) else p
				plot_groups['Correlation'].append(rel_path)

	stats_html = stats_df.to_html(index=True)
	from .stats import get_top_correlations
	top_corrs = get_top_correlations(df, threshold=0.8, top_n=10)

	# Reorder plot_groups for report and fix correlation plot order
	corr_order = [
		'correlation_pearson.png',
		'correlation_spearman.png',
		'correlation_kendall.png',
		'correlation_pearson_upper.png'
	]
	plot_groups['Correlation'] = [
		os.path.relpath(os.path.join(plots_dir, fname), start=output_dir)
		for fname in corr_order
		if os.path.exists(os.path.join(plots_dir, fname))
	]

	ordered_keys = [
		'Distribution Plots',
		'Boxplots',
		'Violin Plots',
		'Swarm Plots',
		'Correlation',
		'QQ Plots',
		'Feature Importance',
		'PCA',
		't-SNE'
	]
	# Remove duplicate Feature Importance plot if present (after all additions)
	if len(plot_groups['Feature Importance']) > 1:
		# Only keep the first occurrence
		plot_groups['Feature Importance'] = [plot_groups['Feature Importance'][0]]
	ordered_plot_groups = {k: plot_groups[k] for k in ordered_keys if k in plot_groups}

	html_out = template.render(
		readiness_score=readiness_score,
		suggestions=suggestions,
		stats_html=stats_html,
		outliers=formatted_outliers,
		normality=normality,
		top_corrs=top_corrs,
		leakage=leakage,
		imbalance=imbalance,
		plot_groups=ordered_plot_groups,
		corr_plot=corr_plot,
		timestamp=timestamp
	)

	# 🧱 Save HTML
	with open(html_file, "w", encoding="utf-8") as f:
		f.write(html_out)

	# 🪄 Generate PDF (if wkhtmltopdf available)
	try:
		pdfkit.from_file(html_file, html_file.replace(".html", ".pdf"))
		print(f"\n[green]PDF report generated: {html_file.replace('.html', '.pdf')}[/green]")
	except Exception as e:
		if "wkhtmltopdf" in str(e).lower():
			print("[red]wkhtmltopdf not found! Install via: choco install wkhtmltopdf (run as admin)[/red]")
		else:
			print(f"[yellow]PDF generation skipped: {e}[yellow]")

	# 🧩 JSON Summary
	summary = {
		"timestamp": timestamp,
		"ml_readiness_score": readiness_score,
		"readiness_suggestions": suggestions,
		"outliers": formatted_outliers,
		"leakage": leakage,
		"imbalance": imbalance,
		"normality": normality,
		"rows": df.shape[0],
		"columns": df.shape[1]
	}

	with open(json_file, "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=4)

	print(f"[green]JSON summary saved: {json_file}[green]")
"""
HTML/PDF/JSON report generation for Datalyzer package
"""

# Functions: generate_interactive_report
