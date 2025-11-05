
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rich.console import Console
import matplotlib
matplotlib.use('Agg')

console = Console()

def save_unique_plot(fig, output_dir, filename_base):
	os.makedirs(output_dir, exist_ok=True)
	# Sanitize filename: remove invalid characters for Windows/Linux
	safe_filename = filename_base.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace('°', 'deg').replace('(', '').replace(')', '')
	base_path = os.path.join(output_dir, f"{safe_filename}.png")
	if not os.path.exists(base_path):
		fig.savefig(base_path, dpi=300, bbox_inches='tight')
		return base_path
	counter = 2
	while True:
		new_path = os.path.join(output_dir, f"{safe_filename}_{counter}.png")
		if not os.path.exists(new_path):
			fig.savefig(new_path, dpi=300, bbox_inches='tight')
			return new_path
		counter += 1

def plot_distributions(df, plots_dir="plots", clean=True):
	os.makedirs(plots_dir, exist_ok=True)
	if clean:
		for f in os.listdir(plots_dir):
			if f.endswith(".png"):
				try:
					os.remove(os.path.join(plots_dir, f))
				except Exception:
					pass
	numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
	distribution_paths = []
	boxplot_paths = []
	violin_paths = []
	swarm_paths = []
	def plot_all(col):
		paths = []
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.histplot(df[col].dropna(), kde=True, bins=30, edgecolor="black", linewidth=0.6, ax=ax)
		ax.set_title(f"Distribution of {col}", fontsize=18, fontweight="bold", fontname="Times New Roman")
		ax.set_xlabel(col, fontsize=15, fontweight="bold", fontname="Times New Roman")
		ax.set_ylabel("Frequency", fontsize=15, fontweight="bold", fontname="Times New Roman")
		ax.tick_params(labelsize=13)
		plt.grid(True, alpha=0.3, linestyle="--")
		plt.tight_layout()
		path = save_unique_plot(fig, plots_dir, f"{col}_distribution")
		plt.close(fig)
		paths.append(('distribution', path))
		fig, ax = plt.subplots(figsize=(6, 4))
		sns.boxplot(y=df[col].dropna(), color=plt.cm.Blues(0.6), linewidth=1.2, ax=ax)
		ax.set_title(f"Boxplot of {col}", fontsize=13, fontweight="bold", fontname="Times New Roman")
		ax.set_ylabel(col, fontsize=12, fontweight="bold", fontname="Times New Roman")
		ax.tick_params(labelsize=10)
		plt.grid(True, alpha=0.3, linestyle="--")
		plt.tight_layout()
		path = save_unique_plot(fig, plots_dir, f"{col}_boxplot")
		plt.close(fig)
		paths.append(('boxplot', path))
		fig, ax = plt.subplots(figsize=(6, 4))
		sns.violinplot(y=df[col].dropna(), color=sns.color_palette("deep")[0], linewidth=1.2, ax=ax)
		ax.set_title(f"Violin Plot of {col}", fontsize=13, fontweight="bold", fontname="Times New Roman")
		ax.set_ylabel(col, fontsize=12, fontweight="bold", fontname="Times New Roman")
		ax.tick_params(labelsize=10)
		plt.tight_layout()
		path = save_unique_plot(fig, plots_dir, f"{col}_violinplot")
		plt.close(fig)
		paths.append(('violin', path))
		# Swarm plot - limit to 2000 points for performance
		data_for_swarm = df[col].dropna()
		if len(data_for_swarm) > 2000:
			data_for_swarm = data_for_swarm.sample(n=2000, random_state=42)
		
		fig, ax = plt.subplots(figsize=(6, 4))
		sns.boxplot(y=data_for_swarm, color=plt.cm.cividis(0.6), fliersize=0, linewidth=1, ax=ax)
		sns.swarmplot(y=data_for_swarm, color="black", size=3, alpha=0.7, ax=ax)
		ax.set_title(f"Swarm Plot of {col}", fontsize=13, fontweight="bold", fontname="Times New Roman")
		ax.set_ylabel(col, fontsize=12, fontweight="bold", fontname="Times New Roman")
		ax.tick_params(labelsize=10)
		plt.tight_layout()
		path = save_unique_plot(fig, plots_dir, f"{col}_swarmplot")
		plt.close(fig)
		paths.append(('swarm', path))
		return paths
	for col in numeric_cols:
		res = plot_all(col)
		for kind, path in res:
			if kind == 'distribution':
				distribution_paths.append(path)
			elif kind == 'boxplot':
				boxplot_paths.append(path)
			elif kind == 'violin':
				violin_paths.append(path)
			elif kind == 'swarm':
				swarm_paths.append(path)
	all_plot_paths = {
		"distribution": distribution_paths,
		"boxplot": boxplot_paths,
		"violin": violin_paths,
		"swarm": swarm_paths
	}
	return all_plot_paths

def plot_correlation(df, plots_dir="plots"):
	os.makedirs(plots_dir, exist_ok=True)
	numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
	corr_methods = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}
	def plot_corr(name, method):
		corr = df[numeric_cols].corr(method=method)
		plt.figure(figsize=(10,8))
		sns.heatmap(
			corr, annot=True, fmt=".2f", cmap='plasma', cbar=True,
			square=True, linewidths=0.5,
			annot_kws={'fontweight': 'bold', 'fontsize': 14, 'fontname': 'Times New Roman'}
		)
		plt.title(f"{name} Correlation Matrix", fontsize=20, fontweight="bold", fontname="Times New Roman", pad=15)
		plt.xticks(fontsize=15, fontweight='bold', ha='right', fontname="Times New Roman")
		plt.yticks(fontsize=15, fontweight='bold', rotation=0, fontname="Times New Roman")
		plt.tight_layout()
		path = os.path.join(plots_dir, f"correlation_{method}.png")
		plt.savefig(path, dpi=300, bbox_inches='tight')
		plt.close()
		return path
	corr_files = []
	for name, method in corr_methods.items():
		corr_files.append(plot_corr(name, method))
	corr = df[numeric_cols].corr(method='pearson')
	mask = np.triu(np.ones_like(corr, dtype=bool))
	plt.figure(figsize=(10,8))
	sns.heatmap(
		corr, mask=mask, annot=True, fmt=".2f", cmap='plasma',
		cbar=True, square=True, linewidths=0.5,
		annot_kws={'fontweight': 'bold', 'fontsize': 14, 'fontname': 'Times New Roman'}
	)
	plt.title("Pearson Correlation (Upper Triangle)", fontsize=20, fontweight="bold", fontname="Times New Roman")
	plt.xticks(fontsize=15, fontweight='bold', ha='right', fontname="Times New Roman")
	plt.yticks(fontsize=15, fontweight='bold', rotation=0, fontname="Times New Roman")
	plt.tight_layout()
	upper_path = os.path.join(plots_dir, "correlation_pearson_upper.png")
	plt.savefig(upper_path, dpi=300, bbox_inches='tight')
	plt.close()
	corr_files.append(upper_path)
	return corr_files

def compute_feature_importance(df, target_col, plots_dir):
	if target_col not in df.columns:
		return None, {}
	X = df.drop(columns=[target_col]).select_dtypes(include=['float64', 'int64'])
	y = df[target_col]
	if y.nunique() < 2 or X.empty:
		return None, {}
	is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > len(y) * 0.5
	if is_regression:
		model = RandomForestRegressor(n_estimators=100, random_state=42)
	else:
		model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X.fillna(0), y)
	importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
	plt.figure(figsize=(10, 6))
	sns.barplot(
		x=importances.values,
		y=importances.index,
		hue=importances.index,
		palette="viridis",
		legend=False
	)
	plt.title("Feature Importances", fontsize=24, fontweight="bold", fontname="Times New Roman")
	plt.xlabel("Importance Score", fontsize=18, fontweight="bold", fontname="Times New Roman")
	plt.ylabel("Feature", fontsize=18, fontweight="bold", fontname="Times New Roman")
	plt.xticks(fontsize=16, fontweight='bold', ha='right', fontname="Times New Roman")
	plt.yticks(fontsize=16, fontweight='bold', rotation=0, fontname="Times New Roman")
	plt.tight_layout()
	path = os.path.join(plots_dir, "feature_importance.png")
	plt.savefig(path, dpi=300, bbox_inches='tight')
	plt.close()
	return path, importances.to_dict()

def dimensionality_estimation(df, output_dir):
	num_df = df.select_dtypes(include=['float64', 'int64']).fillna(0)
	if num_df.shape[1] < 2:
		console.print("[red]❌ PCA/t-SNE plots not generated: less than 2 numeric columns.[/red]")
		return None, None
	plots_dir = os.path.join(output_dir, "plots")
	os.makedirs(plots_dir, exist_ok=True)
	try:
		pca = PCA(n_components=min(5, num_df.shape[1]))
		pca.fit(num_df)
		explained = pca.explained_variance_ratio_
		pca_path = os.path.join(plots_dir, "pca_variance.png")
		plt.figure(figsize=(10, 6))
		plt.bar(range(1, len(explained)+1), explained, alpha=0.7, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(explained))))
		plt.xlabel("Principal Component", fontsize=18, fontweight="bold", fontname="Times New Roman")
		plt.ylabel("Explained Variance Ratio", fontsize=18, fontweight="bold", fontname="Times New Roman")
		plt.title("PCA Variance Explained", fontsize=24, fontweight="bold", fontname="Times New Roman", pad=15)
		plt.xticks(fontsize=16, fontweight='bold', fontname="Times New Roman")
		plt.yticks(fontsize=16, fontweight='bold', fontname="Times New Roman")
		plt.grid(True, alpha=0.3, linestyle="--")
		plt.tight_layout()
		plt.savefig(pca_path, dpi=300, bbox_inches="tight")
		plt.close()
	except Exception:
		pca_path = None
	try:
		tsne = TSNE(n_components=2, random_state=42, perplexity=5)
		tsne_data = tsne.fit_transform(num_df)
		tsne_path = os.path.join(plots_dir, "tsne_scatter.png")
		plt.figure(figsize=(10, 6))
		plt.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.6, c=tsne_data[:, 0], cmap="viridis", edgecolor="k", s=60)
		plt.title("t-SNE 2D Projection", fontsize=24, fontweight="bold", fontname="Times New Roman")
		plt.xlabel("t-SNE 1", fontsize=18, fontweight="bold", fontname="Times New Roman")
		plt.ylabel("t-SNE 2", fontsize=18, fontweight="bold", fontname="Times New Roman")
		plt.xticks(fontsize=16, fontweight='bold', fontname="Times New Roman")
		plt.yticks(fontsize=16, fontweight='bold', fontname="Times New Roman")
		plt.grid(True, alpha=0.3, linestyle="--")
		plt.tight_layout()
		plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
		plt.close()
	except Exception:
		tsne_path = None
	return pca_path, tsne_path
