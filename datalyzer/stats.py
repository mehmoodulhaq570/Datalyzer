"""
Feature statistics, normality, outlier detection, leakage, imbalance, etc.
"""

# Functions: feature_statistics, detect_outliers, detect_target_leakage, detect_imbalance, suggest_model, check_normality, compute_covariance_matrix, feature_target_association, class_imbalance_ratio, sensitivity_analysis, compute_vif, compute_mutual_info
import pandas as pd
import numpy as np
from scipy.stats import zscore, normaltest, chi2_contingency, f_oneway, kruskal, shapiro, kstest, skew, kurtosis, probplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from rich.table import Table
from datalyzer.config import console
def sensitivity_analysis(df, target_col, model=None, n_repeats=5, random_state=42):
	"""
	Perform sensitivity analysis using permutation importance.
	If no model is provided, uses RandomForest for regression/classification.
	Returns a DataFrame of feature importances.
	"""
	from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
	from sklearn.inspection import permutation_importance
	import pandas as pd
	X = df.drop(columns=[target_col])
	y = df[target_col]
	if model is None:
		if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
			model = RandomForestRegressor(random_state=random_state)
		else:
			model = RandomForestClassifier(random_state=random_state)
	X = X.select_dtypes(include=["float64", "int64"])
	if X.empty:
		return pd.DataFrame(columns=["Feature", "Importance"])
	model.fit(X.fillna(0), y)
	result = permutation_importance(model, X.fillna(0), y, n_repeats=n_repeats, random_state=random_state)
	importances = pd.DataFrame({
		"Feature": X.columns,
		"Importance": result.importances_mean
	}).sort_values(by="Importance", ascending=False)
	return importances

def compute_covariance_matrix(df, output_dir=None):
	"""
	Compute the covariance matrix of numeric columns in the DataFrame.
	If output_dir is provided, save the matrix as CSV and return the file path.
	"""
	import os
	numeric_df = df.select_dtypes(include=["float64", "int64"])
	if numeric_df.empty:
		return None
	cov_matrix = numeric_df.cov()
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
		file_path = os.path.join(output_dir, "covariance_matrix.csv")
		cov_matrix.to_csv(file_path)
		return file_path
	return cov_matrix
	cov_matrix = numeric_df.cov()
	return cov_matrix
	"""
	Perform sensitivity analysis using permutation importance.
	If no model is provided, uses RandomForest for regression/classification.
	Returns a DataFrame of feature importances.
	"""
	from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
	from sklearn.inspection import permutation_importance
	import pandas as pd
	X = df.drop(columns=[target_col])
	y = df[target_col]
	if model is None:
		if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
			model = RandomForestRegressor(random_state=random_state)
		else:
			model = RandomForestClassifier(random_state=random_state)
	X = X.select_dtypes(include=["float64", "int64"])
	if X.empty:
		return pd.DataFrame(columns=["Feature", "Importance"])
	model.fit(X.fillna(0), y)
	result = permutation_importance(model, X.fillna(0), y, n_repeats=n_repeats, random_state=random_state)
	importances = pd.DataFrame({
		"Feature": X.columns,
		"Importance": result.importances_mean
	}).sort_values(by="Importance", ascending=False)
	return importances

def feature_statistics(df):
	stats = df.describe(include='all').transpose()
	stats['missing_values'] = df.isnull().sum()
	stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
	stats['unique_values'] = df.nunique()
	numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
	stats.loc[numeric_cols, 'variance'] = df[numeric_cols].var()
	table = Table(show_header=True, header_style="bold magenta")
	for col in stats.columns:
		table.add_column(str(col))
	for index, row in stats.iterrows():
		table.add_row(*[str(x) for x in row.values])
	console.print(table)
	readiness = 100
	suggestions = []
	if stats['missing_pct'].max() > 0:
		readiness -= 10
		suggestions.append("Consider imputing missing values.")
	if stats['unique_values'].max() < 2:
		readiness -= 10
		suggestions.append("Some features may be constant; remove them.")
	if len(numeric_cols) == 0:
		readiness -= 20
		suggestions.append("No numeric features detected; may affect modeling.")
	console.print(f"[bold green]ML Readiness Score: {readiness}/100[/bold green]")
	if suggestions:
		console.print("[yellow]Suggestions:[/yellow]")
		for s in suggestions:
			console.print(f"- {s}")
	if df.duplicated().sum() > 0:
		readiness -= 5
		suggestions.append("Duplicate rows detected. Consider removing duplicates.")
	if df.isnull().sum().max() > len(df) * 0.5:
		readiness -= 15
		suggestions.append("Some columns have >50% missing values.")
	if len(df.columns) > 100:
		readiness -= 5
		suggestions.append("High dimensionality may require feature selection.")
	return stats, readiness, suggestions

def detect_outliers(df, numeric_columns):
	outliers = {}
	for col in numeric_columns:
		try:
			z_scores = zscore(df[col].dropna())
			outlier_idx = np.where(abs(z_scores) > 3)[0]
			outlier_pct = len(outlier_idx) / len(df) * 100
			outliers[col] = {'count': len(outlier_idx), 'percent': round(outlier_pct,2)}
		except Exception:
			outliers[col] = {'count': 0, 'percent': 0}
	return outliers

def detect_target_leakage(df, target_col):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	leakage_features = []
	X = df.drop(columns=[target_col])
	y = df[target_col]
	results = []
	for col in X.columns:
		if X[col].dtype in ['int64', 'float64']:
			try:
				clf = RandomForestClassifier(n_estimators=50)
				clf.fit(X[[col]].fillna(0), y)
				y_pred = clf.predict(X[[col]].fillna(0))
				if pd.api.types.is_numeric_dtype(y) and y.nunique() > len(y) * 0.5:
					from sklearn.metrics import r2_score
					acc = r2_score(y, y_pred)
				else:
					acc = accuracy_score(y, y_pred)
				results.append((col, acc))
				if acc > 0.95:
					leakage_features.append(col)
			except Exception:
				pass
	return leakage_features

def detect_imbalance(df, target_col):
	try:
		counts = df[target_col].value_counts(normalize=True)
		imbalance = counts[counts < 0.3]
		return imbalance.to_dict()
	except Exception:
		return {}

def suggest_model(df, target_col):
	col_type = df[target_col].dtype
	unique_vals = df[target_col].nunique()
	if col_type in ['int64', 'float64']:
		suggestion = "Classification (numeric)" if unique_vals <= 10 else "Regression"
	else:
		suggestion = "Classification (categorical)"
	console.print(f"[bold magenta]Suggested Model Type: {suggestion}[/bold magenta]")
	return suggestion

def run_statistical_diagnostics(df, target_col):
	normality_dict = check_normality(df)
	normality_results = list(normality_dict.values())
	vif = compute_vif(df)
	mi = compute_mutual_info(df, target_col)
	return {"normality": normality_results, "vif": vif.to_dict(), "mutual_info": mi.to_dict()}

def get_top_correlations(df, threshold=0.8, top_n=10):
	if df is None or df.empty:
		return []
	corr_matrix = df.corr().abs()
	corr_matrix_values = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
	top_pairs = (
		corr_matrix_values.stack()
		.sort_values(ascending=False)
		.head(top_n)
		.reset_index()
		.values.tolist()
	)
	top_pairs = [tuple(x) for x in top_pairs if x[2] >= threshold]
	return top_pairs

def check_normality(df, output_dir="reports/plots"):
	import os
	results = {}
	os.makedirs(output_dir, exist_ok=True)
	numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
	for col in numeric_cols:
		data = df[col].dropna()
		if len(data) < 10:
			continue
		try:
			sample = data.sample(min(5000, len(data)), random_state=42)
			shapiro_stat, shapiro_p = shapiro(sample)
			dagostino_stat, dagostino_p = normaltest(sample)
			ks_stat, ks_p = kstest(sample, 'norm', args=(sample.mean(), sample.std()))
			results[col] = {
				"feature": col,
				"skew": round(skew(sample), 3),
				"kurtosis": round(kurtosis(sample), 3),
				"Shapiro_p": round(shapiro_p, 4),
				"DAgostino_p": round(dagostino_p, 4),
				"KS_p": round(ks_p, 4),
				"p_value": round(np.mean([shapiro_p, dagostino_p, ks_p]), 4),
				"is_normal": all(p > 0.05 for p in [shapiro_p, dagostino_p, ks_p]),
			}
			import matplotlib.pyplot as plt
			probplot(sample, dist="norm", plot=plt)
			plt.title(f"Distribution of {col} (QQ Plot)", fontsize=20, fontweight="bold", fontname="Times New Roman")
			plt.xlabel(col, fontsize=16, fontweight='bold', fontname="Times New Roman")
			plt.ylabel("Quantiles", fontsize=16, fontweight='bold', fontname="Times New Roman")
			plt.xticks(fontsize=14, fontweight='bold', fontname="Times New Roman")
			plt.yticks(fontsize=14, fontweight='bold', fontname="Times New Roman")
			plt.tight_layout()
			plot_path = os.path.join(output_dir, f"qqplot_{col}.png")
			plt.savefig(plot_path)
			plt.close()
		except Exception as e:
			results[col] = {
				"feature": col,
				"skew": 0,
				"kurtosis": 0,
				"Shapiro_p": 0,
				"DAgostino_p": 0,
				"KS_p": 0,
				"is_normal": False,
				"error": str(e)
			}
	return results

def compute_vif(df):
	numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()
	if numeric_df.shape[1] < 2:
		return pd.DataFrame(columns=["Feature", "VIF"])
	vif_data = []
	for i, col in enumerate(numeric_df.columns):
		try:
			vif_value = variance_inflation_factor(numeric_df.values, i)
			vif_data.append({"Feature": col, "VIF": round(vif_value, 3)})
		except Exception:
			vif_data.append({"Feature": col, "VIF": np.nan})
	vif_df = pd.DataFrame(vif_data)
	return vif_df.sort_values(by="VIF", ascending=False)

def compute_mutual_info(df, target_col):
	if target_col not in df.columns:
		return pd.DataFrame(columns=["Feature", "Mutual_Info"])
	y = df[target_col]
	X = df.drop(columns=[target_col])
	X_num = X.select_dtypes(include=["float64", "int64"])
	if X_num.empty:
		return pd.DataFrame(columns=["Feature", "Mutual_Info"])
	try:
		if pd.api.types.is_numeric_dtype(y):
			mi = mutual_info_regression(X_num, y, random_state=42)
		else:
			mi = mutual_info_classif(X_num, y.astype("category").cat.codes, random_state=42)
		mi_df = pd.DataFrame({
			"Feature": X_num.columns,
			"Mutual_Info": np.round(mi, 4)
		}).sort_values(by="Mutual_Info", ascending=False)
	except Exception as e:
		mi_df = pd.DataFrame(columns=["Feature", "Mutual_Info"])
	return mi_df

def feature_target_association(df, target):
    """Measure strength of relationship between each feature and target."""
    results = {}
    if target not in df.columns:
        return results
    y = df[target]
    for col in df.columns:
        if col == target:
            continue
        x = df[col].dropna()
        try:
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                # ANOVA (parametric)
                stat, p = f_oneway(x, y)
                test = "ANOVA"
            elif pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_numeric_dtype(y):
                # Kruskal–Wallis (nonparametric)
                groups = [x[y == val] for val in np.unique(y)]
                stat, p = kruskal(*groups)
                test = "Kruskal–Wallis"
            else:
                # Chi-square test for categorical association
                table = pd.crosstab(x, y)
                stat, p, dof, _ = chi2_contingency(table)
                test = "Chi-square"
            results[col] = {"test": test, "stat": round(stat, 3), "p_value": round(p, 4)}
        except Exception as e:
            results[col] = {"error": str(e)}
    return results

def class_imbalance_ratio(df, target):
    """Compute class imbalance ratio for classification targets."""
    if target not in df.columns:
        return None
    y = df[target]
    if y.nunique() <= 1:
        return None
    counts = y.value_counts()
    ir = round(counts.max() / counts.min(), 2)
    return {"counts": counts.to_dict(), "imbalance_ratio": ir}
