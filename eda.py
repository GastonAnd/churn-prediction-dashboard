import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import time 

sns.set_style("whitegrid")
df_customer = pd.read_csv("Data/telecom_churn.csv", sep=",", encoding="latin1", low_memory=False)


def display_plot() -> None:
    # Esto le pone un nombre único a cada gráfico (ej: grafico_170000.png)
    nombre_archivo = f"grafico_{int(time.time() * 1000)}.png"
    
    # Guarda la imagen en alta calidad (dpi=300) y ajusta los bordes
    plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
    print(f"✅ Gráfico guardado: {nombre_archivo}")
    
    # Cierra el gráfico para que Python pueda hacer el siguiente sin trabarse
    plt.close()

def univariate_report(df: pd.DataFrame) -> None:
	print("=" * 80)
	print("EDA UNIVARIADO - TELECOM CHURN")
	print("=" * 80)
	print("Shape:", df.shape)
	print("\nPrimeras filas:\n", df.head())
	print("\nTipos de datos:\n", df.dtypes)
	print("\nNulos por variable:\n", df.isnull().sum())

	target_col = "Churn"
	binary_cols = ["Churn", "ContractRenewal", "DataPlan"]
	numeric_cols = [
		"AccountWeeks",
		"DataUsage",
		"CustServCalls",
		"DayMins",
		"DayCalls",
		"MonthlyCharge",
		"OverageFee",
		"RoamMins",
	]

	print("\n" + "=" * 80)
	print("1) Variables categoricas/binarias principales")
	print("=" * 80)
	for col in binary_cols:
		counts = df[col].value_counts(dropna=False)
		pct = df[col].value_counts(normalize=True, dropna=False).mul(100).round(2)
		summary = pd.DataFrame({"conteo": counts, "porcentaje": pct})
		print(f"\nDistribucion de {col}:")
		print(summary)

	print("\n" + "=" * 80)
	print("2) Variables numericas principales")
	print("=" * 80)
	desc = df[numeric_cols].describe().T
	desc["iqr"] = desc["75%"] - desc["25%"]
	desc["cv"] = (desc["std"] / desc["mean"]).round(3)
	print(desc)

	print("\n" + "=" * 80)
	print("3) Visualizaciones univariadas")
	print("=" * 80)

	fig, axes = plt.subplots(1, len(binary_cols), figsize=(15, 4))
	for i, col in enumerate(binary_cols):
		sns.countplot(data=df, x=col, hue=col, ax=axes[i], palette="Set2", legend=False)
		axes[i].set_title(f"Distribucion de {col}")
		axes[i].set_xlabel(col)
		axes[i].set_ylabel("Frecuencia")
	plt.tight_layout()
	display_plot()

	for col in numeric_cols:
		fig, ax = plt.subplots(1, 2, figsize=(12, 4))
		sns.histplot(data=df, x=col, kde=True, ax=ax[0], color="steelblue")
		ax[0].set_title(f"Histograma de {col}")
		ax[0].set_xlabel(col)
		ax[0].set_ylabel("Frecuencia")

		sns.boxplot(data=df, x=col, ax=ax[1], color="lightgreen")
		ax[1].set_title(f"Boxplot de {col}")
		ax[1].set_xlabel(col)

		plt.tight_layout()
		display_plot()

	churn_rate = df[target_col].mean() * 100
	print("\n" + "=" * 80)
	print("4) Insight rapido")
	print("=" * 80)
	print(f"Tasa global de churn: {churn_rate:.2f}%")
	print("Recomendacion: continuar con analisis bivariado segmentando por Churn.")


def cramers_v(contingency_table: pd.DataFrame) -> float:
	chi2 = chi2_contingency(contingency_table)[0]
	n = contingency_table.values.sum()
	r, k = contingency_table.shape
	phi2 = chi2 / n
	phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
	rcorr = r - ((r - 1) ** 2) / (n - 1)
	kcorr = k - ((k - 1) ** 2) / (n - 1)
	denominator = min((kcorr - 1), (rcorr - 1))
	if denominator <= 0:
		return 0.0
	return (phi2corr / denominator) ** 0.5


def cohens_d(x0: pd.Series, x1: pd.Series) -> float:
	n0, n1 = len(x0), len(x1)
	if n0 < 2 or n1 < 2:
		return 0.0
	var0 = x0.var(ddof=1)
	var1 = x1.var(ddof=1)
	pooled_std = (((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)) ** 0.5
	if pooled_std == 0:
		return 0.0
	return (x1.mean() - x0.mean()) / pooled_std


def bivariate_report(df: pd.DataFrame) -> None:
	target_col = "Churn"
	binary_predictors = ["ContractRenewal", "DataPlan"]
	numeric_cols = [
		"AccountWeeks",
		"DataUsage",
		"CustServCalls",
		"DayMins",
		"DayCalls",
		"MonthlyCharge",
		"OverageFee",
		"RoamMins",
	]

	print("\n" + "=" * 80)
	print("EDA BIVARIADO - VARIABLES VS CHURN")
	print("=" * 80)

	print("\n" + "=" * 80)
	print("1) Numericas vs Churn (resumen por grupo)")
	print("=" * 80)
	grouped = df.groupby(target_col)[numeric_cols].agg(["mean", "median", "std"])
	print(grouped)

	num_results = []
	for col in numeric_cols:
		x0 = df.loc[df[target_col] == 0, col].dropna()
		x1 = df.loc[df[target_col] == 1, col].dropna()

		u_stat, p_value = mannwhitneyu(x0, x1, alternative="two-sided")
		d_value = cohens_d(x0, x1)

		num_results.append(
			{
				"variable": col,
				"mean_churn_0": x0.mean(),
				"mean_churn_1": x1.mean(),
				"diff_mean_1_minus_0": x1.mean() - x0.mean(),
				"p_value_mannwhitney": p_value,
				"cohens_d": d_value,
			}
		)

	num_results_df = pd.DataFrame(num_results).sort_values("p_value_mannwhitney")
	print("\nRanking de variables numericas mas asociadas con Churn (por p-valor):")
	print(num_results_df.round(4))

	print("\n" + "=" * 80)
	print("2) Categoricas/binarias vs Churn")
	print("=" * 80)

	cat_results = []
	for col in binary_predictors:
		ct = pd.crosstab(df[col], df[target_col])
		chi2, p_value, _, _ = chi2_contingency(ct)
		v = cramers_v(ct)
		churn_rate = (
			df.groupby(col)[target_col].mean().mul(100).round(2).rename("churn_rate_pct")
		)

		print(f"\nTabla de contingencia para {col}:")
		print(ct)
		print("Tasa de churn (%):")
		print(churn_rate)

		cat_results.append(
			{
				"variable": col,
				"chi2_p_value": p_value,
				"cramers_v": v,
			}
		)

	cat_results_df = pd.DataFrame(cat_results).sort_values("chi2_p_value")
	print("\nRanking de variables categoricas mas asociadas con Churn:")
	print(cat_results_df.round(4))

	print("\n" + "=" * 80)
	print("3) Visualizaciones bivariadas")
	print("=" * 80)

	for col in numeric_cols:
		fig, ax = plt.subplots(1, 2, figsize=(12, 4))

		sns.boxplot(data=df, x=target_col, y=col, hue=target_col, palette="Set2", legend=False, ax=ax[0])
		ax[0].set_title(f"{col} vs Churn (Boxplot)")
		ax[0].set_xlabel("Churn")
		ax[0].set_ylabel(col)

		sns.kdeplot(data=df, x=col, hue=target_col, fill=True, common_norm=False, alpha=0.35, ax=ax[1])
		ax[1].set_title(f"{col} vs Churn (Distribucion)")
		ax[1].set_xlabel(col)
		ax[1].set_ylabel("Densidad")

		plt.tight_layout()
		display_plot()

	for col in binary_predictors:
		plot_df = df.groupby(col, as_index=False)[target_col].mean()
		plot_df[target_col] = plot_df[target_col] * 100

		plt.figure(figsize=(6, 4))
		sns.barplot(data=plot_df, x=col, y=target_col, hue=col, palette="Set3", legend=False)
		plt.title(f"Tasa de Churn (%) por {col}")
		plt.xlabel(col)
		plt.ylabel("Churn (%)")
		plt.tight_layout()
		display_plot()

	corr_df = df[numeric_cols + [target_col]].corr()
	plt.figure(figsize=(10, 7))
	sns.heatmap(corr_df, annot=True, cmap="RdYlBu_r", fmt=".2f", square=True)
	plt.title("Matriz de correlacion (incluye Churn)")
	plt.tight_layout()
	display_plot()

	print("\n" + "=" * 80)
	print("4) Hallazgos rapidos")
	print("=" * 80)
	top_num = num_results_df.head(3)[["variable", "p_value_mannwhitney", "cohens_d"]]
	top_cat = cat_results_df.head(2)[["variable", "chi2_p_value", "cramers_v"]]
	print("Top numericas por asociacion con Churn:")
	print(top_num.round(4))
	print("\nTop categoricas por asociacion con Churn:")
	print(top_cat.round(4))


def train_and_evaluate_models(df: pd.DataFrame) -> None:
	target_col = "Churn"
	feature_cols = [
		"AccountWeeks",
		"ContractRenewal",
		"DataPlan",
		"DataUsage",
		"CustServCalls",
		"DayMins",
		"DayCalls",
		"MonthlyCharge",
		"OverageFee",
		"RoamMins",
	]

	X = df[feature_cols]
	y = df[target_col]

	X_train, X_valid, y_train, y_valid = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	print("\n" + "=" * 80)
	print("MODELADO PREDICTIVO - CHURN")
	print("=" * 80)
	print(f"Train: {X_train.shape[0]} filas ({(X_train.shape[0] / len(df)) * 100:.0f}%)")
	print(f"Validacion: {X_valid.shape[0]} filas ({(X_valid.shape[0] / len(df)) * 100:.0f}%)")

	models = {
		"Regresion Logistica": Pipeline(
			[
				("scaler", StandardScaler()),
				("model", LogisticRegression(max_iter=1000, random_state=42)),
			]
		),
		"Arbol de Decision (Entropia)": DecisionTreeClassifier(
			criterion="entropy",
			random_state=42,
			max_depth=5,
			min_samples_leaf=20,
		),
	}

	results = []

	for model_name, model in models.items():
		model.fit(X_train, y_train)
		y_pred = model.predict(X_valid)

		acc = accuracy_score(y_valid, y_pred)
		prec = precision_score(y_valid, y_pred, zero_division=0)
		rec = recall_score(y_valid, y_pred, zero_division=0)
		f1 = f1_score(y_valid, y_pred, zero_division=0)

		results.append(
			{
				"modelo": model_name,
				"accuracy": acc,
				"precision": prec,
				"recall": rec,
				"f1": f1,
			}
		)

		cm = confusion_matrix(y_valid, y_pred)

		print("\n" + "-" * 80)
		print(f"Modelo: {model_name}")
		print(f"Accuracy: {acc:.4f}")
		print(f"Precision: {prec:.4f}")
		print(f"Recall: {rec:.4f}")
		print(f"F1-score: {f1:.4f}")
		print("Matriz de confusion:")
		print(cm)

		plt.figure(figsize=(6, 5))
		sns.heatmap(
			cm,
			annot=True,
			fmt="d",
			cmap="Blues",
			xticklabels=["Pred: No Churn", "Pred: Churn"],
			yticklabels=["Real: No Churn", "Real: Churn"],
		)
		plt.title(f"Matriz de Confusion - {model_name}")
		plt.ylabel("Valor real")
		plt.xlabel("Prediccion")
		plt.tight_layout()
		display_plot()

	results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
	print("\n" + "=" * 80)
	print("Comparativa de modelos")
	print("=" * 80)
	print(results_df.round(4))


if __name__ == "__main__":
	univariate_report(df_customer)
	bivariate_report(df_customer)
	train_and_evaluate_models(df_customer)
