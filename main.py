import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.diagnostic as sd

df = pd.read_csv("dataset_1.csv")
df.dropna(inplace=True)

print("Resumo estatístico:")
print(df.describe())

#Variáveis categóricas
cat_vars = df.select_dtypes(include='object').columns.tolist()
print("\nVariáveis categóricas encontradas:")
print(cat_vars)

#Regressão Linear Múltipla
df_encoded = pd.get_dummies(df, columns=cat_vars, drop_first=True)

df_encoded = df_encoded.astype(float)

#Separar variável dependente
y = df_encoded["tempo_resposta"]
X = df_encoded.drop(columns=["tempo_resposta"])
X_const = sm.add_constant(X)

model = sm.OLS(y, X_const).fit()

print("\nResumo do modelo:")
print(model.summary())

#Diagnóstico de Multicolinearidade
vif = pd.DataFrame()
vif["variavel"] = X_const.columns
vif["VIF"] = [oi.variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print("\nFatores de inflação da variância (VIF):")
print(vif)

#Heterocedasticidade
residuos = model.resid
valores_ajustados = model.fittedvalues

plt.figure(figsize=(8, 5))
sns.scatterplot(x=valores_ajustados, y=residuos)
plt.axhline(0, color='red', linestyle='--')
plt.title('Heterocedasticidade')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.tight_layout()
plt.savefig("grafico.png")
plt.close()

#Teste de Breusch-Pagan
_, pval, _, _ = sd.het_breuschpagan(residuos, X_const)
print(f"\nTeste de Breusch-Pagan - p-valor: {pval:.4f}")
if pval < 0.05:
    print("→ Heterocedasticidade detectada.")
else:
    print("→ Homoscedasticidade (sem heterocedasticidade).")

modelo_1 = model

#Modelo 2 removendo variável ram_gb
if "ram_gb" in X.columns:
    X2 = X.drop(columns=["ram_gb"])
else:
    X2 = X.copy()
X2_const = sm.add_constant(X2)
modelo_2 = sm.OLS(y, X2_const).fit()

print("\nResumo Modelo 2 (sem ram_gb):")
print(modelo_2.summary())

print("\nComparação de R² ajustado:")
print(f"Modelo 1: {modelo_1.rsquared_adj:.4f}")
print(f"Modelo 2: {modelo_2.rsquared_adj:.4f}")

# Teste F para comparar os modelos
anova = sm.stats.anova_lm(modelo_2, modelo_1)
print("\nTeste F entre os modelos:")
print(anova)