import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Isključivanje sitnih matplotlib upozorenja radi čistijeg ispisa u terminalu
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Podešavanje stila za grafike
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Dinamičko podešavanje putanja na osnovu lokacije skripte
script_dir = os.path.dirname(os.path.abspath(__file__))
projekat_folder = os.path.join(script_dir, 'Life Expectancy Prediction - projekat')
data_dir = os.path.join(projekat_folder, 'data')
plots_dir = os.path.join(projekat_folder, 'izvestaj_grafici')

csv_path = os.path.join(data_dir, 'data.csv')
cleaned_csv_path = os.path.join(data_dir, 'data_cleaned.csv')
model_path = os.path.join(projekat_folder, 'izabrani_model.pkl')

# Kreiranje potrebnih foldera ukoliko ne postoje
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# =====================================================================
# FAZA 1: POČETNO PREPROCESIRANJE PODATAKA
# =====================================================================
print("\n" + "="*60)
print(" FAZA 1: POČETNO PREPROCESIRANJE PODATAKA")
print("="*60)

print(f"Učitavam sirove podatke sa lokacije:\n   {csv_path}")
try:
    df = pd.read_csv(csv_path)
    print("[OK] Podaci uspešno učitani.")
except FileNotFoundError:
    print(f" Greška: Fajl 'data.csv' nije pronađen u folderu {data_dir}")
    exit()

# Čišćenje naziva kolona od skrivenih razmaka
df.columns = df.columns.str.strip()
print(f"Ukupan broj redova na početku: {df.shape[0]}")

# 1. Uklanjanje redova gde fali ciljna promenljiva
df = df.dropna(subset=['Life expectancy'])
print(f"Broj redova nakon uklanjanja uzoraka bez ciljne promenljive: {df.shape[0]}")

# 2. Imputacija nedostajućih vrednosti (prosek države, pa globalni prosek)
num_cols = df.select_dtypes(include=[np.number]).columns
num_cols = [col for col in num_cols if col != 'Year']

for col in num_cols:
    df[col] = df.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))

# Ako je neka država imala sve NaN vrednosti za neki atribut, menjamo globalnim prosekom
preostali_nan = df.isnull().sum().sum()
if preostali_nan > 0:
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
print("[OK] Nedostajuće vrednosti uspešno popunjene (Imputacija završena).")

# 3. Enkodiranje kategorijskih promenljivih
df['Status'] = df['Status'].map({'Developed': 1, 'Developing': 0})
print("[OK] Kolona 'Status' uspešno enkodirana (Developed: 1, Developing: 0).")

# Čuvanje očišćenih podataka
df.to_csv(cleaned_csv_path, index=False)
print(f" Očišćeni podaci sačuvani na: {cleaned_csv_path}")


# =====================================================================
# FAZA 2: EKSPLORATIVNA ANALIZA PODATAKA (EDA)
# =====================================================================
print("\n" + "="*60)
print("  FAZA 2: EKSPLORATIVNA ANALIZA PODATAKA (EDA)")
print("="*60)
print("Generišem vizuelne prikaze i grafikone...")

# 1. Grafik raspodele ciljne promenljive
plt.figure(figsize=(10, 6))
sns.histplot(df['Life expectancy'], kde=True, color='skyblue', bins=30)
plt.title('Raspodela očekivanog životnog veka (Target Variable)')
plt.xlabel('Očekivani životni vek (godine)')
plt.ylabel('Učestalost')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '1_raspodela_targeta.png'))
plt.close()

# 2. Matrica korelacije
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Matrica korelacije svih numeričkih atributa')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '2_matrica_korelacije.png'))
plt.close()

# 3. Socio-ekonomski faktori grafici
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(data=df, x='Income composition of resources', y='Life expectancy', hue='Status', alpha=0.7, ax=axes[0])
axes[0].set_title('Uticaj indeksa ljudskog razvoja na životni vek')
sns.scatterplot(data=df, x='Schooling', y='Life expectancy', hue='Status', alpha=0.7, ax=axes[1])
axes[1].set_title('Uticaj obrazovanja (Schooling) na životni vek')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '3_socio_ekonomski_faktori.png'))
plt.close()

# 4. Imunizacija grafici
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df, x='Hepatitis B', y='Life expectancy', alpha=0.5, color='teal', ax=axes[0])
axes[0].set_title('Imunizacija: Hepatitis B')
sns.scatterplot(data=df, x='Polio', y='Life expectancy', alpha=0.5, color='coral', ax=axes[1])
axes[1].set_title('Imunizacija: Polio')
sns.scatterplot(data=df, x='Diphtheria', y='Life expectancy', alpha=0.5, color='crimson', ax=axes[2])
axes[2].set_title('Imunizacija: Diphtheria (DTP3)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '4_imunizacija_faktori.png'))
plt.close()

# 5. Boxplot za anomalije
anomalije_kolone = ['Adult Mortality', 'Alcohol', 'Total expenditure', 'HIV/AIDS']
plt.figure(figsize=(14, 8))
for i, col in enumerate(anomalije_kolone, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Detekcija anomalija: {col}')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '5_detekcija_anomalija.png'))
plt.close()

print(f"[OK] Generisano 5 EDA grafikona u folderu:\n     {plots_dir}")


# =====================================================================
# FAZA 3: ODABIR I TRENIRANJE MODELA (BASELINES)
# =====================================================================
print("\n" + "="*60)
print(" FAZA 3: ODABIR I TRENIRANJE MODELA (BASELINES)")
print("="*60)

# Selekcija atributa na osnovu donetih EDA odluka (uklanjanje multikolinearnosti i nekorigovanih kolona)
kolone_za_izbacivanje = ['Country', 'Population', 'infant deaths', 'percentage expenditure', 'thinness 5-9 years']
df_model = df.drop(columns=kolone_za_izbacivanje)

print(f"Broj atributa pre selekcije: {df.shape[1] - 1}")
print(f"Broj atributa nakon selekcije (EDA odluke): {df_model.shape[1] - 1}")

X = df_model.drop(columns=['Life expectancy'])
y = df_model['Life expectancy']

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Veličina trening skupa: {X_train.shape[0]} uzoraka")
print(f"Veličina test skupa: {X_test.shape[0]} uzoraka")

# Skaliranje (potrebno prvenstveno za Linearnu Regresiju)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Pokrećem obučavanje baznih modela ---")
modeli = {
    "Linear Regression": LinearRegression(),
    "Random Forest (Base)": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

rezultati = {}
for ime_modela, model in modeli.items():
    # Pokretanje sa skalanim podacima zbog linearne regresije, stabla rade podjednako dobro
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    rezultati[ime_modela] = {"MAE": mae, "RMSE": rmse, "R2 Score": r2}
    print(f"[OK] Model '{ime_modela}' je uspešno obučen.")

df_rezultati = pd.DataFrame(rezultati).T
print("\n--- UPOREDNI PREGLED PERFORMANSI POLAZNIH MODELA ---")
print(df_rezultati.to_string())


# =====================================================================
# FAZA 4: PODEŠAVANJE HIPERPARAMETARA KREIRANOG MODELA
# =====================================================================
print("\n" + "="*60)
print(" FAZA 4: PODEŠAVANJE HIPERPARAMETARA (GRID SEARCH)")
print("="*60)
print("Pokrećem unakrsnu validaciju za optimizaciju Random Forest modela...")

rf_base = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', None]
}

# Unakrsna validacija kroz 3 nabora nad trening skupom (n_jobs=-1 koristi sva jezgra Mac-a)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print("\n NAJBOLJI PRONAĐENI HIPERPARAMETRI:")
print(grid_search.best_params_)

# Evaluacija optimizovanog modela na test skupu
best_rf = grid_search.best_estimator_
y_pred_opt = best_rf.predict(X_test)

mae_opt = mean_absolute_error(y_test, y_pred_opt)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
r2_opt = r2_score(y_test, y_pred_opt)

print("\n KONAČNE PERFORMANSE OPTIMIZOVANOG MODELA (16 ATRIBUTA):")
print(f"   MAE:      {mae_opt:.6f} (prosečna greška u godinama života)")
print(f"   RMSE:     {rmse_opt:.6f}")
print(f"   R2 Score: {r2_opt:.6f}")


# =====================================================================
# FAZA 5: ODABIR NAJZNAČAJNIJIH ATRIBUTA I DEPLOYMENT
# =====================================================================
print("\n" + "="*60)
print(" FAZA 5: ODABIR NAJZNAČAJNIJIH ATRIBUTA I DEPLOYMENT")
print("="*60)

# Izdvajanje značajnosti atributa
importances = best_rf.feature_importances_
features = X.columns
df_importance = pd.DataFrame({'Atribut': features, 'Značajnost': importances}).sort_values(by='Značajnost', ascending=False)

# Čuvanje 6. grafika značajnosti (eksplicitno postavljen hue i isključen legend radi izbegavanja opomena)
plt.figure(figsize=(12, 8))
sns.barplot(x='Značajnost', y='Atribut', data=df_importance, hue='Atribut', palette='viridis', legend=False)
plt.title('Značajnost atributa u optimizovanom Random Forest modelu')
plt.xlabel('Koeficijent značajnosti (Gini Importance)')
plt.ylabel('Atributi')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '6_znacajnost_atributa.png'))
plt.close()
print("[OK] Grafik '6_znacajnost_atributa.png' uspešno sačuvan.")

print("\n📊 RANG LISTA ZNAČAJNOSTI ATRIBUTA:")
print(df_importance.to_string(index=False))

# Odabir TOP 5 najznačajnijih atributa
top_features = df_importance['Atribut'].head(5).tolist()
print(f"\n Selektujem top 5 najznačajnijih atributa za redukovani model:")
for idx, feat in enumerate(top_features, 1):
    print(f"   {idx}. {feat}")

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Obučavanje konačnog redukovanog modela na Top 5 atributa sa najboljim hiperparametrima
final_model = RandomForestRegressor(
    n_estimators=grid_search.best_params_['n_estimators'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    max_features=grid_search.best_params_['max_features'],
    random_state=42
)
final_model.fit(X_train_top, y_train)

# Evaluacija redukovanog modela
y_pred_top = final_model.predict(X_test_top)
mae_top = mean_absolute_error(y_test, y_pred_top)
rmse_top = np.sqrt(mean_squared_error(y_test, y_pred_top))
r2_top = r2_score(y_test, y_pred_top)

# Uporedna tabela performansi kompletnog i redukovanog modela
poredjenje = {
    "Svi selektovani atributi (16)": {"MAE": mae_opt, "RMSE": rmse_opt, "R2 Score": r2_opt},
    "Samo najbitniji atributi (Top 5)": {"MAE": mae_top, "RMSE": rmse_top, "R2 Score": r2_top}
}
df_poredjenje = pd.DataFrame(poredjenje).T
print("\n⚖️ UPOREDNI PREGLED KOMPLEKSNOSTI I TAČNOSTI MODELA:")
print(df_poredjenje.to_string())

# Serijalizacija (čuvanje) modela na disk (.pkl)
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\n💾 Konačni model uspešno eksportovan u: {model_path}")


# =====================================================================
#  INTERAKTIVNI KORISNIČKI INTERFEJS (UI)
# =====================================================================
print("\n" + "="*60)
print(" INTERAKTIVNI KORISNIČKI INTERFEJS (UI)")
print("="*60)
print("Unesite tražene socio-ekonomske i zdravstvene pokazatelje")
print("kako bi model uživo izračunao očekivani životni vek u godinama.\n")

try:
    print("--- Unos podataka (u zagradama su prosečne vrednosti radi orijentacije) ---")
    hiv = float(input("1. Unesite stopu HIV/AIDS-a (0.1 - 50.0): "))
    income = float(input("2. Indeks ljudskog razvoja - Income composition (0.3 - 0.9): "))
    mortality = float(input("3. Smrtnost odraslih na 1000 stanovnika - Adult Mortality (50 - 600): "))
    schooling = float(input("4. Prosečan broj godina školovanja - Schooling (4 - 20): "))
    under_five = float(input("5. Broj smrti dece do 5 godina na 1000 stanovnika (0 - 200): "))
    
    # Pakovanje podataka u DataFrame sa identičnim nazivima kolona
    korisnicki_unos = pd.DataFrame([[hiv, income, mortality, schooling, under_five]], columns=top_features)
    
    # Predikcija na osnovu učitanog/istreniranog modela
    predikcija = final_model.predict(korisnicki_unos)[0]
    
    print("\n" + "*"*50)
    print(f" REZULTAT PREDIKCIJE MODELA:")
    print(f"   Očekivani životni vek za unete parametre je: {predikcija:.2f} godina.")
    print("*"*50 + "\n")

except ValueError:
    print("\n Greška: Molimo unesite ispravne numeričke vrednosti.")
