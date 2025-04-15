import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Charger les données
df = pd.read_csv('data\churn.csv', index_col="customerID")

# Convertir la colonne qui n'est pas dans le bon format
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Supprimer les valeurs nulles
df.dropna(inplace=True)

# Encodage des variables
df['Churn'] = df['Churn'].map({'No' : 0, 'Yes' : 1})
df_dummies = pd.get_dummies(df, dtype=int, drop_first=True)

## Séparation des données

X = df_dummies.drop('Churn', axis=1)
y = df_dummies['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Normalisation des données

scaler = StandardScaler()

X_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
X_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Entrainer le modèle 
model = SVC(C=0.1, gamma=1, kernel='linear')
model.fit(X_train, y_train)

# Sauvegarder le modèle et le scaler
joblib.dump(model, 'model/churn_model.pkl')
joblib.dump(scaler, 'model/churn_scaler.pkl')