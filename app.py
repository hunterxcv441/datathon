import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------------------------------------------------------
# Função para carregar os dados
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("dados_tratados.csv")
    int_columns = ["Ano_ingresso", "mean", "std", "min", "25%"]
    for col in int_columns:
        if col in data.columns:
            data[col] = data[col].round().astype(int)
    return data


df = load_data()


def calculate_selection_percentage(original_df, filtered_df):
    total = len(original_df)
    selected = len(filtered_df)
    percentage = (selected / total) * 100 if total > 0 else 0
    return percentage


# -----------------------------------------------------------------------------
# Função para treinar o modelo com atualização dinâmica
# -----------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def train_model(data):
    features = ["IAN", "IEG", "IPV", "IPS", "IDA", "Turma", "Gênero", "Instituição_de_ensino"]

    df_encoded = pd.get_dummies(data[features], drop_first=True).fillna(0)
    scaler = StandardScaler()
    numeric_features = ["IAN", "IEG", "IPV", "IPS", "IDA"]
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

    X = df_encoded
    y = data["Defasagem"]

    if len(X) < 2:
        return None, df_encoded.columns, scaler, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5,
                                              random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                                                      random_state=42),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'LinearRegression': LinearRegression()
    }

    best_model = None
    best_mae = float('inf')
    best_r2 = float('-inf')

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if mae < best_mae:
            best_mae = mae
            best_r2 = r2
            best_model = model

    model = best_model
    model.fit(X_train, y_train)

    if len(X_test) >= 2:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    else:
        mae = None
        r2 = None

    return model, df_encoded.columns, scaler, best_mae, best_r2


# -----------------------------------------------------------------------------
# Processamento dos filtros e entrada do usuário
# -----------------------------------------------------------------------------
select_all = st.checkbox("Selecionar todas as variáveis para análise e travar a seleção")

df_filtered = df.copy()

turma_options = df_filtered["Turma"].unique()
turma = st.multiselect("Selecione a Turma", turma_options, default=turma_options if select_all else [])
df_filtered = df_filtered[df_filtered["Turma"].isin(turma)] if turma else df_filtered

genero_options = df_filtered["Gênero"].unique()
genero = st.selectbox("Selecione o Gênero", genero_options if not select_all else ["Todos"])
df_filtered = df_filtered[df_filtered["Gênero"] == genero] if genero != "Todos" else df_filtered

instituicao_options = df_filtered["Instituição_de_ensino"].unique()
instituicao = st.multiselect("Selecione a Instituição", instituicao_options,
                             default=instituicao_options if select_all else [])
df_filtered = df_filtered[df_filtered["Instituição_de_ensino"].isin(instituicao)] if instituicao else df_filtered

# Exibir barra de progresso da porcentagem de dados filtrados
selection_percentage = calculate_selection_percentage(df, df_filtered)
st.progress(selection_percentage / 100)
st.write(f"📊 **{selection_percentage:.2f}%** da base de dados foi selecionada.")

# Treinar modelo com os filtros aplicados
model, feature_names, scaler, mae, r2 = train_model(df_filtered)

# Inputs do usuário via sliders
ian = st.slider("IAN (Adequação ao Nível)", min_value=2.5, max_value=10.0, value=7.5)
ieg = st.slider("IEG (Engajamento)", min_value=0.0, max_value=10.0, value=7.5)
ipv = st.slider("IPV (Ponto de Virada)", min_value=2.9, max_value=10.0, value=7.0)
ips = st.slider("IPS (Psicossocial)", min_value=2.5, max_value=10.0, value=6.5)
ida = st.slider("IDA (Aprendizagem)", min_value=0.0, max_value=10.0, value=6.5)

# Criar DataFrame com os inputs
input_data = pd.DataFrame(
    [[ian, ieg, ipv, ips, ida, "; ".join(map(str, turma)) if turma else "Todas",
      genero if genero != "Todos" else "Todos",
      "; ".join(map(str, instituicao)) if instituicao else "Todas"]],
    columns=["IAN", "IEG", "IPV", "IPS", "IDA", "Turma", "Gênero", "Instituição_de_ensino"]
)

input_encoded = pd.get_dummies(input_data).reindex(columns=feature_names, fill_value=0)
input_encoded[["IAN", "IEG", "IPV", "IPS", "IDA"]] = scaler.transform(
    input_encoded[["IAN", "IEG", "IPV", "IPS", "IDA"]])

if model is not None:
    predicted_defasagem = model.predict(input_encoded)[0]
    st.write(f"### Defasagem Prevista: {predicted_defasagem:.2f} anos")
    st.write(f"- **Turma:** {', '.join(map(str, turma)) if turma else 'Todas'}")
    st.write(f"- **Gênero:** {genero if genero != 'Todos' else 'Todos'}")
    st.write(f"- **Instituição:** {', '.join(map(str, instituicao)) if instituicao else 'Todas'}")
else:
    st.error("⚠️ Dados insuficientes para treinar o modelo. Ajuste os filtros para incluir mais amostras.")

st.subheader("Desempenho do Modelo Atualizado:")
if mae is not None and r2 is not None:
    st.write(f"📏 **Erro Médio Absoluto (MAE):** {mae:.2f} anos")
    st.write(f"📈 **Coeficiente de Determinação (R²):** {r2:.2f}")
else:
    st.warning("⚠️ Não há amostras suficientes para calcular métricas do modelo.")
