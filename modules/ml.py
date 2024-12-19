import gradio as gr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .shared_state import state  # Estado compartilhado
import io
from PIL import Image

# Global model state to save the trained model
global_model = {"model": None, "scaler": None, "columns": None}

# Train the model
def apply_ml(df, var_dep, ml_model_name, test_size):
    if df is None:
        df = state.get('new_df')  # Busca o DataFrame no estado compartilhado
    if df is None:
        raise ValueError("Nenhum DataFrame disponível para aplicação.")
    df = df.dropna()
    y = df[var_dep]
    X = df.drop(columns=[var_dep])

    # Remover a coluna "Índice" se ela existir
    if "Índice" in X.columns:
        X = X.drop(columns=["Índice"])

    # Normalizar os dados com MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Divisão em treino e teste com test_size ajustável
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    # Escolha do modelo
    if ml_model_name == "Linear Regression":
        model = LinearRegression()
    elif ml_model_name == "Ridge Regression":
        model = Ridge(alpha=0.5)
    elif ml_model_name == "Bayesian Ridge":
        model = BayesianRidge()
    elif ml_model_name == "Decision Tree":
        model = DecisionTreeRegressor()
    elif ml_model_name == "Random Forest":
        model = RandomForestRegressor()
    elif ml_model_name == "Support Vector Regression (SVR)":
        model = SVR()
    elif ml_model_name == "Neural Network (MLP)":
        model = MLPRegressor(max_iter=5000, tol=0.1, random_state=1)
    elif ml_model_name == "K-Neighbors Regressor":
        model = KNeighborsRegressor(n_neighbors=5)
    else:
        raise ValueError("Modelo de ML inválido.")

    # Treinamento e avaliação
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"Train R²: {train_r2}, Test R²: {test_r2}")

    # Save the trained model, scaler, and column names for prediction
    global_model["model"] = model
    global_model["scaler"] = scaler
    global_model["columns"] = df.drop(columns=[var_dep]).columns.tolist()

    # Gerar o gráfico
    plt.figure(figsize=(6, 4))
    plt.bar(["Treino", "Teste"], [train_r2, test_r2], color=["blue", "orange"])
    plt.title(f"Desempenho do Modelo: {ml_model_name} - Test Size: {test_size}")
    plt.ylabel("R²")
    plt.ylim(0, 1)  # Limite entre 0 e 1 para facilitar a visualização
    plt.tight_layout()

    # Salvar o gráfico em um buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convert the buffer to a PIL Image
    image = Image.open(buffer)

    return image

# Função para atualizar as opções de variáveis dependentes
def update_var_dep_dropdown(df):
    if df is None:
        df = state.get('new_df')  # Busca o DataFrame no estado compartilhado
    if df is None:
        return gr.update(choices=[])
    return gr.update(choices=df.columns.tolist())

def predict_new_values(*inputs):
    if global_model["model"] is None:
        return "O modelo ainda não foi treinado. Execute o modelo primeiro."
    
    # Reshape input to match model expectation
    new_data = [float(value) for value in inputs]
    new_data_scaled = global_model["scaler"].transform([new_data])
    
    # Predict
    prediction = global_model["model"].predict(new_data_scaled)[0]
    return f"Previsão: {prediction:.4f}"

# Função para criar a aba Machine Learning
def ml_tab(new_df_output):
    with gr.Tab("Machine Learning"):
        var_dep_dropdown = gr.Dropdown(choices=[], label="Variável Dependente")
        ml_model_dropdown = gr.Dropdown(
            choices=[
                "Linear Regression", "Ridge Regression", "Bayesian Ridge",
                "Decision Tree", "Random Forest", "Support Vector Regression (SVR)",
                "Neural Network (MLP)", "K-Neighbors Regressor"
            ],
            label="Modelo de Machine Learning"
        )
        test_size_slider = gr.Slider(minimum=0.1, maximum=0.5, step=0.05, value=0.3, label="Tamanho do Teste")
        submit_button = gr.Button("Executar Modelo")
        r2_graph_output = gr.Image(label="Gráfico de Desempenho")

        # Callback to execute the function
        submit_button.click(
            apply_ml,
            inputs=[new_df_output, var_dep_dropdown, ml_model_dropdown, test_size_slider],
            outputs=[r2_graph_output]
        )

        # Update dropdown options
        new_df_output.change(update_var_dep_dropdown, inputs=[new_df_output], outputs=[var_dep_dropdown])

        # Add prediction section
        gr.Markdown("### Previsão de Novos Valores")

        inputs = []
        if global_model["columns"]:  # Check if columns exist
            for col in global_model["columns"]:
                inputs.append(gr.Textbox(label=f"Valor para '{col}'"))
        else:
            gr.Markdown("O modelo ainda não foi treinado. Execute o modelo primeiro para realizar previsões.")

        predict_button = gr.Button("Prever Valores")
        prediction_output = gr.Textbox(label="Resultado da Previsão")

        # Predict only if inputs were generated
        if inputs:
            predict_button.click(predict_new_values, inputs=inputs, outputs=prediction_output)

    return locals()
