import gradio as gr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import product
from .shared_state import state  # Importa o estado compartilhado

# OTIMIZA
def apply_transformation(data, transformation):
    if transformation == "exp" and (data > 50).any():
        return data
    if transformation == "direct":
        return data
    elif transformation == "inverse":
        return 1 / (data + 0.001)
    elif transformation == "log":
        return np.log(data + 0.001)
    elif transformation == "exp":
        return np.exp(data)
    elif transformation == "square":
        return data ** 2

def find_best_transformations(df, var_dep, ignore_dichotomous):
    if df is None:
        df = state.get('new_df')  # Busca o DataFrame no estado compartilhado
    if df is None:
        raise ValueError("Nenhum DataFrame disponível para otimização.")
    df = df.dropna()
    y = df[var_dep]
    X = df.drop(columns=[var_dep])
    
    # Remover a coluna "Índice" se ela existir
    if "Índice" in X.columns:
        X = X.drop(columns=["Índice"])
    
    dichotomous_columns = [col for col in X.columns if set(X[col].unique()).issubset({0, 1})]
    if ignore_dichotomous:
        X = X.drop(columns=dichotomous_columns)
    transformations = ["direct", "inverse", "log", "exp", "square"]
    scores = []
    for y_transformation in transformations:
        y_transformed = apply_transformation(y, y_transformation)
        for transformation_combo in product(transformations, repeat=X.shape[1]):
            X_transformed = X.copy()
            for i, transformation in enumerate(transformation_combo):
                column = X.iloc[:, i]
                X_transformed.iloc[:, i] = apply_transformation(column, transformation)
            model = LinearRegression()
            try:
                model.fit(X_transformed, y_transformed)
                predictions = model.predict(X_transformed)
                score = r2_score(y_transformed, predictions)
                scores.append((transformation_combo, y_transformation, score, model))
            except ValueError as e:
                if "Input X contains NaN" in str(e):
                    raise ValueError("O conjunto de dados apresenta valores nulos.") from e
                else:
                    raise e  # Propague outras exceções
                    
    scores = sorted(scores, key=lambda x: x[2], reverse=True)[:5]
    top_equations = []
    top_transformation_info = []
    top_scores = []
    for combo, y_trans, score, model in scores:
        equation = f"y = {model.intercept_:.4f} " + " ".join(
            [f"{'+' if coef >= 0 else '-'} ({abs(coef):.4f}) * {trans}" for coef, trans in zip(model.coef_, X.columns)]
        )
        transformation_info = {"y": y_trans}
        transformation_info.update(dict(zip(X.columns, combo)))
        top_equations.append([equation])
        top_transformation_info.append(transformation_info)
        top_scores.append([float(score)])
    return top_equations, top_transformation_info, top_scores

def update_var_dep_dropdown(df):
    if df is None:
        df = state.get('new_df')  # Busca o DataFrame no estado compartilhado
    if df is None:
        return gr.update(choices=[])
    return gr.update(choices=df.columns.tolist())

def otimiza_tab(new_df_output):
    with gr.Tab("OTIMIZA"):
        var_dep_dropdown = gr.Dropdown(
            choices=[],  # Inicialmente vazio
            label="Variável Dependente"
        )
        ignore_dichotomous_checkbox = gr.Checkbox(
            label="Ignorar Variáveis Dicotômicas", value=False
        )
        submit_button = gr.Button("Otimizar variáveis")

        with gr.Row():
            equations_output = gr.Dataframe(headers=["Equação"], label="Equações (Top 5)")
        with gr.Row():
            transformations_output = gr.JSON(label="Transformações Aplicadas (Top 5)")
        with gr.Row():
            scores_output = gr.Dataframe(headers=["R2_Score"], label="R2_Scores (Top 5)")

        # Callback para executar a função
        submit_button.click(
            find_best_transformations,
            inputs=[new_df_output, var_dep_dropdown, ignore_dichotomous_checkbox],
            outputs=[equations_output, transformations_output, scores_output]
        )

        # Atualiza o dropdown de variáveis dependentes quando o DataFrame é atualizado
        new_df_output.change(
            update_var_dep_dropdown,
            inputs=[new_df_output],
            outputs=[var_dep_dropdown]
        )

        return locals()
