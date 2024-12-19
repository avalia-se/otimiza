import gradio as gr
import pandas as pd
from modules.utils import create_new_dataframe_with_index_and_value_unit  # Importe a função necessária
from .shared_state import state  # Importa o estado compartilhado


# Variável global para armazenar o DataFrame original
original_df = None

# PLANILHA
def list_sheets(file):
    if file is None:
        return "Nenhum arquivo carregado.", []
    excel_file = pd.ExcelFile(file.name)
    sheet_names = excel_file.sheet_names
    return {"Abas disponíveis": sheet_names}, sheet_names

def load_sheet(file, sheet_name):
    if file is None or not sheet_name:
        return pd.DataFrame({"Erro": ["Carregue um arquivo e selecione uma aba."]}), "", []
    df = pd.read_excel(file.name, sheet_name=sheet_name)
    rows, cols = df.shape
    column_list = df.columns.tolist()
    return df, f"{rows} linhas e {cols} colunas", column_list

def update_column_selector(file, sheet_name):
    if file is None or not sheet_name:
        return gr.update(choices=[], value=[])
    df = pd.read_excel(file.name, sheet_name=sheet_name)
    columns = df.columns.tolist()
    return gr.update(choices=columns, value=columns)

def select_all_columns(select_all, choices):
    if select_all:
        return gr.update(value=choices)
    else:
        return gr.update(value=[])

def toggle_sheet_visibility(view_sheet):
    return gr.update(visible=view_sheet)

def toggle_operations_inputs(enable_operations):
    return gr.update(visible=enable_operations)

def add_new_variable(file, sheet_name, first_var, operation, second_var, new_var_name):
    if file is None or not sheet_name or not first_var or not second_var or not new_var_name:
        return pd.DataFrame({"Erro": ["Preencha todos os campos necessários."]})
    df = pd.read_excel(file.name, sheet_name=sheet_name)
    if first_var not in df.columns or second_var not in df.columns:
        return pd.DataFrame({"Erro": ["Variáveis selecionadas não existem no DataFrame."]})

    try:
        if operation == "Adição":
            df[new_var_name] = df[first_var] + df[second_var]
        elif operation == "Subtração":
            df[new_var_name] = df[first_var] - df[second_var]
        elif operation == "Multiplicação":
            df[new_var_name] = df[first_var] * df[second_var]
        elif operation == "Divisão":
            df[new_var_name] = df[first_var] / df[second_var]
    except ZeroDivisionError:
        return pd.DataFrame({"Erro": ["Divisão por zero detectada."]})
    except Exception as e:
        return pd.DataFrame({"Erro": [f"Erro ao realizar a operação: {str(e)}"]})

    return df

def update_variable_choices(file, sheet_name):
    if file is None or not sheet_name:
        return gr.update(choices=[]), gr.update(choices=[])
    df = pd.read_excel(file.name, sheet_name=sheet_name)
    return gr.update(choices=df.columns.tolist()), gr.update(choices=df.columns.tolist())

def update_dropdown(file):
    json_result, sheet_names = list_sheets(file)
    return json_result, gr.update(choices=sheet_names)

def update_columns(file, sheet_name):
    df, info, columns = load_sheet(file, sheet_name)
    return (
        df,
        info,
        columns,
        gr.update(choices=columns, value=[], interactive=True)
    )

# Função para restaurar o DataFrame ao estado inicial
def restore_dataframe():
    global original_df
    if original_df is not None:
        return original_df
    return pd.DataFrame()  # Retorna vazio se nada foi carregado


def finalize_dataframe(file, sheet_name, selected_columns, first_var, operation, second_var, new_var_name, add_index):
    if file is None or not sheet_name:
        return pd.DataFrame({"Erro": ["Carregue um arquivo e selecione uma aba."]})
    df = pd.read_excel(file.name, sheet_name=sheet_name)
    
    # Adiciona a nova variável, se necessário
    if new_var_name and first_var and second_var:
        try:
            if operation == "Adição":
                df[new_var_name] = df[first_var] + df[second_var]
            elif operation == "Subtração":
                df[new_var_name] = df[first_var] - df[second_var]
            elif operation == "Multiplicação":
                df[new_var_name] = df[first_var] * df[second_var]
            elif operation == "Divisão":
                df[new_var_name] = df[first_var] / df[second_var]
        except ZeroDivisionError:
            return pd.DataFrame({"Erro": ["Divisão por zero detectada."]})
        except Exception as e:
            return pd.DataFrame({"Erro": [f"Erro ao realizar a operação: {str(e)}"]})

    # Adiciona a nova variável às colunas selecionadas, se existir
    if new_var_name and new_var_name in df.columns:
        selected_columns.append(new_var_name)
    
    if selected_columns:
        df = df[selected_columns]
    
    # Adiciona índice na primeira coluna, se necessário
    if add_index:
        df.insert(0, 'Índice', range(1, len(df) + 1))
    
    return df

def planilha_tab():
    # Criação da aba
    with gr.Tab("PLANILHA"):
        # Upload do arquivo Excel
        with gr.Row():
            excel_file = gr.File(label="Carregue sua planilha Excel")
        with gr.Row():
            list_button = gr.Button("Listar Abas")

        # Exibição das abas disponíveis e seleção de aba
        with gr.Row():
            sheet_output = gr.JSON(label="Abas disponíveis")
            sheet_dropdown = gr.Dropdown(label="Selecione uma aba")

        # Botão para carregar aba e exibir colunas
        with gr.Row():
            load_button = gr.Button("Carregar Aba")

        # Informações e exibição de colunas da aba carregada
        with gr.Row():
            sheet_info = gr.Text(label="Informação de Linhas e Colunas")
            sheet_columns = gr.JSON(label="Lista de Colunas", visible=False)
            view_sheet_checkbox = gr.Checkbox(label="Visualizar a planilha", value=False)

        # Exibição do conteúdo da aba selecionada
        with gr.Row():
            sheet_content = gr.Dataframe(
                label="Conteúdo da Aba Selecionada",
                visible=False
            )

        # CheckboxGroup para seleção de colunas
        with gr.Row():
            column_selector = gr.CheckboxGroup(
                label="Selecione colunas para análise",
                choices=[],  # Inicia vazio
                interactive=True
            )

        # Checkbox para adicionar índice na primeira coluna
        with gr.Row():
            add_index_checkbox = gr.Checkbox(label="Adicionar índice na primeira coluna", value=False)

        # Checkbox para habilitar operações com variáveis
        operations_checkbox = gr.Checkbox(label="Operações com variáveis", value=False)

        # Elementos relacionados às operações, inicialmente invisíveis
        with gr.Row(visible=False) as operations_inputs:
            variable_name_textbox = gr.Textbox(
                label="Nome da nova variável",
                placeholder="Digite o nome da nova variável",
                interactive=True
            )
            first_variable_dropdown = gr.Dropdown(
                label="Selecione a primeira variável",
                choices=[],
                interactive=True
            )
            operation_dropdown = gr.Dropdown(
                label="Selecione a operação",
                choices=["Adição", "Subtração", "Multiplicação", "Divisão"],
                interactive=True
            )
            second_variable_dropdown = gr.Dropdown(
                label="Selecione a segunda variável",
                choices=[],
                interactive=True
            )

        # Associações de eventos
        list_button.click(update_dropdown, inputs=[excel_file], outputs=[sheet_output, sheet_dropdown])
        load_button.click(update_columns, inputs=[excel_file, sheet_dropdown], outputs=[sheet_content, sheet_info, sheet_columns])
        load_button.click(update_column_selector, inputs=[excel_file, sheet_dropdown], outputs=[column_selector])
        view_sheet_checkbox.change(toggle_sheet_visibility, inputs=[view_sheet_checkbox], outputs=[sheet_content])
        operations_checkbox.change(toggle_operations_inputs, inputs=[operations_checkbox], outputs=[operations_inputs])

        apply_operations_button = gr.Button("Criar Nova Variável")
        new_df_output = gr.Dataframe(label="Novo DataFrame", interactive=True)
        apply_operations_button.click(add_new_variable, inputs=[excel_file, sheet_dropdown, first_variable_dropdown, operation_dropdown, second_variable_dropdown, variable_name_textbox], outputs=[new_df_output])
        load_button.click(update_variable_choices, inputs=[excel_file, sheet_dropdown], outputs=[first_variable_dropdown, second_variable_dropdown])

        # Botão para finalizar o DataFrame
        finalize_button = gr.Button("Finalizar DataFrame")
        finalize_button.click(finalize_dataframe, inputs=[excel_file, sheet_dropdown, column_selector, first_variable_dropdown, operation_dropdown, second_variable_dropdown, variable_name_textbox, add_index_checkbox], outputs=[new_df_output])

        # Adicionar botões "Restaurar" e "Limpar"
        with gr.Row():
            restore_button = gr.Button("Restaurar")
            clear_button = gr.ClearButton(components=[excel_file, sheet_output, sheet_dropdown, sheet_info, sheet_columns, view_sheet_checkbox, sheet_content, column_selector, add_index_checkbox, operations_checkbox, variable_name_textbox, first_variable_dropdown, operation_dropdown, second_variable_dropdown, new_df_output], 
                                          value="Limpar")

        # Associações de eventos para os botões "Restaurar"
        restore_button.click(restore_dataframe, outputs=[new_df_output])

    return locals(), new_df_output