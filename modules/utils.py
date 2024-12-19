# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:36:45 2024

@author: fernando.schwartzer
"""
import pandas as pd
from .shared_state import state

def create_new_dataframe_with_index_and_value_unit(file, sheet_name, selected_columns, add_index, calculate_unit_value, col_value, col_area):
    if file is None or not sheet_name or not selected_columns:
        return pd.DataFrame({"Erro": ["Carregue um arquivo, selecione uma aba e colunas."]})
    
    # Carrega o DataFrame da aba especificada
    df = pd.read_excel(file.name, sheet_name=sheet_name)
    
    # Seleciona apenas as colunas especificadas
    new_df = df[selected_columns]
    
    # Adiciona um índice se necessário
    if add_index:
        new_df.insert(0, "Índice", range(1, len(new_df) + 1))
    
    # Renomeia e formata a coluna de valor total
    if col_value:
        new_df.rename(columns={col_value: "Valor Total"}, inplace=True)
        new_df["Valor Total"] = new_df["Valor Total"].round(2)
        cols = new_df.columns.tolist()
        cols.insert(1, cols.pop(cols.index("Valor Total")))
        new_df = new_df[cols]
    
    # Calcula o valor unitário se necessário
    if calculate_unit_value and col_value and col_area:
        try:
            new_df["Valor Unitário"] = (new_df["Valor Total"] / new_df[col_area]).round(2)
            cols = new_df.columns.tolist()
            cols.insert(2, cols.pop(cols.index("Valor Unitário")))
            new_df = new_df[cols]
        except ZeroDivisionError:
            new_df["Valor Unitário"] = "Erro: Divisão por zero"
        except KeyError:
            new_df["Valor Unitário"] = "Erro: Coluna inválida"
    
    # Salva o novo DataFrame no estado compartilhado
    new_df.columns = new_df.columns.map(str)
    state['new_df'] = new_df  
    return new_df