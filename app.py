import gradio as gr
from modules.planilha import planilha_tab  # Importe apenas planilha_tab, save_new_df é usado internamente no módulo
from modules.otimiza import otimiza_tab
from modules.ml import ml_tab

# Cria o app principal
with gr.Blocks() as app:
    with gr.Tabs():     
        # Adiciona abas importadas
        planilha_ui, new_df_output = planilha_tab()  # Agora captura o new_df_output retornado
        otimiza_ui = otimiza_tab(new_df_output)     # Passa new_df_output para a aba OTIMIZA
        ml_ui = ml_tab(new_df_output)     # Passa new_df_output para a aba OTIMIZA

# Executa o app
if __name__ == "__main__":
    app.launch(share=False)


