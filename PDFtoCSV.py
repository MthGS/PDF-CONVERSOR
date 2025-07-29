import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from io import StringIO # Usado para ler a string CSV no Pandas

# --- Configurações da Página ---
st.set_page_config(layout="wide")

# Tenta carregar a chave do Streamlit Secrets (ambiente de deploy)
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]

# Se não encontrar (está rodando localmente), carrega do arquivo .env
except (KeyError, FileNotFoundError):
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")
    
# Configura a API uma única vez, se a chave foi encontrada
if API_KEY is None or API_KEY == "":
    st.error("Chave da API do Google não configurada. Adicione-a nos Secrets do Streamlit.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- Funções de Extração e Conversão ---

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(image_file):
    """Extrai texto de um arquivo de imagem usando OCR."""
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

def get_gemini_response(text, prompt):
    """Envia o texto extraído e um prompt para o Gemini e retorna a resposta."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Ou 'gemini-pro'
        response = model.generate_content([prompt, text])
        return response.text
    except Exception as e:
        st.error(f"Erro ao chamar a API do Gemini: {e}")
        return None

# --- Layout Principal do Dashboard ---
def main():
    st.title("🤖 Conversor Inteligente de Arquivos para CSV")
    st.markdown("Faça o upload de um arquivo **PDF** ou **Imagem** contendo dados tabulares e converta-o para CSV com a ajuda da IA.")

    # Colunas para layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Faça o Upload do Arquivo")
        uploaded_file = st.file_uploader("Arraste ou selecione seu arquivo", type=['pdf', 'png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            # Botão para iniciar a conversão
            if st.button("Converter para CSV", use_container_width=True, type="primary"):
                with st.spinner('Processando o arquivo... Por favor, aguarde.'):
                    raw_text = ""
                    # Verifica o tipo de arquivo e extrai o texto
                    if uploaded_file.type == "application/pdf":
                        raw_text = extract_text_from_pdf(uploaded_file)
                    else: # Imagem
                        raw_text = extract_text_from_image(uploaded_file)

                    if raw_text:
                        #st.text_area("Texto Extraído (Bruto)", raw_text, height=200) # Opcional: mostrar texto bruto

                        # Prompt para o Gemini
                        prompt = """
                        Você é um assistente especialista em análise de dados.
                        Sua tarefa é analisar o texto fornecido abaixo, que foi extraído de um documento.
                        Identifique a estrutura de dados principal, que provavelmente é uma tabela ou uma lista de registros.
                        Converta esses dados para um formato CSV limpo e bem estruturado.

                        Regras importantes:
                        1. A primeira linha da sua resposta DEVE ser o cabeçalho (header) do CSV.
                        2. Use vírgula (,) como delimitador.
                        3. Ignore qualquer texto introdutório, rodapés, ou informações que não façam parte dos dados tabulares principais.
                        4. Não inclua qualquer explicação, apenas o resultado em formato CSV.
                        5. Se o texto não contiver dados tabulares claros, retorne uma mensagem de erro indicando isso.
                        """

                        gemini_response = get_gemini_response(raw_text, prompt)

                        if gemini_response:
                            st.session_state['csv_data'] = gemini_response
                            st.session_state['file_name'] = f"{os.path.splitext(uploaded_file.name)[0]}.csv"
                    else:
                        st.error("Não foi possível extrair texto do arquivo.")

    with col2:
        st.subheader("2. Resultado e Download")
        if 'csv_data' in st.session_state:
            csv_string = st.session_state['csv_data']
            try:
                # Usa StringIO para que o pandas leia a string como se fosse um arquivo
                df = pd.read_csv(StringIO(csv_string))
                st.dataframe(df)
                
                # Botão de Download
                st.download_button(
                    label="📥 Baixar CSV",
                    data=csv_string.encode('utf-8'),
                    file_name=st.session_state['file_name'],
                    mime='text/csv',
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Não foi possível processar a resposta da IA como um CSV. Verifique o resultado abaixo.")
                st.text_area("Resposta recebida da IA", csv_string, height=300)

if __name__ == "__main__":
    main()




