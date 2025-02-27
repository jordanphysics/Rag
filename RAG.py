import pandas as pd
import json
import os
import random
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq


# Configurar embeddings con all-MiniLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cargar el índice FAISS previamente guardado
loaded_db = FAISS.load_local("faiss_principal", embeddings=embeddings, allow_dangerous_deserialization=True)
print("Índice cargado correctamente:", loaded_db)

# Función para buscar contexto relevante con MMR
def obtener_contexto_mmr(consulta, k=50, lambda_param=0.5):
    """
    Busca el contexto más relevante utilizando Maximal Marginal Relevance (MMR).
    """
    try:
        # Realizar búsqueda MMR
        resultados = loaded_db.max_marginal_relevance_search(consulta, k=k, lambda_param=lambda_param )
        
        if not resultados:
            return "No se encontró contexto relevante."
        
        # Concatenar resultados en un solo texto
        contexto = "\n\n".join([doc.page_content for doc in resultados])
        return contexto
    except Exception as e:
        return f"Error al buscar contexto: {str(e)}"

# Configurar cliente Groq
client = Groq(api_key="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")

# Función para generar respuesta utilizando el LLM de Groq
def generar_respuesta_groq(consulta, contexto, temperature=0.0):
    """
    Genera una respuesta utilizando el modelo LLM de Groq, basándose en la consulta y el contexto recuperado.
    """
    try:
        # Prompt refinado para asignar códigos CIE-10
        prompt = (
            "Eres un asistente médico experto en codificación CIE-10. "
            "Tu tarea es asignar el código CIE-10 correcto basándote en la consulta del usuario y en el contexto proporcionado.\n\n"
            "El contexto consiste en fragmentos de un corpus que relaciona códigos y descripciones médicas.\n"
            "Responde únicamente en el siguiente formato:\n\n"
            "Código: <código> - Descripción: <descripción>\n\n"
            "Si no puedes determinar un código con certeza, responde exactamente: 'No se pudo determinar el código CIE-10.'\n\n"
            f"Consulta: {consulta}\n\n"
            f"Contexto:\n{contexto}"
        )
        
        # Generar respuesta con Groq
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Modelo adecuado para consultas médicas
            temperature=temperature,
            messages=[
                {"role": "system", "content": "Eres un asistente médico experto."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar la respuesta: {str(e)}"

# Ejemplo de búsqueda y generación de respuesta
consulta_ejemplo ="Trazo sugestivo de fractura de cúpula radial de codo derecho"

#"Trauma hombro derecho" S407
#"Fractura de 5to metatarsiano de pie izquierdo" S923
#"Trazo sugestivo de fractura de tuberosidad mayor de húmero derecho" S424
#"Trazo sugestivo de fractura de cúpula radial de codo derecho"
#fractura de la cabeza del radio del codo derecho s52.5
#"TRAUMA PIE IZQUIERDO" s94.1
#"Trazo sugestivo de fractura de radio distal de muñeca derecha"S525. k= 30
contexto = obtener_contexto_mmr(consulta_ejemplo, k=30, lambda_param=0.7) # Hasta el momento es muy bueno con k=30 , lambda= 0.7
print("Contexto recuperado:\n", contexto)
respuesta = generar_respuesta_groq(consulta_ejemplo, contexto, temperature=0.0)
print("\nRespuesta generada:\n", respuesta)