import pandas as pd
import json
import re
import logging
from unidecode import unidecode
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

class CIE10Retriever:
    def __init__(self, faiss_index_path, estructura_json_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = self._cargar_indice(faiss_index_path)
        self.estructura = self._cargar_estructura(estructura_json_path)
        self.normalizador = self._crear_normalizador()
    
    def _cargar_indice(self, path):
        try:
            return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            raise RuntimeError(f"Error cargando índice FAISS: {str(e)}")
    
    def _cargar_estructura(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _crear_normalizador(self):
        return {
            'sinonimos': {
                r"\b(trazo\s+sugestivo\b|fx|fract)": "fractura",
                r"\b(dcha|der)\b": "derecho",
                r"\b(izq|izqda)\b": "izquierdo",
                r"\b(tac)\b": "tomografía axial computarizada",
                r"\b(5to|quinto)\b": "5"
            },
            'reemplazos': {
                "QUEMADURA POR FRICCION": "QUEMADURA POR FRICCIÓN",
                "TRAZO SUGESTIVO DE FRACTURA": "FRACTURA"
            }
        }
    
    def normalizar_consulta(self, texto):
        texto = texto.upper()
        for patron, reemplazo in self.normalizador['reemplazos'].items():
            texto = texto.replace(patron, reemplazo)
        for patron, reemplazo in self.normalizador['sinonimos'].items():
            texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)
        return unidecode(texto).strip()
    
    def buscar(self, consulta, k=30, lambda_param=0.5, nivel_minimo=2):
        consulta_norm = self.normalizar_consulta(consulta)
        
        resultados = self.db.max_marginal_relevance_search(
            consulta_norm,
            k=k,
            lambda_param=lambda_param,
            filter={"nivel": {"$gte": nivel_minimo}}
        )
        
        return self._procesar_resultados(resultados)
    
    def _procesar_resultados(self, resultados):
        jerarquia = {
            'bloques': [],
            'categorias': [],
            'subcategorias': [],
            'detalles': []
        }
        
        for doc in resultados:
            metadata = doc.metadata
            nivel = metadata['nivel']
            entrada = {
                'codigo': metadata['codigo'],
                'descripcion': doc.page_content.split(":")[-1].strip(),
                'nivel': nivel,
                'ruta': metadata.get('ruta', ''),
                'score': self._calcular_score(nivel)
            }
            
            if nivel == 1:
                jerarquia['bloques'].append(entrada)
            elif nivel == 2:
                jerarquia['categorias'].append(entrada)
            else:
                jerarquia['subcategorias'].append(entrada)
            
            jerarquia['detalles'].append(entrada)
        
        return jerarquia
    
    def _calcular_score(self, nivel):
        # Prioriza subcategorías > categorías > bloques
        return {1: 0.3, 2: 0.6, 3: 1.0}.get(nivel, 0)
    
    def obtener_ruta_completa(self, codigo):
        for bloque in self.estructura['bloques']:
            if codigo == bloque['codigo']:
                return [bloque]
            for categoria in bloque['categorias']:
                if codigo == categoria['codigo']:
                    return [bloque, categoria]
                for subcat in categoria['subcategorias']:
                    if codigo == subcat['codigo']:
                        return [bloque, categoria, subcat]
        return []
    
    def validar_codigo(self, codigo):
        return any(self.obtener_ruta_completa(codigo))
    
    def mejores_resultados(self, resultados, top_n=3):
        return sorted(
            resultados['detalles'],
            key=lambda x: x['score'],
            reverse=True
        )[:top_n]

# Uso del sistema
if __name__ == "__main__":
    retriever = CIE10Retriever(
        faiss_index_path="faiss_principal_jerarquico",
        estructura_json_path="cie10_estructura_completa.json"
    )
    
    consulta = "Trazo sugestivo de fractura de la cabeza radial de codo derecho"
    #"Fractura del quinto metatarsiano del pie izquierdo"
    resultados = retriever.buscar(consulta)
    
    print("\nResultados organizados por jerarquía:")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    
    print("\nTop 3 resultados:")
    for i, res in enumerate(retriever.mejores_resultados(resultados), 1):
        print(f"{i}. Código: {res['codigo']} - Score: {res['score']:.2f}")
        print(f"   Descripción: {res['descripcion']}")
        print(f"   Ruta: {res['ruta']}\n")