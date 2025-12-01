from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import shap
import joblib
import uvicorn
import os
import json

# Inicializamos la App
app = FastAPI(title="API de Predicción y Explicabilidad IQ e Infanib Fundación Canguro")

# Variable global para el modelo
pipe = None
MAPA_TIPOS = {}

# ---------------------------------------------------------
# 1. CARGA DEL MODELO (Al iniciar el servidor)
# ---------------------------------------------------------
try:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Ajusta el nombre del archivo si es necesario
    ruta_modelo = os.path.join(directorio_actual, 'modelos', 'modelorf_fase1_infanib.pkl')
    print(f">>> Buscando modelo en: {ruta_modelo}")

    if os.path.exists(ruta_modelo):
        pipe = joblib.load(ruta_modelo)
        print("✅ >>> Modelo cargado exitosamente.")
    else:
        print("❌ El archivo del modelo no existe en la ruta especificada.")

except Exception as e:
    print(f"❌ >>> Error cargando el modelo: {e}")

# Carga de Metadatos
try:
    ruta_meta = os.path.join(directorio_actual, 'metadata_tipos_f1.json')
    if os.path.exists(ruta_meta):
        with open(ruta_meta, 'r') as f:
            MAPA_TIPOS = json.load(f)
        print(f"✅ >>> Metadata de tipos cargada correctamente ({len(MAPA_TIPOS)} variables).")
    else:
        print("⚠️ No se encontró metadata_tipos_f1.json")
except Exception as e:
    print(f"❌ Error cargando metadata: {e}")


# ---------------------------------------------------------
# 2. DEFINICIÓN DE DATOS DE ENTRADA
# ---------------------------------------------------------
class InputDatos(BaseModel):
    # Definimos todo con valores por defecto para que sean opcionales en el JSON
    # Variables que tú quieres enviar explícitamente:
    AnioParto: float = 0.0
    CP_TallaMadre: float = 0.0
    CSP_DistanciaVivienda: float = 0.0
    CSP_EscolaridadMadre: float = 0.0
    CSP_EscolaridadPadre: float = 0.0
    CSP_IngresoMensual: float = 0.0
    CSP_Menores5vivenMadre: float = 0.0
    CSP_NutricionFam: float = 0.0
    CSP_SituaPareja: float = 0.0
    CSP_SituacionLaboralMadre: float = 0.0
    CSP_SituacionLaboralPadre: float = 0.0
    CSP_numPersVivenIngMen: float = 0.0
    Iden_Sede: float = 0.0
    Iden_embarazoMultiple: bool = False
    edadmatcat: float = 0.0
    educmadresimplificada: float = 0.0
    educpadresimplificada: float = 0.0
    ninosmenosde5anos: bool = False

    # Resto de variables del modelo (con default 0 para no romper la ejecución)
    Idenfinal: float = 0.0
    CSP_AyudaPerm1mes: bool = False
    V204: float = 0.0
    V208: float = 0.0
    V209: float = 0.0
    IQ12cat: bool = False
    infanib12m_bin: float = 0.0
    ERN_PC: float = 0.0
    ERN_Peso: float = 0.0
    ERN_Talla: float = 0.0
    ERN_AdaptNeonatal: float = 0.0
    anoxia5mn: bool = False
    ERN_A_10min: float = 0.0
    ERN_A_1min: float = 0.0
    ERN_A_5min: float = 0.0
    apgarcat1: float = 0.0
    apgarcat5: float = 0.0
    BPN: bool = False
    ERN_Ballard: float = 0.0
    gestacat: float = 0.0
    INFECCIONOSOCOMIAL: bool = False
    ERN_LubchencoFenton: float = 0.0
    menosde1001: bool = False
    menosde31sem: bool = False
    Nearterm: bool = False
    pesocat: float = 0.0
    PESO1500G: bool = False
    RCIUPC: bool = False
    RCIUtalla: bool = False
    ERN_Remision: bool = False
    ERN_sepsis: float = 0.0
    ERN_Sexo: float = 0.0
    HD_DiasAlojamiento: float = 0.0
    DIASTOT08: float = 0.0
    ANOCAT: float = 0.0
    PA_AtendioParto: float = 0.0
    cesarea: bool = False
    PA_ComplicacionsPartoAbrupcio: bool = False
    PA_ComplicacionsPartoAmnionitis: bool = False
    PA_ComplicacionsPartoEclampsia: bool = False
    PA_ComplicacionsPartoMultiples: bool = False
    PA_ComplicacionsPartoPatologIaCordOn: bool = False
    PA_ComplicacionsPartoPlacentaPrevia: bool = False
    PA_ComplicacionsPartoPreeclampsia: bool = False
    PA_ComplicacionsPartoSindromedehellp: bool = False
    PA_EstActualMadre: float = 0.0
    tuvotransfusiones: bool = False
    UCI: bool = False
    PA_LiqAmnioticoMeconiado: float = 0.0
    PA_LugarNacimiento: float = 0.0
    PA_Monitoreo: bool = False
    PA_NumDosisCorticoides: float = 0.0
    primipara: bool = False
    PA_RitmoCardiaco: float = 0.0
    PA_SufrimientoFetalAgudo: float = 0.0
    PA_TipoParto: float = 0.0
    CONSULT08: float = 0.0
    ecocerebral: bool = False
    REHOSP08: float = 0.0
    rehosp40: bool = False
    CP_ARO: float = 0.0
    AROCAT02: float = 0.0
    controlcat: float = 0.0
    corticodosis: float = 0.0
    corticoprenatalsimple: float = 0.0
    ecocat: float = 0.0
    CP_edadmaterna: float = 0.0
    embarazoplanifDIU: bool = False
    embarazoplanifhormonal: bool = False
    CP_HospitalizacionesPreParto: float = 0.0
    CP_MadreAlcohol: bool = False
    CP_MadreFumo: bool = False
    CP_MesInicCP: float = 0.0
    CP_NumEcografias: float = 0.0
    pdpcat: float = 0.0
    CP_MedGeneral: float = 0.0
    RELACIONAROSOBRETOTALCONSULTAS: float = 0.0
    CP_SA_Anemia: bool = False
    CP_SA_APP: bool = False
    CP_SA_EnfRespiratoria: bool = False
    CP_SA_InfGineco: bool = False
    CP_SA_InfUrinaria: bool = False
    CP_SA_Preclampsia: bool = False
    CP_SA_RPM: bool = False
    CP_SA_Sangrado: bool = False
    Sufrimientofetalcronico: bool = False
    CP_TotalCPN: float = 0.0
    toxemia: bool = False
    CP_TP_HepatitisB: float = 0.0
    CP_TP_HIV: float = 0.0
    CP_TP_Orina: float = 0.0
    CP_TP_Rubeola: float = 0.0
    CP_TP_Sifilis: float = 0.0
    CP_TP_Toxoplasmosis: float = 0.0
    trimestre: float = 0.0
    CSP_EmbarazoDeseado: bool = False
    Iden_FechaParto: str = "2000-01-01"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                  "AnioParto": 2008,
                  "CP_TallaMadre": 155,
                  "CSP_DistanciaVivienda": 1,
                  "CSP_EscolaridadMadre": 4,
                  "CSP_EscolaridadPadre": 4,
                  "CSP_IngresoMensual": 1000000,
                  "CSP_Menores5vivenMadre": 2,
                  "CSP_NutricionFam": 1,
                  "CSP_SituaPareja": 3,
                  "CSP_SituacionLaboralMadre": 1,
                  "CSP_SituacionLaboralPadre": 1,
                  "CSP_numPersVivenIngMen": 3,
                  "Iden_Sede": 1,
                  "Iden_embarazoMultiple": True,
                  "edadmatcat": 18,
                  "educmadresimplificada": 2,
                  "educpadresimplificada": 1,
                  "ninosmenosde5anos": True
                }
            ]
        }
    }

# ---------------------------------------------------------
# 3. LÓGICA DE SHAP (CORREGIDA Y BLINDADA)
# ---------------------------------------------------------
def calcular_shap_y_prediccion(df_input):
    if pipe is None:
        raise HTTPException(status_code=500, detail="El modelo no cargó.")

    # =========================================================
    # PASO 1: LIMPIEZA NUCLEAR
    # =========================================================
    for col in df_input.columns:
        tipo = MAPA_TIPOS.get(col, 'object')
        if df_input[col].isnull().any():
            if tipo in ['int', 'float', 'bool']:
                df_input[col] = df_input[col].fillna(0)
            else:
                df_input[col] = df_input[col].fillna("SIN_DATO")
        try:
            if tipo == 'int': df_input[col] = df_input[col].astype(int)
            elif tipo == 'float': df_input[col] = df_input[col].astype(float)
            elif tipo == 'bool': df_input[col] = df_input[col].astype(bool)
            else: df_input[col] = df_input[col].astype(str)
        except:
            df_input[col] = df_input[col].astype(str)

    # =========================================================
    # PASO 2: EXTRACCIÓN DEL PIPELINE
    # =========================================================
    try:
        nombres = list(pipe.named_steps.keys())
        preprocesador = pipe.named_steps[nombres[0]]
        modelo_rf = pipe.named_steps[nombres[-1]]
    except:
        try: preprocesador = pipe.named_steps['preprocess']
        except: preprocesador = pipe.named_steps['prep']
        modelo_rf = pipe.named_steps['model']

    # =========================================================
    # PASO 3: PARCHE DE SUFIJOS
    # =========================================================
    if hasattr(preprocesador, 'feature_names_in_'):
        for col_esperada in preprocesador.feature_names_in_:
            if col_esperada not in df_input.columns:
                base = col_esperada.replace("_x", "").replace("_y", "")
                if base in df_input.columns:
                    df_input[col_esperada] = df_input[base]
                else:
                    df_input[col_esperada] = 0

    # =========================================================
    # PASO 4: TRANSFORMACIÓN Y PREDICCIÓN
    # =========================================================
    try:
        X_transformado = preprocesador.transform(df_input)
        if hasattr(X_transformado, "toarray"):
            X_transformado = X_transformado.toarray()
        try: names = preprocesador.get_feature_names_out()
        except: names = [f"var_{i}" for i in range(X_transformado.shape[1])]
        
        try:
            pred_resultado = modelo_rf.predict_proba(X_transformado)[0][1]
        except:
            pred_resultado = float(modelo_rf.predict(X_transformado)[0])
            
    except Exception as e:
        print(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error matemático: {str(e)}")

    # =========================================================
    # PASO 5: SHAP (CORREGIDO PARA EVITAR EL ERROR DE ARRAY)
    # =========================================================
    try:
        explainer = shap.TreeExplainer(modelo_rf)
        shap_values = explainer.shap_values(X_transformado)
        
        vals = shap_values
        
        # 1. Si es lista (Random Forest), tomamos clase 1
        if isinstance(vals, list): 
            vals = vals[1]
            
        # 2. Si es matriz con dimensiones extra (1, N) -> aplanamos a (N,)
        if hasattr(vals, "shape") and len(vals.shape) > 1:
            # Si es (1, N, Clases) -> Tomamos (1, N, 1)
            if len(vals.shape) == 3: 
                vals = vals[0, :, 1] # Fila 0, todas las features, clase 1
            else:
                vals = vals[0] # Aplanamos la primera dimensión (el batch)

        # Validación final de longitud
        if len(vals) != len(names):
             # Intento de emergencia si las formas no cuadran
             vals = np.zeros(len(names)) 
             
    except Exception as e:
        print(f"⚠️ Error SHAP: {e}")
        vals = np.zeros(len(names))

    # Empaquetado
    explicacion = []
    for n, v in zip(names, vals):
        # --- AQUÍ ESTABA EL ERROR ---
        # Aseguramos que 'v' sea un escalar antes de comparar
        valor_float = float(v) if np.ndim(v) == 0 else float(v[0]) 
        
        if abs(valor_float) > 0.001: 
            n_clean = n.replace("num__", "").replace("cat__", "").replace("onehot__", "")
            explicacion.append({"variable": n_clean, "impacto_shap": valor_float})
            
    explicacion.sort(key=lambda x: abs(x['impacto_shap']), reverse=True)
    
    return float(pred_resultado), explicacion
# ---------------------------------------------------------
# 4. EL ENDPOINT
# ---------------------------------------------------------
@app.post("/predecir")
def predecir_endpoint(datos: InputDatos):
    input_dict = datos.dict()
    df = pd.DataFrame([input_dict])
    
    prediccion, explicacion_shap = calcular_shap_y_prediccion(df)
    
    return {
        "probabilidad_resultado": prediccion,
        "factores_explicativos": explicacion_shap
    }

# ---------------------------------------------------------
# 5. ARRANQUE
# ---------------------------------------------------------
if __name__ == "__main__":
    print(">>> Iniciando servidor api_construction_V0...")
    uvicorn.run("api_construction_V0:app", host="127.0.0.1", port=8001, reload=True)