Universidad de los Andes - Maestría en Ingeniería de la Información Profesor: Fabian Camilo Peña, Juan Pablo Reyes 
Fecha entrega: 30 de noviembre de 2025 Estudiantes: Edgar Javier Toquica Gahona
La entrega del proyecto incluye un notebook que hace uso del archivo "KMC-70k-93-2024-Obes 19-conVel-DATA-SPSS-20250322V1.csv" como dataset, el cual está en: https://drive.google.com/drive/folders/1Vx-NhoxvPNpHgcgt418aHuilTbIxG8R2?usp=drive_link
Este enlace se desactivará en una semana.
1. Cargue el dataset
2. Ejecute el notebook
3. Se adjunta en Bloque Neon informe ejecutivo con las conclusiones generales del ejercicio.
4.Para ejecutar el API:
- Descargue todos los archivos de este git en el mismo repositorio del notebook.
- En el directorio donde descargues el proyecto crea una nueva carpeta llamada "modelos"
- Guarda el modelo de prueba en esa carpeta. Descárgalo desde: https://drive.google.com/drive/folders/1Vx-NhoxvPNpHgcgt418aHuilTbIxG8R2?usp=drive_link
- Abre Anaconda Prompt.
- Activa tu entorno.
- Ubícate en la carpeta donde descargues el proyecto
- Enciende el motor del API: python api_construction_V0.py 
- EL API corre en http://127.0.0.1:8001
- Ya estás listo para ejecutar el archivo de PowerBI.
5. Si Power BI te da un error de que no encuentra el archivo CSV:
- Ve a Transformar Datos.
- En el panel derecho ("Pasos aplicados"), haz clic en el engranaje ⚙️ del paso Origen (Source).
- Busca el nuevo archivo datos_prueba_powerbi.xlsx.
- Dale Aceptar, luego "Cerrar y Aplicar".
