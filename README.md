# Redes Neuronales III — Karen

Clasificación de dígitos de 8 por 8 píxeles con una red densa 64-128-64-10. Cinco configuraciones y tres semillas, selección por F1 macro en desarrollo y prueba aislada.

Este proyecto conserva el código que produjo las figuras y cifras del informe. Los entrenamientos se ejecutaron localmente en CPU, con la prueba aislada de la selección. El cuaderno es autónomo y se puede abrir en Google Colab; no se atribuye a Colab una ejecución que no se hizo allí.

## Archivos

- Karen_Redes_Informe.pdf: informe de entrega en PDF.
- Karen_Redes.ipynb: cuaderno con salida guardada.
- Script .py: ejecución desde línea de comandos.
- resultados.json y CSV/PNG: mediciones y evidencias de la ejecución.
- main.tex y archivo Overleaf .zip: fuente editable del informe.

## Repetir el experimento

Instala las dependencias con python -m pip install -r requirements.txt y ejecuta:

    python experimento_redes.py --persona Karen --salida resultados

En Colab, instala las dependencias que falten y ejecuta el cuaderno de arriba abajo. La corrida reconstruye la carpeta resultados. Para comparar, revisa los CSV/JSON sin modificar la partición de prueba.

Informe en Overleaf: https://www.overleaf.com/project/6aca7060478245289173ebd0

Repositorio: https://github.com/sofiadelcal/karen-redes-neuronales-iii-digitos
