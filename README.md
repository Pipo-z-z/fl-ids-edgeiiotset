# Aprendizaje federado para IDS en redes IoT de borde

Código del artículo **"Aprendizaje federado para mejorar la capacidad de detección de los sistemas de detección de intrusiones en redes IoT de borde, 2026"** (Examen Parcial, Inteligencia Artificial II, UNI-FIEE). Autor: Carlos Ccama. El artículo y la sistematización están en [`docs/`](docs/).

El experimento compara cuatro formas de entrenar el mismo detector de intrusiones (un perceptrón multicapa), usando tráfico real de dispositivos IoT/IIoT:

| Esquema | Qué representa |
|---|---|
| **Local (aislado)** | Pre-test: cada dispositivo entrena solo con su propio tráfico |
| **FedAvg (no-IID)** | Post-test: aprendizaje federado con datos heterogéneos entre dispositivos |
| FedAvg (IID) y FedProx (μ = 0,01 y 0,1) | Variantes de la variable independiente (heterogeneidad y estrategia de agregación) |
| Centralizado | Cota superior: todos los datos en un servidor |

Todo el procesamiento parte de las **capturas PCAP crudas** de Edge-IIoTset (Ferrag et al., 2022). No se usan los CSV preprocesados del dataset.

## Estructura

```
fl_ids/
  flows.py       extractor propio de flujos bidireccionales (PCAP -> 31 características)
  dataset.py     inventario de PCAP, etiquetado de flujos de ataque, dataset final
  partition.py   división entrenamiento/prueba, reparto IID / no-IID, divergencia JS
  model.py       MLP + Adam + término proximal (FedProx), implementado en NumPy
  federated.py   entrenamiento centralizado, local, FedAvg/FedProx y métricas
  experiment.py  orquestación (5 semillas x 6 esquemas)
  analysis.py    tablas, pruebas de Wilcoxon, registro de observación y figuras
notebooks/
  01_extraccion_flujos.ipynb     PCAP crudos -> data/processed/flows_labeled.parquet
  02_experimento_federado.ipynb  experimento, figuras y tablas -> results/
data/raw/        aquí va el dataset descomprimido (no se sube a GitHub)
results/         tablas CSV y figuras que genera el notebook 02
docs/            artículo (PDF), sistematización con matrices (PDF) y lista de fuentes
```

## Instalación (Windows, PowerShell)

```powershell
cd fl-ids-edgeiiotset
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Funciona con Python 3.10 o superior, en CPU (no requiere GPU, Npcap ni Wireshark). En Linux/macOS es igual, activando el entorno con `source .venv/bin/activate`.

## Datos

El repositorio incluye `data/processed/flows_labeled.parquet` (flujos etiquetados que usa el notebook 02), de modo que el experimento puede reproducirse sin volver a procesar los PCAP. Para regenerarlo desde cero:


1. Descarga **Edge-IIoTset** desde [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) o desde su espejo en Kaggle (zip de ~1,5 GB).
2. Descomprímelo dentro de `data/raw/` (el dataset se distribuye con licencia CC BY-NC-SA 4.0 para uso académico; cita a Ferrag et al., 2022). Solo se usan las carpetas de **tráfico normal** (`Normal traffic/<Sensor>/<Sensor>.pcap`) y de **tráfico de ataque** (`Attack traffic/*.pcap`). El código las busca de forma recursiva, así que no importa el nombre exacto de las carpetas superiores.

## Ejecución

Abre los notebooks en VS Code o Jupyter y ejecútalos en orden:

1. `01_extraccion_flujos.ipynb`: extrae y etiqueta los flujos. Tarda alrededor de 1–2 min por cada millón de paquetes y guarda caché por archivo, así que se puede interrumpir y retomar.
2. `02_experimento_federado.ipynb`: ejecuta el experimento completo (3–6 min por semilla en una laptop) y deja en `results/` todas las tablas y figuras.

## Decisiones metodológicas

**Extracción de flujos.** Los flujos se agrupan por 5-tupla bidireccional. Cada flujo termina por inactividad (60 s), al cumplir una ventana de 120 s o al final de la captura. La ventana divide las conexiones MQTT y Modbus/TCP persistentes, que duran horas, en varias observaciones. Por cada flujo se calculan 31 características: protocolo, duración, paquetes y bytes por sentido, estadísticas de tamaño de paquete y de tiempo entre llegadas, banderas TCP, tasas y relación de bajada/subida. Las capturas se truncan a 1 millón de paquetes.

**Sin identificadores.** Las IP, los puertos y las marcas de tiempo se conservan solo como metadatos y no entran al modelo. Así el modelo aprende el comportamiento del tráfico y no la identidad de los equipos de la maqueta.

**Etiquetado.** Un flujo de una captura de ataque se etiqueta como ataque si alguno de sus extremos es una IP que no aparece en ninguna captura de tráfico normal: el atacante, o las IP falsificadas en las inundaciones DDoS. Los demás flujos de esas capturas son tráfico de fondo y se descartan. La tabla `results/diagnostico_etiquetado.csv` permite verificar esta regla por ataque.

**Limpieza.** Se eliminan los vectores duplicados dentro de cada clase, para que no queden filas idénticas a la vez en entrenamiento y prueba. Una clase que queda con menos de 50 vectores únicos conserva duplicados hasta llegar a 50, y se marca como "baja variabilidad". Cada clase se limita a 15 000 flujos y cada sensor a 15 000 flujos normales. Se excluyen las clases con menos de 100 flujos etiquetados: en Edge-IIoTset, MITM es un ARP spoofing de capa 2 que deja solo 2 flujos IP, así que el estudio trabaja con 13 ataques.

**Clientes.** Hay K = 5 clientes, uno por tipo de sensor: temperatura y humedad, ultrasónico, nivel de agua, pH y humedad del suelo. Cada cliente representa la pasarela (gateway) que concentra el tráfico de los sensores de ese tipo, no el sensor en sí (escenario *cross-silo*).
- **no-IID:** cada cliente entrena con el tráfico normal de su sensor y con un subconjunto disjunto de los 13 ataques (reparto round-robin).
- **IID:** el mismo total se baraja y se reparte en partes iguales.
- **Prueba:** el conjunto de prueba de cada cliente es el tráfico normal de su sensor más todos los ataques, lo que permite medir la detección de ataques que el cliente nunca vio.

**Supuesto de supervisión.** El detector es supervisado: se supone que cada cliente dispone de flujos etiquetados. En el experimento, las etiquetas provienen de la regla de etiquetado del testbed; en un despliegue llegarían con retraso (IDS por firmas, analistas, pruebas de penetración). Por eso los resultados son una cota superior del beneficio del FL con etiquetas escasas o tardías.

**Estandarización federada.** Primero se aplica log(1 + |x|) con signo. Después, la media y la desviación global se calculan a partir de sumas enviadas por cada cliente, sin compartir datos.

**Modelo.** Es un MLP 31 → 64 → 32 → 14 (Normal + 13 ataques) con ReLU y softmax, entrenado con Adam (tasa de aprendizaje 0,001, lote de 256). Usa R = 30 rondas y E = 2 épocas locales. Los esquemas centralizado y local reciben el mismo presupuesto total de R × E = 60 épocas.

**FedAvg / FedProx.** El servidor promedia los pesos ponderando por el número de muestras de cada cliente (McMahan et al., 2017). FedProx añade (μ/2)·‖w − w_global‖² a la pérdida local (Li et al., 2020).

**Métricas por cliente.**
- F1 macro.
- Tasa de falsos positivos.
- Brecha frente al modelo centralizado (ΔF1).
- F1 sobre los ataques no vistos localmente.
- Volumen de comunicación: 2 · K · R · |w| · 4 bytes.
- Tiempo de entrenamiento local por ronda.

**Estadística.** Se usan 5 semillas: cada una cambia la división de los datos, la inicialización y el orden de los lotes. Los resultados se reportan como media ± desviación estándar, y se contrastan con la prueba de rangos con signo de Wilcoxon, pareada por (semilla, cliente), con α = 0,05.

## Resultados obtenidos (Edge-IIoTset, 5 semillas)

| Esquema | F1 macro (%) | FPR (%) | Brecha vs. central (pp) | F1 ataques no vistos (%) | Comunicación (MB) | Tiempo local/ronda (s) |
|---|---|---|---|---|---|---|
| Centralizado | 83.55 ± 0.51 | 0.02 ± 0.01 | 0.00 ± 0.00 | 82.33 ± 0.56 | 11.90 ± 0.00 | 0.34 ± 0.01 |
| Local | 12.13 ± 0.39 | 0.01 ± 0.01 | 71.42 ± 0.73 | 0.03 ± 0.07 | 0.00 ± 0.00 | 0.07 ± 0.00 |
| FedAvg (IID) | 82.91 ± 1.21 | 0.03 ± 0.03 | 0.64 ± 0.84 | 81.64 ± 1.29 | 5.51 ± 0.00 | 0.07 ± 0.00 |
| FedAvg (no-IID) | 30.41 ± 4.96 | 0.04 ± 0.02 | 53.14 ± 4.89 | 25.05 ± 5.29 | 5.51 ± 0.00 | 0.07 ± 0.00 |
| FedProx μ=0.01 (no-IID) | 52.57 ± 4.32 | 0.02 ± 0.01 | 30.98 ± 4.63 | 48.85 ± 4.67 | 5.51 ± 0.00 | 0.08 ± 0.02 |
| FedProx μ=0.1 (no-IID) | 51.28 ± 3.38 | 0.03 ± 0.03 | 32.27 ± 3.81 | 47.45 ± 3.65 | 5.51 ± 0.00 | 0.07 ± 0.00 |

* 176 118 flujos etiquetados (Normal + 13 ataques; MITM excluido). Partición no-IID: JS promedio 0,216.
* Wilcoxon (n = 25): FedAvg no-IID supera a Local en F1, F1 de ataques no vistos y brecha ΔF1 (p < 0,001); FPR sin diferencia significativa (p = 0,076). FedProx supera a FedAvg no-IID (p < 0,001).
* Sensibilidad (3 semillas, no-IID): con R = 100 rondas, FedProx (E = 5) alcanza 63,1 % de F1, todavía a 20,5 pp del centralizado.
* Todas las tablas están en `results/` y las figuras en `results/figuras/`; los notebooks se entregan ejecutados.

## Reproducibilidad

Todas las semillas son fijas. El notebook 02 imprime las versiones de Python y de las librerías, y guarda la configuración usada en `results/configuracion_experimento.json`.

## Referencias

- Ferrag, M. A., Friha, O., Hamouda, D., Maglaras, L., & Janicke, H. (2022). Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning. *IEEE Access, 10*, 40281–40306. https://doi.org/10.1109/ACCESS.2022.3165809
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems, 2*, 429–450.
- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of AISTATS 2017*, PMLR 54, 1273–1282.
