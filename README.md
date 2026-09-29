# Mosca conectoma resolviendo un cubo de Rubik

Demo para clase de machine learning. Una red de neuronas de disparo construida
con el **conectoma real del sistema nervioso central de *Drosophila* macho**
(MaleCNS v1.0, HHMI Janelia / U. Cambridge / MRC LMB / Google Research) elige
los giros que resuelven un cubo de Rubik.

## Arrancar

Doble clic en `iniciar.bat`, o:

    venv\Scripts\python.exe app\server.py

Y abrir http://127.0.0.1:5000

## Qué es y qué no es

`male-cns.janelia.org` publica un **conectoma**, no un simulador: son las
neuronas, sus sinapsis y sus neurotransmisores. La simulación la corre esta
máquina, con `app/brain.py`.

La mosca **no razona sobre el cubo**. Lo que hay es una pregunta legítima y
comprobable: *¿el cableado real de un cerebro sirve como sustrato de cómputo?*
El montaje es **reservoir computing**:

- El conectoma es un **reservorio recurrente fijo**. No se entrena ni una sinapsis.
- Solo se entrena una **lectura lineal** (regresión ridge) sobre las tasas de
  disparo. Es el único parámetro aprendido.
- Las etiquetas vienen de una **tabla BFS exacta** (3.064 estados a ≤5 giros),
  así que la precisión reportada es verdad de referencia, no autoevaluación.

**Resultados medidos:** 87,5 % de acierto en test contra 78 % de una lectura
lineal sobre el cubo crudo sin conectoma, y **25/25 cubos resueltos en 5,0 giros
de media (el óptimo es 5)**. ~0,09 s por decisión.

## La vía que se simula

| Población | Neuronas | Papel | Actividad en esta tarea |
|---|---|---|---|
| Fotorreceptores (`ol_sensory`) | 6.098 | entrada visual | casi muda |
| Proyección visual (`visual_projection`) | 9.201 | **entrada del estímulo** | ~146 Hz |
| Células de Kenyon (`Kenyon_Cell`) | 4.064 | código disperso (olfato) | ~0 Hz |
| Dopaminérgicas (`DAN`) | 340 | recompensa / RPE | ~0 Hz |
| MBON | 97 | lectura de valor | ~1 Hz |
| Complejo central (`CX`) | 2.950 | navegación | ~0 Hz |
| Descendentes (`descending_neuron`) | 1.314 | **salida motora** | activa |

**24.064 neuronas, 666.489 sinapsis reales.** La vía que de verdad computa aquí
es **proyección visual → neuronas descendentes** (17.119 sinapsis reales entre
ambas), o sea el eje visión→motor de la mosca.

El cuerpo pedunculado (Kenyon → DAN → MBON) y el complejo central quedan casi
mudos, y **eso es un resultado, no un fallo**: en el conectoma real reciben 885 y
641 sinapsis desde la vía visual, frente a las 17.119 que van a las descendentes.
Son circuitos olfativos y de navegación. La anatomía decide qué se activa.
Buen momento de clase: el paralelo Kenyon↔capa dispersa y DAN↔error de predicción
de recompensa es real en la mosca, pero **no en esta tarea**.

### Limitaciones que conviene decir en voz alta
- Las neuronas de proyección visual disparan a ~146 Hz, más de lo fisiológico: el
  modelo reducido omite las 89.403 neuronas intrínsecas del lóbulo óptico, que
  aportan buena parte de la inhibición que equilibraría la vía.
- Por lo mismo, el estímulo se inyecta directamente en las neuronas de proyección
  visual (la salida del lóbulo óptico), no en los fotorreceptores.
- La mosca no razona sobre el cubo, y la profundidad es 5 giros con 6 movimientos
  posibles: 3.064 estados. Es una demo de sustrato de cómputo, no de cognición.

## Detalles del modelo

- Neuronas de **integración y disparo con fugas** (LIF), τ=20 ms, dt=1 ms, 40 ms por decisión.
- Signo sináptico según el **neurotransmisor predicho**: acetilcolina excitatoria;
  GABA y glutamato (vía GluCl) inhibitorios — convención de Shiu et al. 2024.
  2.144 de 24.064 neuronas resultan inhibitorias.
- Se descartan contactos de peso < 3 sinapsis (ruido de segmentación).

## Archivos

    preprocess.py       extrae la subred de los .feather (ya ejecutado)
    app/cube.py         cubo 3x3; permutaciones derivadas de geometría 3D
    app/test_cube.py    pruebas de validación del cubo
    app/brain.py        reservorio LIF + BFS de verdad de referencia
    app/train.py        entrena la lectura lineal
    app/server.py       API Flask
    app/static/         interfaz (Three.js + canvas)
    data/               conectoma descargado (~1,1 GB) y subred extraída

Datos bajo licencia CC-BY 4.0.
