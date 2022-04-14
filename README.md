# Transformers Video


## Arquitectura del Modelo

* La mayoría de los modelos competitivos de transducción de secuencias neuronales tienen una estructura de codificador-decodificador.
Aquí, el codificador mapea una secuencia de entrada de representaciones de símbolos (x1, ..., xn) a una secuencia
de representaciones continuas z = (z1, ..., zn). Dado z, el descodificador genera una secuencia de salida
(y1, ..., ym) de símbolos, un elemento cada vez. En cada paso el modelo es autorregresivo.

## Encoder and Decoder

* El codificador está compuesto por una pila de N = 6 capas idénticas. Cada capa tiene dos
subcapas. La primera es un mecanismo de autoatención multicabezal, y la segunda es una red simple, totalmente conectada en función de la posición, de tipo feed-forward.

* El decodificador también está compuesto por una pila de N = 6 capas idénticas. Además de las dos
Además de las dos subcapas de cada capa del codificador, el decodificador inserta una tercera subcapa, que realiza la atención multicabeza
atención sobre la salida de la pila de codificadores. Al igual que en el codificador, empleamos conexiones residuales
alrededor de cada una de las subcapas, seguidas de la normalización de las capas.