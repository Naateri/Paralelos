# Tarea - Multiplicación Matriz-Vector y Ordenamiento Paralelo

Tarea del curso Computación Paralela y Distribuida - UCSP 2020-1
Códigos basados de las implementaciones de [Peter Pacheco](https://www.cs.usfca.edu/~peter/ipp/)

Se debe instalar libopenmpi-dev para poder correr los programas.

```
sudo apt-get install libopenmpi-dev
```

Los tres programas ya incluyen sus binarios y un script para ejecutar los casos de prueba.

## Multiplicación Matriz-Vector

El programa matrix\_vector\_multiplication.c es uno hecho por cuenta propia, no optimizado sobre todo en lo que respecta a la memoria. Su binario respectivo es **mat-vec**.

Para correr los casos de prueba, primero hay que darle permisos de ejecución a **mat-vec.sh**.

```
chmod +x mat-vec.sh
```

Posteriormente, se ejecuta el script.

```
./mat-vec.sh
```

El programa mpi\_mat\_vect\_time.c está basado en la implementación hecha por Peter Pacheco. Su binario respectivo es **pet-mat-vec**.

De igual manera, para correr los casos de prueba, primero se le debe dar permisos de ejecución a **pet-mat-vec.sh**.

```
chmod +x pet-mat-vec.sh
```

Posteriormente se ejecuta el script.

```
./pet-mat-vec.sh
```

## Ordenamiento odd-even sort

El programa mpi\_odd\_even.c está basado en la implentación hecha por Peter Pacheco, al cuál se le han realizado modificaciones ligeras. Su binario respectivo es **pet-odd-even**.

Para correr los casos de pruebas, primero se le debe dar permisos de ejecución a **pet-odd-even.sh**.

```
chmod +x pet-odd-even.sh
```

Posteriormente, se ejecuta el script.

```
./pet-odd-even.sh
```
