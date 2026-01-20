# CUDA Convex Hull – Projekt końcowy

Projekt końcowy zaliczeniowy z kursu CUDA – Obliczenia równoległe na kartach graficznych.
Celem projektu jest implementacja i porównanie algorytmów wyznaczania otoczki wypukłej dla dużych zbiorów punktów na płaszczyźnie, zarówno na CPU, jak i GPU.

## Opis problemu

Dany jest zbiór punktów P na płaszczyźnie.
Zadaniem jest znalezienie najmniejszego wypukłego zbioru punktów P', takiego że:
P ⊆ P'

Innymi słowy, należy wyznaczyć wielokąt wypukły, wewnątrz którego (lub na jego krawędziach) znajdują się wszystkie punkty ze zbioru wejściowego.

## Zaimplementowane algorytmy

CPU

- Monotone Chain

    - Złożoność czasowa: O(n log n)
    - Sortowanie punktów leksykograficznie
    - Prosty w implementacji

Wykorzystywany jako algorytm referencyjny do walidacji wyników GPU

Źródło:
https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain

GPU (CUDA)

- QuickHull
    - Średnia złożoność: O(n log n)
    - Pesymistyczna złożoność: O(n^2)

Algorytm został zaimplementowany na 2 sposoby:

- Naiwna równoległa implementacja QuickHull

- Równoległy QuickHull inspirowany publikacją

Implementacja oparta na pracy:

Parallelizing Two Dimensional Convex Hull on NVIDIA GPU and Cell BE
https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/JournalPublications/2009/Parallelizing_nvidia_gpu.pdf


## Problemy implementacyjne

- Publikacja opisuje algorytm w sposób ogólnikowy
- Występują drobne błędy utrudniające bezpośrednią implementację
- Autorzy sugerują użycie przestarzałej biblioteki CUDPP:

## Dane testowe

Punkty losowane są z okręgu jednostkowego z rozkładem jednostajnym.

Wielkości zbiorów punktów wejściowych:

- 1 mln
- 5 mln
- 10 mln
- 50 mln


## Metodologia pomiarów

Każdy algorytm uruchamiany jest 3 razy dla 10 niezależnych zestawów danych. Wyniki czasowe zostały uśrednione

## Rezulatat

[](img/results.png)

## Wizualizacja

- Rysowanie odcinków: algorytm Bresenhama
- Rysowanie punktów: naiwne rysowanie kół

[50 punktów wejściowych](img/img_50.jpg)
[50000 punktów wejściowych](img/img_50000.jpg)