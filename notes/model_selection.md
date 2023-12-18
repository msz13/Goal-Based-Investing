### Cele:
*	Optymalizacja porfela
*	Zaprojektowanie glide path – który porfel, przy danym horyzoncie czasowym, zapewnia optymalne ryzyko
*	Oszacowanie potrzebnego kapitału



### 1. ESG cechy jakościowe:
•	Current market conditons – uwzględnia aktualną sytację rynkową, np. po kryzysie nastąpi szybsze odbicie
•	Stylised facts – zapenią odpowiednie modele
•	Long term assumtions – długotrwałe założenia średniej do głównych czynników
•	Inne czynniki rynkowe


Jakie modele
1. caskade
   1. regime switching (także zfx, poszukac korelacje z equity) 
   2. hamilton
2. var/two elements
   1. ldi benchmark
   2. vangauard
   3. asset - są to premie/boxy (equity premiem, stopa, inflacja) - regresja do macro dla tych czynnikow 

### 2. Wybór modelu

| kryterium | Cascade | var | 
| --- | --- | --- |
| current market situation | tylko uwzględnia aktualne ryzyko | bieże pod uwagę czynniki makorekonomiczne |  
| wprowadzanie target/views | proste dla jednego czynniku | skomplikowane musi uwzględniać czynniki makro | 
| trudność - modele do stworzenia | hamilton jump/ewentualnie markov synchronisation, złożenie całość | AR scenario generator |
| wielość parametrów | mało, ale trudne do estymacji  | czynniki marko, assety, regime prediction |
| potrzeba nauki | hamilton, chyba, że markov switching | MC regresja |
| trudność | mało zmiennych, nietypowa estymacja | wiele, ale prosta estymacja | 

werjsa pośrednia
- oszacowac target z wykorzystaniem regresji, i kalibrować do tego

może caskade, jesli jest regime switching fx i łatwa będzie synchronizacja

jaka jest moja teoria

model fundamtentalny akcji
equity = divident yeld + earnings growth + time varign violality/p/e ratio

regime switching - target
interpretacja fundamentalna - inna srednia dłuokresowa niz w danych, np. niższa, wynika z niższej średniej regimu rozwoju lub niższej średniej kryzsyów, większej długości okresów kryzysów

hamilton switching - target 
interpretacja fundamentalna - inna srednia dłuokresowa niz w danych, np. niższa, wynika z niższej średniej akcji lub większej volality

powyższe modele nie uwzględniają wprost wartościowania, zmian p/e jak to powiązać?


Do czytania:
- vanguard return forecast
