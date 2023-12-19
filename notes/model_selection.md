### Cele:
*	Optymalizacja porfela
*	Zaprojektowanie glide path – który porfel, przy danym horyzoncie czasowym, zapewnia optymalne ryzyko
*	Oszacowanie potrzebnego kapitału


Kroki wyboru:
1. Czynniki ryzyka
2. Real wordl vs market consistent
3. Modele i parametryzacja
4. Kalibracja
5. Liczba scenariuszy
6. Okres
7. Częstotliwość


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
   4. two step vanguard model 

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

jaka jest moja teoria - boogle building block, based on fundamentals

jesli cascadowy, to
regime switching vs heston model violality

| model | za | przeciw | 
| --- | --- | --- |
heston | chyba lepsze wyjasnienie fundamentalne, rozbicie modelu na stock return, violality, i jumps |

model fundamtentalny akcji
equity = divident yeld + earnings growth + time varign violality/p/e ratio

regime switching - target
interpretacja fundamentalna - inna srednia dłuokresowa niz w danych, np. niższa, wynika z niższej średniej regimu rozwoju lub niższej średniej kryzsyów, większej długości okresów kryzysów

hamilton switching - target 
interpretacja fundamentalna - inna srednia dłuokresowa niz w danych, np. niższa, wynika z niższej średniej akcji lub większej volality

powyższe modele nie uwzględniają wprost wartościowania, zmian p/e jak to powiązać?


Jesli model regime switching, 
- to czy jest różnica między total return, a risk free, equity premium, dywidend yeld
- czy mogą być serie z różnymi modelami, np. lognorm, regresja, cir
- czy może być seria implied premiemum i zwroty akcji


#Two step vanguard model:
plusy
- fajny model, ale pozwala na predykcję 10 plus
- podoba mi się poprzez założenia filozoficzne, uwzględnie valuation
minusy
- mmożliwa do zastosowania tylko w developed world, gdzie jest wiele danych, wskaźniki oszacowano z okresu 33 lat
znaki zapytania
- nie wiadomo, jak się zachowają wyniki od 1-10 lat
- nie wiadomo jak uwzględnić time varing volality model (propozycje, volality jest mierzona w modelu var, model garch, heston model gdzie return drift wynika z modelu, a volality jest stochastyczna, regime switching) 
- pytanie jak uwzględnić powiązania z innymi rynkami i walutą
- 

  
