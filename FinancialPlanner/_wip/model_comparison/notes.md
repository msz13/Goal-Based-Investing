### Podsumowanie różnych modeli

Wykorzystywane są następujące modele do generowania scenariuszy stóp zwrotów aktywów

1. Multivariate Normal
   1. z historycznymi parametrami
   2. z założeniami
2. VAR
   1. z excess returns
   2. z dividend growth i dicidend yeld
3. Cascade
4. Macro Finance


Najbardziej mi się podoba model Macro Finance, tylko:
- model norges bank nie mam danych na temat oczekiwanego wzrostu dywidend z dividend strips
- model msci - opraruje na FCF - nie ma danych na yteamt długi, nie wiem, jak obliczane jest expected growth i returns
Alternatywne moje specyfikacje, z ograniczonymi danymi:
    - present value model
    - kalibracja oczekiwanego wzrostu dywidend, i obliczanie w modelu tylko erp, kalibracja oczekiwanych dywidend
      - srednia z x lat
      - exponential moving average z x lat
      - NOWE - założenie stałej ERP i modelowanie tylko dywidend growth expectations jako element długotrwały i krótkotrwały


Problem z:
- present value - trudnośc w dodaniu innych parametrów do modelu oblicznaych oddzielnie, a także TCVAR do dywidend i equity premium
- kalibracja oczekiwan wzrostu dywidend - problem z initial value, ktora zmienia wynik

Wyzwanie:
- estymacja modelu, gdy expectation dividend growth jest niezależny od dividend growth, możliwości
  - dy = EWA(divdidend growth) + b1* (divgrowth - expexteddividend t-1) *  lampda * Expeceted Volatility * B * X (egzegonous, np. term)