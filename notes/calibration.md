
Kalibracja, z views:
*  Założenie średnich długoterminowych, zamiana intercept w modelu
*  Model dodawania views z artykułu calibration..
*  Model view z ortec
*  model z conning: estymacja parametrów mle + kalman filter + constrained mle, aby zminimalizować różnicę między target i aktual
*  moj - zrobic target do symulacji i grid search parametrow
  

Moja inicjatywa:
* targetowanie akcji na podstawie estymacji market expextation dla 10 lat, ale dla 20 lat z historii

Dostosowanie do targetu:
- grid seach parametru, gdy kalibracja jest najbliżej targetu
- optymalizacja - minmilalizacja różnicy między modelem a targetem z constrained parameters z bounderies z baye estimation, np. jesli mean volatility w confidenve interval 50% jest 0.01 - 0.03 - to to są constraindes!!!
- constrainded mle - dostosowanie parametrów likehood i targetu - chyba dotyczy dostosowania do długoletnich targetów, maximum likehood z penatly function (target - estimated)
- wariancja 1 - np. przy regime switching - przyjęcie pewnych parametrów stałych, np. średniej, szukanie tylko pozostałych parametrów


Targotowanie polski do innego podobnego kraju, np. hiszpania (też przechodziła modernizację, podobna wilekość)


Testing USA
Forecast | training |
| 2023 | 1998-2013 |
| 2018 | 1993 - 2008 |
| 2013 | 1988 - 2003 |


connging model
- dylemat, jak oszacujemy parametry do targetu, czy korelacje resuduals obliczyc oszacowanych parametrow, czy historycznych? 

constrained mle:
- maksymalne prawdopodobienstwo podobieństwa do wyników do aktualneych (MLe) i osiągniecia target values

Prosty zabieg:
- dodać rożnice między historical and target do danych



dsd