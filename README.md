# Eksāmenu sesijas plānošana ar tabu meklēšanu

Praktiskais darbs kursā "Praktiskā kombinatoriskā optimizācija".

**Uzdevums:** sastādīt universitātes eksāmenu sesiju ar pēc iespējas mazāk laika slotiem tā,
lai nevienam studentam divi eksāmeni nenotiktu vienlaicīgi. Problēma tiek modelēta kā
grafu krāsošana: eksāmens = virsotne, kopīgi studenti = šķautne, laika slots = krāsa.

**Algoritms:** tabu meklēšana (TabuCol) ar dinamisku tabu ilgumu, aspirāciju un restartiem
pēc stagnācijas. Sākuma risinājums ar DSatur, pēc tam slotu skaitu samazina pa vienam.
NumPy izmantots tikai masīvu aprēķiniem.

## Struktūra

```
data/                  Toronto testa instances (.exm eksāmeni, .stu studentu reģistrācijas)
src/problem.py         datu ielāde un konfliktu grafa izveide
src/dsatur.py          DSatur sākuma risinājums
src/tabucol.py         tabu meklēšana k-krāsošanai
src/solver.py          slotu skaita minimizācija (k -> k-1), restarti
src/clique.py          apakšējā robeža: maksimālā klika (zaru un robežu metode)
main.py                vienas instances risināšana no komandrindas
run_experiments.py     visu testu palaišana -> results/
make_figures.py        grafiki aprakstam -> report/img/
tests/test_basic.py    pamata testi
results/               testēšanas rezultāti (CSV, atrastie saraksti)
```

## Lietošana

Nepieciešams Python 3.10+ ar `numpy` (grafikiem arī `matplotlib`).

```bash
pip install numpy matplotlib
python main.py data/hec-s-92.exm --time 30          # viena instance
python run_experiments.py --time 60 --seeds 10      # visi testi (~30 min ar 2 kodoliem)
python make_figures.py                              # grafiki
python tests/test_basic.py                          # testi
```

## Google Colab

Bez instalācijas: [Google Colab](https://colab.research.google.com) → *New notebook*
un katru bloku ievieto atsevišķā šūnā.

```python
!git clone https://github.com/IlyaPoloz/graph-ts.git
%cd graph-ts
!pip install -q -r requirements.txt
```

```python
!python main.py data/hec-s-92.exm --time 30                 # viena instance
!python run_experiments.py --time 60 --seeds 10 --jobs 2    # visi testi (~30 min)
!python make_figures.py                                     # grafiki
```

```python
!zip -qr rezultati.zip results report/img                   # rezultātu lejupielāde
from google.colab import files
files.download("rezultati.zip")
```

## Testa dati

Toronto eksāmenu plānošanas testa kopa (Carter, Laporte & Lee, 1996), I versija -
reālas eksāmenu sesijas no Kanādas un citām mācību iestādēm. Repozitorijā iekļautas
8 no 13 instancēm (81-622 eksāmeni); faili ņemti no publiskā repozitorija
[J4NN0/examination-timetabling](https://github.com/J4NN0/examination-timetabling)
un pārbaudīti pret publicētajiem raksturlielumiem (eksāmenu, studentu, reģistrāciju
skaits, konfliktu blīvums). Oriģinālā kopa: <https://people.cs.nott.ac.uk/pszrq/data.htm>.
