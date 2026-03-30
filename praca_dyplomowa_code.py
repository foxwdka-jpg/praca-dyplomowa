import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import json

def zapisz(dane,nazwa_pliku):
    with open(nazwa_pliku, 'w', encoding='utf-8') as f:
        json.dump(dane, f, indent=4, ensure_ascii=False)

def wczytaj(nazwa_pliku):
    with open(nazwa_pliku, 'r', encoding='utf-8') as f:
        return json.load(f)


# pobierania danych z api
def pobierz_dane_z_nbp(data_start, data_koniec):
    print('# Rozpoczynam pobieranie danych #')

    start_dt = datetime.strptime(data_start, '%Y-%m-%d')
    koniec_dt = datetime.strptime(data_koniec, '%Y-%m-%d')

    lista_notowan = []
    aktualny_start = start_dt

    # pętla odpowiedzina za dzielnie na okraes max 90 dni
    while aktualny_start < koniec_dt:

        aktualny_koniec = aktualny_start + timedelta(days=90)
        if aktualny_koniec > koniec_dt:
            aktualny_koniec = koniec_dt


        s_str = aktualny_start.strftime('%Y-%m-%d')
        k_str = aktualny_koniec.strftime('%Y-%m-%d')
        url = f'http://api.nbp.pl/api/exchangerates/rates/A/EUR/{s_str}/{k_str}/'

        try:
            odpowiedz = requests.get(url, headers={'Accept': 'application/json'}, timeout=15)
            if odpowiedz.status_code == 200:
                dane = odpowiedz.json()
                lista_notowan.extend(dane['rates'])
            else:
                print(f' Brak danych dla okresu: {s_str} - {k_str}')
        except Exception as e:
            print(f' Błąd połączenia: {e}')

        aktualny_start = aktualny_koniec + timedelta(days=1)
        time.sleep(0.1)

    return lista_notowan


# zmiana na tabele wybranych danych
def utworz_tabele(lista_notowan):
    print('# Tworzenia tabeli DataFrame #')
    df = pd.DataFrame(lista_notowan)
    if df.empty:
        return df
    df = df[['effectiveDate', 'mid']]
    df.columns = ['data', 'kurs']
    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)
    return df


# tworzenie wykresu
def stworz_wykres(dane, kolumna,tytul,xlabel,ylabel):
    print('# Generowanie wykresu #')

    plt.figure(figsize=(10, 6))

    plt.plot(dane.index, dane[kolumna], color='black', linewidth=1.5)

    plt.title(tytul, fontsize=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)

    plt.grid(True, linestyle='--', alpha=0.6, color='gray')
    plt.xticks(rotation=45)

    # automatyczne dopasowanie marginesów
    plt.tight_layout()
    plt.show()


# wyswietalnie statystk opisowych dla danych
def statystyki(df):

    print('# Statystyki opisowe #')

    kurs = df['kurs']

    print('Liczba obserwacji (N):', len(kurs))
    print('Średnia (Mean):', round(kurs.mean(),2))
    print('Mediana (Median):', round(kurs.median(),2))
    print('Odchylenie standardowe (Std):', round(kurs.std(),2))
    print('Minimum:', round(kurs.min(),2))
    print('Maksimum:', round(kurs.max(),2))
    print('Skośność (Skewness):', round(skew(kurs),2))
    print('Kurtoza (Kurtosis):', round(kurtosis(kurs),2))


def test_adf(series):
    # rozszerzony test Dickeya-Fullera (ADF)
    print('# Rozszerzony test Dickeya-Fullera (ADF) #')
    result = adfuller(series, autolag='AIC', regression='c')
    print(f'Statystyka: {result[0]:.4f}, p-value: {result[1]:.4f}')

    print('Wartości krytyczne:')
    for key, value in result[4].items():
        print(f'{key}: {value:.4f}')


def test_kpss(series):
    # test KPSS
    print('# Test KPSS #')
    stat, p, lags, crit = kpss(series, regression='c', nlags='auto')
    print(f'Statystyka: {stat:.4f}, p-value: {p:.4f}')
    print('Wartości krytyczne:', crit)


def rysuj_acf(df, kolumna='kurs_diff'):
    dane = df[kolumna]
    # ustawienia dla ACF
    plot_acf(dane, lags=40, color='black', vlines_kwargs={'colors': 'black'}, bartlett_confint=True, alpha=0.05)
    plt.title(f'Funkcja Autokorelacji (ACF) dla pierwszych różnic kursu')
    plt.xlabel('Opóźnienie (k)')
    plt.ylabel('Autokorelacja')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def rysuj_pacf(df, kolumna='kurs_diff'):
    dane = df[kolumna]
    # ustawienia dla PACF
    plot_pacf(dane, lags=40, method='ywm', color='black', vlines_kwargs={'colors': 'black'},
              alpha=0.05)
    plt.title(f'Funkcja Autokorelacji Częściowej (PACF) dla pierwszych różnic kursu')
    plt.xlabel('Opóźnienie (k)')
    plt.ylabel('Autokorelacja częściowa')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# zakres danych
data_start = '2020-01-01'
data_koniec = '2025-12-31'
nazwa_pliku= 'dane.json'

#dane = pobierz_dane_z_nbp (data_start, data_koniec)
#zapisz(dane, nazwa_pliku)
dane = wczytaj(nazwa_pliku)
tabela = utworz_tabele(dane)
wynik = tabela
stworz_wykres(wynik,'kurs','Kurs EUR/PLN od 01.01.2020 r.do 31.12.2025 r.','Data','Kurs (PLN)')

print(wynik)
statystyki(wynik)
print(test_adf(wynik))
print(test_kpss(wynik))

wynik['kurs_diff'] = wynik['kurs'].diff()

wynik_diff = wynik.dropna()
statystyki(wynik_diff)

stworz_wykres(wynik_diff,'kurs_diff','Pierwsze różnice średniego dziennego kursu EUR/PLN od 01.01.2020 r.do 31.12.2025 r.','Data','PIerwsze różnice kursu')

test_adf(wynik_diff['kurs_diff'])
test_kpss(wynik_diff['kurs_diff'])

rysuj_acf(wynik_diff)
rysuj_pacf(wynik_diff)