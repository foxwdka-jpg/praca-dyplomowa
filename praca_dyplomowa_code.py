import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf,acf,pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import json

# zapisywanie danychj do pliku
def zapisz(dane,nazwa_pliku):
    with open(nazwa_pliku, 'w', encoding='utf-8') as f:
        json.dump(dane, f, indent=4, ensure_ascii=False)

# wczytywanie danych z pliku
def wczytaj(nazwa_pliku):
    with open(nazwa_pliku, 'r', encoding='utf-8') as f:
        return json.load(f)


# pobierania danych z api
def pobierz_dane_z_nbp(data_start, data_koniec):
    print('\n' + '=' * 70)
    print('Pobieranie danych z nbp')
    print('=' * 70)

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
    print('\n' + '=' * 70)
    print('Tworzenie tabeli DataFrame')
    print('=' * 70)
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
    print('\n' + '=' * 70)
    print('Generowanie wykresu')
    print('=' * 70)

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
    print('\n' + '=' * 70)
    print('Statystyki opisowe')
    print('=' * 70)

    kurs = df['kurs']

    print('Liczba obserwacji (N):', len(kurs))
    print('Średnia (Mean):', round(kurs.mean(),2))
    print('Mediana (Median):', round(kurs.median(),2))
    print('Odchylenie standardowe (Std):', round(kurs.std(),2))
    print('Minimum:', round(kurs.min(),2))
    print('Maksimum:', round(kurs.max(),2))
    print('Skośność (Skewness):', round(skew(kurs),2))
    print('Kurtoza (Kurtosis):', round(kurtosis(kurs),2))

# rozszerzony test Dickeya-Fullera (ADF)
def test_adf(series):
    print('\n' + '=' * 70)
    print('Rozszerzony test Dickeya-Fullera (ADF)')
    print('=' * 70)
    result = adfuller(series, autolag='AIC', regression='c')
    print(f'Statystyka: {result[0]:.4f}, p-value: {result[1]:.4f}')

    print('Wartości krytyczne:')
    for key, value in result[4].items():
        print(f'{key}: {value:.4f}')

# test KPSS
def test_kpss(series):
    print('\n' + '=' * 70)
    print('Test KPSS')
    print('=' * 70)

    stat, p, lags, crit = kpss(series, regression='c', nlags='auto')
    print(f'Statystyka: {stat:.4f}, p-value: {p:.4f}')
    print('Wartości krytyczne:', crit)

# wykres acf
def rysuj_acf(df, kolumna='kurs_diff'):
    print('\n' + '=' * 70)
    print('ACF')
    print('=' * 70)
    dane = df[kolumna]
    print(f'Wartość ACF: {acf(dane, nlags=40)}')
    plot_acf(dane, lags=40, color='black', vlines_kwargs={'colors': 'black'}, bartlett_confint=True, alpha=0.05)
    plt.title(f'Funkcja Autokorelacji (ACF) dla pierwszych różnic kursu')
    plt.xlabel('Opóźnienie (k)')
    plt.ylabel('Autokorelacja')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# wykres pacf
def rysuj_pacf(df, kolumna='kurs_diff'):
    print('\n' + '=' * 70)
    print('PACF')
    print('=' * 70)
    dane = df[kolumna]
    print(f'Wartość PACF: {pacf(dane, nlags=40, method='ywm')}')
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
df_obserwacje = utworz_tabele(dane)
print(df_obserwacje)
stworz_wykres(df_obserwacje, 'kurs', 'Kurs EUR/PLN od 01.01.2020 r.do 31.12.2025 r.', 'Data', 'Kurs (PLN)')

print(df_obserwacje)
statystyki(df_obserwacje)
print(test_adf(df_obserwacje))
print(test_kpss(df_obserwacje))

df_obserwacje['kurs_diff'] = df_obserwacje['kurs'].diff()
df_obserwacje_diff = df_obserwacje.dropna()
statystyki(df_obserwacje_diff)

stworz_wykres(df_obserwacje_diff, 'kurs_diff', 'Pierwsze różnice średniego dziennego kursu EUR/PLN od 01.01.2020 r.do 31.12.2025 r.', 'Data', 'PIerwsze różnice kursu')
test_adf(df_obserwacje_diff['kurs_diff'])
test_kpss(df_obserwacje_diff['kurs_diff'])
rysuj_acf(df_obserwacje_diff)
rysuj_pacf(df_obserwacje_diff)

okres_coivd = df_obserwacje['2020-03-01':'2020-06-30']
okres_wojna= df_obserwacje['2022-02-24':'2022-04-30']
okres_spokojny = df_obserwacje.copy()
okres_spokojny = okres_spokojny.drop(okres_coivd.index)
okres_spokojny = okres_spokojny.drop(okres_wojna.index)

trening  = okres_spokojny.iloc[:-30].copy()
wartosc_rzeczywiste = okres_spokojny.iloc[-90:].copy()
test = okres_spokojny.iloc[-30:]

print('wartosć_rzeczywiste', len(wartosc_rzeczywiste))
print('długość testu',len(test))
liczba_dni = len(test)

# grid search dla selekcji parametrów arima
print('\n' + '='*70)
print('GRID SEARCH DLA ARIMA(p,1,q)')
print('='*70)
wyniki_grid = []

for p in range(0, 3):  # p = 0, 1, 2
    for q in range(0, 3):  # q = 0, 1, 2
        try:
            print(f'Estymacja ARIMA({p},1,{q})...', end=' ')
            model_temp = ARIMA(trening['kurs'], order=(p, 1, q)).fit()

            wyniki_grid.append({
                'p': p,
                'q': q,
                'Model': f'ARIMA({p},1,{q})',
                'AIC': model_temp.aic,
                'BIC': model_temp.bic,
                'Parametry': p + q
            })
            print(f'OK (AIC={model_temp.aic:.2f})')
        except:
            print('BŁĄD (model nie zbiegł się)')
            continue

# Tabela z wynikami grid search
df_grid = pd.DataFrame(wyniki_grid)
print('\n' + '=' * 70)
print('TABELA WYNIKÓW GRID SEARCH (sortowane według AIC)')
print('=' * 70)
print(df_grid.sort_values('AIC').to_string(index=False))

# Wybór najlepszego modelu
best_model_row = df_grid.sort_values('AIC').iloc[0]
best_p = int(best_model_row['p'])
best_q = int(best_model_row['q'])

print('\n' + '=' * 70)
print(f'*** NAJLEPSZY MODEL WEDŁUG AIC: ARIMA({best_p},1,{best_q}) ***')
print('=' * 70)
print(f'    AIC = {best_model_row["AIC"]:.4f}')
print(f'    BIC = {best_model_row["BIC"]:.4f}')

# Oszacowanie najlepszego modelu
print(f'\nEstymacja wybranego modelu ARIMA({best_p},1,{best_q})...')
model_arima_best = ARIMA(trening['kurs'], order=(best_p, 1, best_q)).fit()
print(model_arima_best.summary())


print('\n' + '=' * 70)
print('PROGNOZY NA RÓŻNYCH HORYZONTACH: h = 1, 5, 30')
print('=' * 70)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

horyzonty = [1, 5, 30]
wyniki_horyzontow = []

for h in horyzonty:
    print(f'\n--- Horyzont h = {h} dni ---')

    # Podział danych (na zbiorze spokojnym, po wykluczeniu kryzysów)
    trening_h = okres_spokojny.iloc[:-h]
    test_h = okres_spokojny.iloc[-h:]

    # Prognozy wszystkich modeli
    sma_h = np.full(h, trening_h['kurs'].iloc[-30:].mean())
    ses_h = SimpleExpSmoothing(trening_h['kurs']).fit().forecast(h)
    holt_h = ExponentialSmoothing(trening_h['kurs'], trend='add', seasonal=None).fit().forecast(h)
    rw_h = ARIMA(trening_h['kurs'], order=(0, 1, 0)).fit().forecast(h)
    arima_h = ARIMA(trening_h['kurs'], order=(best_p, 1, best_q)).fit().forecast(h)


    # Wartości rzeczywiste
    y_true = test_h['kurs'].values

    # Obliczenie błędów dla każdego modelu
    modele = {
        'SMA(30)': sma_h,
        'SES': ses_h,
        'Holt': holt_h,
        'ARIMA(0,1,0)': rw_h,
        f'ARIMA({best_p},1,{best_q})': arima_h
    }

    for nazwa, pred in modele.items():
        mae = np.mean(np.abs(y_true - pred))
        rmse = np.sqrt(np.mean((y_true - pred) ** 2))
        mape = np.mean(np.abs((y_true - pred) / y_true)) * 100

        wyniki_horyzontow.append({
            'Model': nazwa,
            'Horyzont': h,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        })

# Tabela zbiorcza
df_horyzonty = pd.DataFrame(wyniki_horyzontow)

print('\n' + '=' * 70)
print('TABELA PRZESTAWNA - MAE')
print('=' * 70)
pivot_mae = df_horyzonty.pivot(index='Model', columns='Horyzont', values='MAE')
print(pivot_mae)

print('\n' + '=' * 70)
print('TABELA PRZESTAWNA - RMSE')
print('=' * 70)
pivot_rmse = df_horyzonty.pivot(index='Model', columns='Horyzont', values='RMSE')
print(pivot_rmse)


print('\n' + '=' * 70)
print('WYKRES PORÓWNAWCZY PROGNOZ (h=30)')
print('=' * 70)
h = 30
trening_30 = okres_spokojny.iloc[:-30]
test_30 = okres_spokojny.iloc[-30:]

# Prognozy wszystkich modeli dla h=30
sma_30 = np.full(30, trening_30['kurs'].iloc[-30:].mean())
ses_30 = SimpleExpSmoothing(trening_30['kurs']).fit().forecast(30)
holt_30 = ExponentialSmoothing(trening_30['kurs'], trend='add', seasonal=None).fit().forecast(30)
rw_30 = ARIMA(trening_30['kurs'], order=(0, 1, 0)).fit().forecast(30)
arima_30 = ARIMA(trening_30['kurs'], order=(best_p, 1, best_q)).fit().forecast(30)

# Wykres
plt.figure(figsize=(14, 7))

# Dane treningowe (ostatnie 60 dni dla kontekstu)
plt.plot(trening_30['kurs'].iloc[-60:].index, trening_30['kurs'].iloc[-60:],
         label='Dane treningowe', color='gray', linewidth=1)

# Dane testowe (rzeczywiste)
plt.plot(test_30.index, test_30['kurs'],
         label='Rzeczywiste (test)', color='black', linewidth=2, marker='o')

# Prognozy
plt.plot(test_30.index, sma_30, label='SMA(30)', linestyle='--', linewidth=1.5)
plt.plot(test_30.index, ses_30, label='SES', linestyle='--', linewidth=1.5)
plt.plot(test_30.index, holt_30, label='Holt', linestyle='--', linewidth=1.5)
plt.plot(test_30.index, rw_30, label='ARIMA(0,1,0) - RW', linestyle=':', linewidth=2)
plt.plot(test_30.index, arima_30, label=f'ARIMA({best_p},1,{best_q})', linestyle='-', linewidth=2)

plt.xlabel('Data')
plt.ylabel('Kurs EUR/PLN')
plt.title('Porównanie prognoz na horyzoncie h=30 dni')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wykres_prognozy_h30.png', dpi=300, bbox_inches='tight')
plt.show()

print('\nZapisano wykres: wykres_prognozy_h30.png')

