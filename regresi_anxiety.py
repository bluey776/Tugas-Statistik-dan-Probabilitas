import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


try:
    df = pd.read_csv('dataset_survey.csv')
    print("Berhasil memuat data dari 'dataset_survey.csv'!")
except FileNotFoundError:
    print("File tidak ditemukan.")
    exit()

X = df[['screen time', 'sleep time']] 
Y = df['anxiety(1-10)']              

model = LinearRegression()
model.fit(X, Y)

# ekstraksi nilai matematika regresi
b0 = model.intercept_
b1 = model.coef_[0]
b2 = model.coef_[1]

# Menghitung Koefisien Determinasi (R-Squared)
Y_pred = model.predict(X)
r_squared = r2_score(Y, Y_pred)

# output
print("\n" + "="*58)
print("       HASIL ANALISIS REGRESI LINEAR BERGANDA (DATA ASLI)       ")
print("="*58)
print(f"Konstanta / Intercept (b0)        : {b0:.4f}")
print(f"Koefisien Screen Time (b1)        : {b1:.4f}")
print(f"Koefisien Sleep Time (b2)         : {b2:.4f}")
print(f"Koefisien Determinasi (R-Squared) : {r_squared:.4f} ({r_squared*100:.2f}%)")
print("="*58)

print("\n📈 Persamaan Regresi yang Terbentuk:")
print(f"Y = {b0:.4f} + ({b1:.4f} * X1) + ({b2:.4f} * X2)")
print("-"*58)

# visualisasi
print("\n📊 Sedang membuat grafik visualisasi...")
plt.figure(figsize=(12, 5))

# Grafik 1: Hubungan Screen Time dengan Anxiety
plt.subplot(1, 2, 1)
sns.regplot(data=df, x='screen time', y='anxiety(1-10)', color='blue', scatter_kws={'alpha':0.7})
plt.title('Pengaruh Screen Time terhadap Anxiety')
plt.xlabel('Screen Time (Jam)')
plt.ylabel('Skala Anxiety (1-10)')

# Grafik 2: Hubungan Sleep Time dengan Anxiety
plt.subplot(1, 2, 2)
sns.regplot(data=df, x='sleep time', y='anxiety(1-10)', color='green', scatter_kws={'alpha':0.7})
plt.title('Pengaruh Durasi Tidur terhadap Anxiety')
plt.xlabel('Durasi Tidur (Jam)')
plt.ylabel('Skala Anxiety (1-10)')

plt.tight_layout()
plt.show()
print("Grafik berhasil ditampilkan!")