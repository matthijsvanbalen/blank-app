import streamlit as st

st.title("🎈 My new app")
st.write(#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import cbsodata
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Deze regel toegevoegd

# Functie om gegevens op te halen
@st.cache  # Dit zorgt ervoor dat de gegevens niet opnieuw worden opgehaald als de app herlaadt
def load_data():
    gegevens_bevolking = pd.DataFrame(
        cbsodata.get_data('70072ned', filters="RegioS eq 'GM0363'", select=['ID','Perioden','TotaleBevolking_1', 'Mannen_2', 'Vrouwen_3', 'JongerDan5Jaar_4', 'k_5Tot10Jaar_5', 'k_10Tot15Jaar_6', 'k_15Tot20Jaar_7', 'k_20Tot25Jaar_8', 'k_25Tot45Jaar_9', 'k_45Tot65Jaar_10', 'k_65Tot80Jaar_11', 'k_80JaarOfOuder_12', 'LevendGeborenKinderen_58', 'Overledenen_60', 'Immigratie_74', 'Emigratie_75', 'InwonersOp31December_78', 'Bevolkingsgroei_79', 'GemiddeldAantalInwoners_81'])
    )
    gegevens_economie = pd.DataFrame(
        cbsodata.get_data('70072ned', filters="RegioS eq 'GM0363'", select=['ID','Perioden','TotaalBanen_111', 'ALandbouwBosbouwEnVisserij_112', 'BFNijverheidEnEnergie_113', 'GNCommercieleDienstverlening_114', 'OUNietCommercieleDienstverlening_115', 'ParticuliereHuishoudensExclStudenten_120', 'BronInkomenAlsWerknemer_126', 'BronInkomenAlsZelfstandige_127','Werkloosheid_154','ArbeidsongeschiktheidTotaal_158'])
    )
    totale_data = pd.merge(gegevens_bevolking, gegevens_economie, on='Perioden')
    totale_data.rename(columns={
        'Perioden': 'Jaartal', 
        'TotaleBevolking_1': 'Totale Bevolking', 
        'Mannen_2': 'Mannen', 
        'Vrouwen_3': 'Vrouwen',
        'LevendGeborenKinderen_58': 'Geboortes', 
        'Overledenen_60': 'Sterfte', 
        'Immigratie_74': 'Immigratie',
        'Emigratie_75': 'Emigratie',
        'TotaalBanen_111': 'Totaal aantal banen', 
        'Werkloosheid_154': 'Werkloosheid', 
        'ArbeidsongeschiktheidTotaal_158': 'Arbeidsongeschiktheid'
    }, inplace=True)

    # Berekeningen
    totale_data['Natuurlijke bevolkingsgroei'] = totale_data['Geboortes'] - totale_data['Sterfte']
    totale_data['Bevolkingsgroei door immigratie'] = totale_data['Immigratie'] - totale_data['Emigratie']
    totale_data['Totale bevolkingsgroei'] = totale_data['Natuurlijke bevolkingsgroei'] + totale_data['Bevolkingsgroei door immigratie']

    return totale_data

# Gegevens laden
totale_data = load_data()

# Titel
st.title("Analyse van Bevolking en Economie")

# Plot Werkloosheid
st.subheader("Werkloosheid per Jaar")
werkloosheid_fig = go.Figure()
for jaartal in totale_data['Jaartal'].unique():
    werkloosheid_df = totale_data[totale_data.Jaartal == jaartal]
    werkloosheid_fig.add_trace(go.Bar(x=werkloosheid_df['Jaartal'], y=werkloosheid_df['Werkloosheid'], name=str(jaartal)))

st.plotly_chart(werkloosheid_fig)

# Dropdown voor populatie
st.subheader("Selecteer Populatiegegevens")
options = ['Totale Bevolking', 'Mannen', 'Vrouwen']
selection = st.selectbox('Selecteer:', options)

# Plot voor geselecteerde populatie
plt.figure(figsize=(10, 5))
plt.plot(totale_data['Jaartal'], totale_data[selection], marker='o', label=selection)
plt.title(f'{selection} per Jaar')
plt.xlabel('Jaartal')
plt.ylabel('Aantal')
plt.xticks(totale_data['Jaartal'][::5], rotation=45)
plt.grid(True)
plt.legend()
st.pyplot(plt)

# Correlatie Heatmap
st.subheader("Correlatie Heatmap")
plt.figure(figsize=(10, 8))
corr = totale_data.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
st.pyplot(plt)

# Selecteer de doelvariabele en voorspellende variabelen
st.subheader("Lineair Regressie Model: Werkloosheid voorspellen")

# Kies variabelen met hoge correlatie met 'Werkloosheid' (gebaseerd op de heatmap)
predictors = ['Totaal aantal banen', 'GemiddeldAantalInwoners_81', 'Totale Bevolking', 'Mannen', 'Vrouwen']
target = 'Werkloosheid'

# Dataset voorbereiden
X = totale_data[predictors]
y = totale_data[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineair Regressiemodel trainen
model = LinearRegression()
model.fit(X_train, y_train)

# Voorspellingen maken
y_pred = model.predict(X_test)

# Model evaluatie
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Resultaten weergeven
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2 Score): {r2}")

# Vergelijking van voorspelde en werkelijke waarden
result_df = pd.DataFrame({'Werkelijke Waarde': y_test, 'Voorspelde Waarde': y_pred})
st.write(result_df)

# Scatter plot van werkelijke versus voorspelde waarden
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Werkelijke Waarde')
plt.ylabel('Voorspelde Waarde')
plt.title('Werkelijke vs Voorspelde Werkloosheid')
st.pyplot(plt)

)
# In[ ]:





    
