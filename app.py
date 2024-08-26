import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import numpy as np
import tempfile  # Para manejar archivos temporales
from PIL import Image  # Para abrir y mostrar imágenes
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# ----------------------------------funciones
@st.cache_data
def cargar_datos_ticker(ticker, period):
    return yf.download(ticker, period=period)
# --------------------------graficos de busqueda

# Función para graficar datos
def graficar_datos_con_subplots(df, ticker, height=900):
    fig = make_subplots(
        rows=3, cols=1,  # Tres filas, una columna
        row_heights=[0.4, 0.4, 0.4],  # Ajusta el tamaño relativo de las filas
        shared_xaxes=True,  # Comparte el eje X entre los subplots
        vertical_spacing=0.2  # Espaciado vertical entre subplots
    )

    # Gráfico de velas en la primera fila
    fig.add_trace(
        go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlesticks'),
        row=1, col=1
    )

    # Indicadores técnicos en la segunda fila

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='orange', width=1, dash='dash'), name='Bollinger Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='orange', width=1, dash='dash'), name='Bollinger Lower'), row=1, col=1)

    # MACD y Signal Line en la segunda fila
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='purple', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='red', width=1), name='Signal Line'), row=2, col=1)

    # RSI en la tercera fila
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                             line=dict(color='blue', width=1.5), 
                             name='RSI'), row=3, col=1)
    #---
    fig.add_trace(go.Scatter(
    x=df.index, 
    y=[30] * len(df),  # Una lista del mismo largo que el DataFrame con el valor 30
    line=dict(color='red', width=1, dash='dash'),  # Línea discontinua roja
    name='RSI 30'), 
    row=3, col=1)

# Añadir línea horizontal en RSI=70
    fig.add_trace(go.Scatter(
    x=df.index, 
    y=[70] * len(df),  # Una lista del mismo largo que el DataFrame con el valor 70
    line=dict(color='green', width=1, dash='dash'),  # Línea discontinua verde
    name='RSI 70'), 
    row=3, col=1)
    #---
    # Configuración del layout
    fig.update_layout(
        title=f" {ticker} para 1 año",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark",
        height=height  # Altura total del gráfico
    )

    return fig

# --------------------------funcion compra
def obtener_indicadores_compra(ticker, period="1y", period1=None, boll=False, dates=True):
    """
    Obtiene los datos históricos de un ticker y calcula los indicadores técnicos.

    Args:
    ticker (str): Símbolo del ticker de la acción.
    period (str): Período de los datos históricos. Por defecto es "1y".
    period1 (str): Fecha de fin de los datos históricos.
    boll (bool): Si es True, calcula las bandas de Bollinger.
    dates (bool): Si es True, retorna también las fechas donde 'compra-señal' esté entre 0 y 0,64.

    Returns:
    pd.DataFrame: DataFrame con los datos históricos y los indicadores técnicos.
    """
    try:
        stock = yf.Ticker(ticker)
        if period1:
            df = stock.history(start=period, end=period1)
        else:
            df = stock.history(period=period)

        if df.empty:
            raise ValueError("No se obtuvieron datos para el ticker proporcionado.")

        # Calcular EMA de 50 y 200 días
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # Calcular RSI de 14 días
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI-EMA6'] = df['RSI'].ewm(span=6, adjust=False).mean()

        # Calcular MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calcular Bandas de Bollinger si se solicita
        if boll:
            df['bll_sup'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
            df['bll_inf'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

        df['RSI<30'] = np.where(df['RSI']<=30,1,0)
        df['RSI-Signal'] = df['RSI<30'].diff().fillna(0).astype(int)
        df['compra-senal'] = np.where(df['RSI-Signal'] == -1, df['MACD'] - df['Signal Line'], np.nan)

    except Exception as e:
        print(f'Error al obtener datos para {ticker}: {e}')
        return pd.DataFrame(), {'ticker': ticker, 'error': str(e)}

    # Usar el índice como fechas
    fechas = df[(df['compra-senal'] <= 0.68)].index.tolist()

    if dates:
        return df, fechas
    else:
        return df

def plot_chart_compra(df, ticker, boll=False, vert_line_compra=None):
    """
    Grafica los datos históricos y los indicadores técnicos.

    Args:
    df (pd.DataFrame): DataFrame con los datos e indicadores técnicos.
    ticker (str): Símbolo del ticker de la acción.
    boll (bool): Si es True, grafica las bandas de Bollinger.
    vert_line_compra (list): Lista de fechas para agregar líneas verticales.
    """
    if df.empty:
        print(f"No hay datos para graficar para el ticker {ticker}.")
        return

    # Configurar subplots para MACD y RSI
    apds = [
        mpf.make_addplot(df['EMA50'], color='blue', panel=0),
        mpf.make_addplot(df['EMA200'], color='red', panel=0),
        mpf.make_addplot(df['MACD'], panel=1, color='purple'),
        mpf.make_addplot(df['Signal Line'], panel=1, color='orange'),
        mpf.make_addplot(df['RSI'], panel=2, color='green'),
        mpf.make_addplot(df['RSI-EMA6'], panel=2, color='#000033', alpha=0.7),
        mpf.make_addplot([70] * len(df), panel=2, color='r', linestyle='--'),         # Línea horizontal en y=70
        mpf.make_addplot([50] * len(df), panel=2, color='#000000', linestyle='--'),   # Línea horizontal en y=30
        mpf.make_addplot([30] * len(df), panel=2, color='g', linestyle='--')          # Línea horizontal en y=30
    ]

    # Agregar Bandas de Bollinger si se solicita
    if boll:
        apds.append(mpf.make_addplot(df['bll_sup'], color='purple', panel=0, linestyle='--'))
        apds.append(mpf.make_addplot(df['bll_inf'], color='purple', panel=0, linestyle='--'))

    # Agregar vertices en las fechas especificadas
    if vert_line_compra:
        for date in vert_line_compra:
            if date in df.index:
                vert_line_series = pd.Series(np.nan, index=df.index)
                vert_line_series.loc[date] = df['Close'].max()  # Ajustar a un valor visible en el gráfico
                apds.append(mpf.make_addplot(vert_line_series, type='scatter', markersize=200, marker='v', color='green'))

    # Gráfico de precios con los indicadores
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        mpf.plot(
            df,
            type='candle',
            style='charles',
            addplot=apds,
            title= f'Gráfico de {ticker} ',
            ylabel='Precio',
            ylabel_lower='Indicadores',
            volume=True,
            figsize=(12, 12),  # Aumenta el tamaño del gráfico (ancho, alto)
            figscale=1,
            savefig=tmpfile.name  # Guardar el gráfico en el archivo temporal
        )
        return tmpfile.name  # Retornar la ruta del archivo

def compra_tickers(tickers, chart=False, period="1y", period1=None, boll=False, vert_line_compra=None, dates=True):
    all_dates = {}
    df = pd.DataFrame()  # Inicializar df para evitar UnboundLocalError

    total_steps = len(tickers)  # Número total de pasos basado en los tickers
    progress = st.progress(0)  # Inicializar la barra de progreso

    for i, ticker in enumerate(tickers):
        print(f"{ticker}")
        try:
            if dates:
                df, Dates = obtener_indicadores_compra(ticker, period=period, period1=period1, boll=boll, dates=dates)
                all_dates[ticker] = [date.strftime('%Y-%m-%d') for date in Dates]
                print(f"Procesando...")
            else:
                df = obtener_indicadores_compra(ticker, period=period, period1=period1, boll=boll, dates=dates)

            if not df.empty and chart:
                plot_chart_compra(df, ticker, boll=boll, vert_line_compra=Dates)
        except Exception as e:
            print(f'Error procesando {ticker}: {e}')
        
        # Actualizar la barra de progreso
        progress.progress((i + 1) / total_steps)
    
    if dates:
        return df, all_dates
    else:
        return df

# --------------------------funcion venta
def obtener_indicadores_venta(ticker, period="1y", period1=None, dates=True):
    """
    Obtiene los datos históricos de un ticker y calcula los indicadores técnicos.

    Args:
    ticker (str): Símbolo del ticker de la acción.
    period (str): Período de los datos históricos. Por defecto es "1y".
    period1 (str): Fecha de fin de los datos históricos.
    boll (bool): Si es True, calcula las bandas de Bollinger.

    Returns:
    pd.DataFrame: DataFrame con los datos históricos y los indicadores técnicos.
    """
    try:
        stock = yf.Ticker(ticker)
        if period1:
            df = stock.history(start=period, end=period1)
        else:
            df = stock.history(period=period)

        if df.empty:
            raise ValueError("No se obtuvieron datos para el ticker proporcionado.")

        # Calcular EMA de 50 y 200 días
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

        # Calcular RSI de 14 días
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI-EMA6'] = df['RSI'].ewm(span=6, adjust=False).mean()

        # Calcular MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Calcular Bandas de Bollinger si se solicita
        mar= 0
        df['bll_sup'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
        df['bll_inf'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
        df['venta'] = np.where(df['bll_sup'].notna() & ((df['bll_sup'] - df[['Open', 'Close']].max(axis=1) ) <= mar), 1, 0)
        #df['venta'] = np.where((df[['Open', 'Close']].max(axis=1) - df['bll_sup']) <= 0, 0, 1)

    except Exception as e:
        print(f'Error al obtener datos para {ticker}: {e}')
        return pd.DataFrame(), {'ticker': ticker, 'error': str(e)}

    # Usar el índice como fechas
    fechas = df['venta'][df['venta']==1].index.tolist()

    if dates:
        return df, fechas
    else:
        return df

def plot_chart_venta(df, ticker, vert_line_venta=None):
    """
    Grafica los datos históricos y los indicadores técnicos.

    Args:
    df (pd.DataFrame): DataFrame con los datos e indicadores técnicos.
    ticker (str): Símbolo del ticker de la acción.
    boll (bool): Si es True, grafica las bandas de Bollinger.
    vert_line_venta (list): Lista de fechas para agregar líneas verticales.
    """
    if df.empty:
        print(f"No hay datos para graficar para el ticker {ticker}.")
        return

    # Configurar subplots para MACD y RSI
    apds = [
        mpf.make_addplot(df['EMA50'], color='blue', panel=0),
        mpf.make_addplot(df['EMA200'], color='red', panel=0),
        mpf.make_addplot(df['MACD'], panel=1, color='purple'),
        mpf.make_addplot(df['Signal Line'], panel=1, color='orange'),
        mpf.make_addplot(df['RSI'], panel=2, color='green'),
        mpf.make_addplot(df['RSI-EMA6'], panel=2, color='#000033', alpha=0.7),
        mpf.make_addplot([70] * len(df), panel=2, color='r', linestyle='--'),         # Línea horizontal en y=70
        mpf.make_addplot([50] * len(df), panel=2, color='#000000', linestyle='--'),   # Línea horizontal en y=30
        mpf.make_addplot([30] * len(df), panel=2, color='g', linestyle='--'),          # Línea horizontal en y=30
        mpf.make_addplot(df['bll_sup'], color='purple', panel=0, linestyle='--'),
        mpf.make_addplot(df['bll_inf'], color='purple', panel=0, linestyle='--')]

    # Agregar vertices en las fechas especificadas
    if vert_line_venta:
        for date in vert_line_venta:
            if date in df.index:
                vert_line_series = pd.Series(np.nan, index=df.index)
                vert_line_series.loc[date] = df['Close'].max()  # Ajustar a un valor visible en el gráfico
                apds.append(mpf.make_addplot(vert_line_series, type='scatter', markersize=200, marker='v', color='orange'))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpofile:
        mpf.plot(
            df,
            type='candle',
            style='charles',
            addplot=apds,
            title= f'Gráfico de {ticker} ',
            ylabel='Precio',
            ylabel_lower='Indicadores',
            volume=True,
            figsize=(12, 12),  # Aumenta el tamaño del gráfico (ancho, alto)
            figscale=1,
            savefig=tmpofile.name  # Guardar el gráfico en el archivo temporal
        )
        return tmpofile.name  # Retornar la ruta del archivo

def venta_tickers(tickers, chart=False, period="1y", period1=None, dates=True):
    """
    Procesa una lista de tickers, obtiene indicadores, grafica los datos y maneja errores.

    Args:
    tickers (list): Lista de símbolos de tickers.
    period (str): Período de los datos históricos. Por defecto es "1y".
    period1 (str): Fecha de fin de los datos históricos.
    boll (bool): Si es True, calcula y grafica las bandas de Bollinger.
    vert_line_compra (list): Lista de fechas para agregar líneas verticales.
    dates (bool): Si es True, retorna también las fechas donde 'compra-señal' esté entre 0 y 0,64.
    """
    all_dates = {}
    for ticker in tickers:
        print(f"{ticker}")
        try:
            if dates:
                df, Dates = obtener_indicadores_venta(ticker, period=period, period1=period1, dates=dates)
                all_dates[ticker] = [date.strftime('%Y-%m-%d') for date in Dates]
                print(f"Fechas de {ticker}: {Dates}")
            else:
                df = obtener_indicadores_venta(ticker, period=period, period1=period1, dates=dates)

            if not df.empty and chart:
                plot_chart(df, ticker,  vert_line_venta=Dates)
        except Exception as e:
            print(f'Error procesando {ticker}: {e}')

    if dates:
        return df, all_dates
    else:
        return df

# ----------------------------------visualización

url = 'https://raw.githubusercontent.com/hernanPabloPizarro/PrediccionMercadosFinancieros/main/syp500.csv'
df_syp500 = pd.read_csv(url)
acciones = df_syp500['Symbol'].tolist()
# ----   ----   ----   ----   ----   ----   
st.title('Predictor I - S&P500')
st.write('Esta app permite buscar señales de acciones para ser comprada/vendida  por su ticker. Además busca dentro de las 500 acciones del índice S&P500 cuales pueden ser una opción para ser compradas')
with st.expander("Consulta por una acción"):
    tic = st.text_input("Ingrese ticker")
    charto = st.checkbox("Mostrar gráfico?")

    #operation = st.selectbox("Elige una operación", ("Suma", "Resta", "Multiplicación", "División"))
    peri = st.selectbox("Ingrese período",("1Mo","3Mo","6Mo","1Y","5Y","max"))
    df_check = st.checkbox("Mostrar Tabla?")

    a=[]
    a.append(tic)
    if st.button('Compra'):
        if not tic:
            st.write('Ingrese un Ticker')
        else:
            dat,dato=compra_tickers(a, chart=charto, period=peri, boll=True)
            if df_check==True:
                st.write(dat)
            st.write("Fechas de compra")
            st.write(dato)

            if charto and not dat.empty:
                image_path = plot_chart_compra(dat, tic, boll=True, vert_line_compra=list(dato.values())[0])
                if image_path:
                    image = Image.open(image_path)
                    st.image(image)
    if st.button('Venta'):
        if not tic:
            st.write('Ingrese un Ticker')
        else:
            data,datim=venta_tickers(a, chart=charto, period=peri, dates=True)
            if df_check==True:
                st.write(data)
            st.write("Fechas de venta")
            st.write(datim)

            if charto and not data.empty:
                image_path = plot_chart_venta(data, tic, vert_line_venta=list(datim.values())[0])
                if image_path:
                    image = Image.open(image_path)
                    st.image(image)
# ----   ----   ----   ----   ----   ----   

with st.expander("Buscar"):
    if st.button("Cargar datos actuales"):
        st.write('Cargando acciones, esto demora varios segundos...')

        with st.spinner('Por favor, espere mientras se cargan los datos...'):
            df500, dates500 = compra_tickers(acciones[1:50], chart=False, period='1mo', period1=False, boll=True, dates=True)
        st.write('Datos cargados')
        st.session_state['dates500'] = dates500  # Guardar en session_state

    # Verificar si dates500 está en session_state antes de continuar
    if 'dates500' in st.session_state:
        dates500 = st.session_state['dates500']  # Recuperar de session_state
        FechaB = st.text_input('Ingrese fecha en formato yyyy-mm-dd. Ejemplo: 2024-08-22')
        
        if st.button("Buscar fecha"):
            try:
                encontradas = []
                for ticker, fechas in dates500.items():
                    if FechaB in fechas:
                        encontradas.append(ticker)
                if not encontradas:
                    st.write('No se encontró esa fecha')
                else:
                    st.session_state['encontradas'] = encontradas
                    if encontradas:
                        st.success(f'Acciones con señal de compra el {FechaB}:')
                        st.write(encontradas)
                        #Graficos de encontradas
                        for ticker in encontradas:
                            df = cargar_datos_ticker(ticker, '1y')
                            st.subheader(f"Análisis de {ticker}")
                            st.plotly_chart(graficar_datos_con_subplots(df, ticker))
                        #Graficos de encontradas
            except NameError as e:
                st.write('Aún no hay datos cargados')
    else:
        st.write('Primero debe cargar los datos actuales desde el botón.')
        st.write('Este proceso toma tiempo')
