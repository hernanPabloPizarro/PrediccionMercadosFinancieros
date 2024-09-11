import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import numpy as np
import tempfile  # Para manejar archivos temporales
from PIL import Image  # Para abrir y mostrar imágenes
import concurrent.futures

@st.cache_data
def obtener_indicadores_compra2(ticker, period="1y", period1=None, umbral=1, dates=True):
    """
    Obtiene los datos históricos de un ticker y calcula los indicadores técnicos.

    Args:
    ticker (str): Símbolo del ticker de la acción.
    period (str): Período de los datos históricos. Por defecto es "1y".
    period1 (str): Fecha de fin de los datos históricos.
    umbral (float): Umbral para la señal de compra.
    dates (bool): Si es True, retorna también las fechas donde 'compra-señal' esté entre 0 y 0,64.

    Returns:
    pd.DataFrame: DataFrame con los datos históricos y los indicadores técnicos.
    """
    #fechas, fechasCompra, fechaVenta = [], [], []
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

        # Calcular Bandas de Bollinger
        df['bll_sup'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
        df['bll_inf'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

        # Señal de compra
        df['Min-B.Inf'] = np.minimum(df['Close'], df['Open']) - df['bll_inf']
        df['B.Sup-Max'] = df['bll_sup'] - np.maximum(df['Close'], df['Open'])
        df['MACD_T'] = (df['MACD'] - df['MACD'].shift(1)) / df['MACD'].shift(1)
        df.loc[(df['MACD'] < 0) & (df['MACD'].shift(1) < 0), 'MACD_T'] = -df['MACD_T']         # Ajustar el signo cuando ambos valores consecutivos de MACD sean negativos
        df.loc[(df['MACD'] > 0) & (df['MACD'].shift(1) < 0), 'MACD_T'] = -df['MACD_T']

        df['Bll.inf_T'] = (df['bll_inf'] - df['bll_inf'].shift(1)) / df['bll_inf'].shift(1)
        df['vol_T'] = (df['Volume'] - df['Volume'].shift(1)) / df['Volume'].shift(1)
        df['Shift'] = 0
        df.loc[(df['MACD_T'] > 0) & (df['MACD_T'].shift(1) <= 0), 'Shift'] = 1
        df.loc[(df['MACD_T'] < 0) & (df['MACD_T'].shift(1) >= 0), 'Shift'] = -1

        fechas = df[(df['Min-B.Inf'] < umbral) & (df['Bll.inf_T'] < 0)].index.tolist() #if not df.empty else []
        fechasCompra = df[(df['Shift']==1)&(df['MACD']<df['Signal Line'])].index.tolist() #if not df.empty else []
        fechaVenta = df[(df['Shift'] == -1) & (df['MACD'] > df['Signal Line'])].index.tolist() #if not df.empty else [] #modificado
    except Exception as e:
        print(f'Error al obtener datos para {ticker}: {e}')
        return pd.DataFrame(), {'ticker': ticker, 'error': str(e)}

    if dates:
        return df, fechas, fechasCompra, fechaVenta
    else:
        return df

def plot_chart2(df, ticker, vert_line=None, vert_line_compra=None, vert_line_venta=None):
    """
    Grafica los datos históricos y los indicadores técnicos.

    Args:
    df (pd.DataFrame): DataFrame con los datos e indicadores técnicos.
    ticker (str): Símbolo del ticker de la acción.
    vert_line_compra (list): Lista de fechas para agregar líneas verticales.
    """
    if df.empty:
        print(f"No hay datos para graficar para el ticker {ticker}.")
        return

    #Configurar subplots para MACD y RSI
    apds = [
        mpf.make_addplot(df['EMA50'], color='blue', panel=0),
        mpf.make_addplot(df['EMA200'], color='red', panel=0),
        mpf.make_addplot(df['MACD'], panel=1, color='purple'),
        mpf.make_addplot(df['Signal Line'], panel=1, color='orange'),
        mpf.make_addplot(df['RSI'], panel=2, color='green'),
        mpf.make_addplot(df['RSI-EMA6'], panel=2, color='#000033', alpha=0.7),
        mpf.make_addplot([70] * len(df), panel=2, color='r', linestyle='--'),
        mpf.make_addplot([50] * len(df), panel=2, color='#000000', linestyle='--'),
        mpf.make_addplot([30] * len(df), panel=2, color='g', linestyle='--'),
        mpf.make_addplot(df['bll_sup'], color='purple', panel=0, linestyle='--'),
        mpf.make_addplot(df['bll_inf'], color='purple', panel=0, linestyle='--')
    ]

    #Agregar líneas verticales en las fechas especificadas
    if vert_line:
        for date in vert_line:
            if date in df.index:
                vert_line_series = pd.Series(np.nan, index=df.index)
                vert_line_series.loc[date] = df['Close'].max()
                apds.append(mpf.make_addplot(vert_line_series, type='scatter', markersize=200, marker='v', color='blue'))

        for date in vert_line_compra:
            if date in df.index:
                vert_line_series_compra = pd.Series(np.nan, index=df.index)
                vert_line_series_compra.loc[date] = df['Close'].max()-2
                apds.append(mpf.make_addplot(vert_line_series_compra, type='scatter', markersize=200, marker='v', color='green'))

        for date in vert_line_venta:
            if date in df.index:
                vert_line_series_venta = pd.Series(np.nan, index=df.index)
                vert_line_series_venta.loc[date] = df['Close'].max()-4
                apds.append(mpf.make_addplot(vert_line_series_venta, type='scatter', markersize=200, marker='v', color='red'))

    #Gráfico de precios con los indicadores
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        mpf.plot(
            df,
            type='candle',
            style='charles',
            addplot=apds,
            title=f'Gráfico de Compra para {ticker} con EMA, MACD, RSI y Bollinger Bands',
            ylabel='Precio',
            ylabel_lower='Indicadores',
            volume=True,
            figscale=2,
            savefig=tmpfile.name
        )
        return tmpfile.name  # Retornar la ruta del archivo

def compra_tickers2(tickers, chart=False, period="1y", period1=None, umbral=1, vert_line_compra=None, dates=True):
    """
    Procesa una lista de tickers, obtiene indicadores, grafica los datos y maneja errores.

    Args:
    tickers (list): Lista de símbolos de tickers.
    period (str): Período de los datos históricos. Por defecto es "1y".
    period1 (str): Fecha de fin de los datos históricos.
    umbral (float): Umbral para la señal de compra.
    vert_line_compra (list): Lista de fechas para agregar líneas verticales.
    dates (bool): Si es True, retorna también las fechas donde 'compra-señal' esté entre 0 y 0,64.
    """
    all_dates = {}
    all_datesen ={}
    all_datesenon = {}
    df = pd.DataFrame()  # Inicializar df para evitar UnboundLocalError
    for ticker in tickers:
        print(f"{ticker}")
        try:
            if dates:
                df, fechas, fechasCompra, fechaVenta = obtener_indicadores_compra2(ticker, period=period, period1=period1, umbral=umbral, dates=dates)
                all_dates[ticker] = [date.strftime('%Y-%m-%d') for date in fechas]
                all_datesen[ticker] = [date.strftime('%Y-%m-%d') for date in fechasCompra]
                all_datesenon[ticker] = [date.strftime('%Y-%m-%d') for date in fechaVenta]
                print(f"Procesando...")
            else:
                df = obtener_indicadores_compra2(ticker, period=period, period1=period1, umbral=umbral, dates=dates)

            if not df.empty and chart:
                plot_chart2(df, ticker, vert_line=fechas, vert_line_compra=fechasCompra, vert_line_venta=fechaVenta)
        except Exception as e:
            print(f'Error procesando {ticker}: {e}')

    if dates:
        return df, all_dates, all_datesen, all_datesenon
    else:
        return df

url = 'https://raw.githubusercontent.com/hernanPabloPizarro/PrediccionMercadosFinancieros/main/syp500.csv'
df_syp500 = pd.read_csv(url)
acciones = df_syp500['Symbol'].tolist()

st.title("Predictor de Acciones")

with st.container(border=True):
    st.subheader('_Esta sección muestra información cada una acción_')
    tic = st.text_input("ticker?", value="AAPL")
    umbra = st.text_input("Umbral inferior", value=0)
    peripe = st.selectbox('periodo?',['1y', '6mo', '3mo', '1mo'])
    #señal = st.selectbox('Seleccione la señal para filtrar:',['Señal de baja', 'Compra', 'Venta'])
    charto = st.checkbox("Gráfico?")
    cuadro = st.checkbox("Tabla?")

    a = []
    a.append(tic)
    if st.button('Consultar'):
        dat, señal, compra, venta = compra_tickers2(a, chart=charto, period=peripe, period1=None, umbral=int(umbra), vert_line_compra=None, dates=True)
        if cuadro:
            datim = dat.copy()
            datim.reset_index(inplace=True)
            datimba = datim[['Date','Open', 'High', 'Low', 'Close','MACD','Min-B.Inf','Shift']].copy()
            datimba['Shift'] = datimba['Shift'].replace({1: 'compra', -1: 'venta'})
            datimba = datimba.rename(columns={'Shift': 'Señal'})
            datimba['Date'] = pd.to_datetime(datim['Date']).dt.date
            #df['Date'] = pd.to_datetime(df['Date']).dt.date
            st.dataframe(datimba)

        if charto and not dat.empty:
            image_path = plot_chart2(dat, a, vert_line=list(señal.values())[0], vert_line_compra=list(compra.values())[0], vert_line_venta=list(venta.values())[0])

            if image_path:
                image = Image.open(image_path)
                st.image(image)
                st.write('Triángulo azul: momento relativo de bajo valor')
                st.write('Triángulo verde: señal de compra')
                st.write('Triángulo rojo: señal de venta')


with st.container(border=True):
    st.subheader('_Esta sección busca acciones por criterio_')
    señal = st.selectbox('Seleccione la señal para filtrar:',['Sin criterio','Señal de baja', 'Compra', 'Venta'])
    if señal == 'Señal de baja':
        umb = int(st.text_input("Umbral inferior?", value="0"))
    else:
        umb = 0
    if señal != 'Sin criterio':
        perid = int(st.text_input("Período anterior?   (0= último; 1=penúltimo)", value=0))
        peride = (perid*-1)-1
    else:
        peride=-1
    
    if st.button("Cargar datos actuales"):
        placeholder = st.empty()
        placeholder.write('Cargando acciones, esto demora varios segundos...')
        
        # Inicializar la barra de progreso
        progreso = st.progress(0)
        
        reporte = []
        num_acciones = len(acciones)
        for i in range(num_acciones):
            try:
                e = compra_tickers2([acciones[i]], chart=False, period="6mo", period1=None, umbral=umb, vert_line_compra=None, dates=False)
                if señal == 'Sin criterio':
                    reporte.append(acciones[i])
                
                if señal == 'Señal de baja':
                    if (e.iloc[peride]['Min-B.Inf'] < umb) and (e.iloc[peride]['Bll.inf_T'] < 0):
                        reporte.append(acciones[i])

                if señal == 'Compra':
                    if e.iloc[peride]['shift']==1:
                        reporte.append(acciones[i])

                if señal == 'Venta':
                    if e.iloc[peride]['Shift']==-1:
                        reporte.append(acciones[i])

            except Exception as e:
                print(f'Error procesando {acciones[i]}: {e}')
            
            # Actualizar la barra de progreso
            progreso.progress((i + 1) / num_acciones)

        st.write(f'Acciones encontradas: {len(reporte)}')
        st.write(reporte)
        placeholder.empty()

        st.write('Triángulo azul: momento relativo de bajo valor')
        st.write('Triángulo verde: señal de compra')
        st.write('Triángulo rojo: señal de venta')
        
        # Generar gráficos para cada acción en la lista reporte
        for ticker in reporte:
            dat, señal, compra, venta = compra_tickers2([ticker], chart=True, period="1y", period1=None, umbral=int(umb), vert_line_compra=None, dates=True)

            if not dat.empty:
                image_path = plot_chart2(dat, ticker, vert_line=list(señal.values())[0], vert_line_compra=list(compra.values())[0], vert_line_venta=list(venta.values())[0])
                if image_path:
                    image = Image.open(image_path)
                    st.image(image)
