import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet 
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("STOCK PRICE PREDICTION")

stocks = ("RELIANCE.NS","SBIN.NS","TATAMOTORS.NS","AAPL","GOOG","MSFT")
selected_stocks = st.selectbox("Select dataset for prediction",stocks)

n_years = st.slider("Years of prediction:", 1 , 3)
period = n_years * 365

@st.cache_resource
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...Done!")

st.subheader('RAW DATA')
st.write(data.tail(30))

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close', line=dict(color='blue')))
    fig.update_layout(title_text="TIME SERIES DATA", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds","Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('FORECAST DATA')
st.write(forecast.tail())

st.write('FORECAST DATA')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('FORECAST COMPONENTS')
fig2 = m.plot_components(forecast)
st.write(fig2)