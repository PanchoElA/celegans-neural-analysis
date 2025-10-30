"""
Simple Dash Server Test - Minimal neural app to verify connectivity
"""
import dash
from dash import html, dcc
import plotly.graph_objects as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(" Neural App - Connection Test"),
    html.P("If you can see this, the server is working!"),
    dcc.Graph(
        figure=go.Figure().add_trace(go.Scatter(x=[1,2,3], y=[1,4,2], name="Test")),
        style={"height": "400px"}
    )
])

if __name__ == '__main__':
    print("Starting test server on http://127.0.0.1:8052")
    print("Open your browser to that URL to verify connection")
    app.run(debug=False, port=8052, host='127.0.0.1')
