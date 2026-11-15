import plotly.express as px
import plotly.graph_objects as go

# Shared light editorial theme used by every chart in TrafficSense.
DARK_LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans, sans-serif", color="#263b34", size=13),
    title=dict(font=dict(family="Manrope, sans-serif", size=18, color="#172b2a")),
    margin=dict(l=24,r=24,t=56,b=24),
    colorway=["#147D64", "#e0a34c", "#4b91a8", "#8a78b5", "#d47c70", "#6f9c79"],
    xaxis=dict(showgrid=False,zeroline=False,linecolor="#dce6df"),
    yaxis=dict(gridcolor="#e7eee9",zeroline=False,linecolor="#dce6df"),
    legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#40564d")))
LIGHT_LAYOUT = DARK_LAYOUT

def line_chart(data,x,y,title,color=None):
    fig=px.line(data,x=x,y=y,color=color,title=title); fig.update_layout(**DARK_LAYOUT); return fig

def bar_chart(data,x,y,title,color=None):
    fig=px.bar(data,x=x,y=y,color=color,title=title); fig.update_layout(**DARK_LAYOUT); return fig
