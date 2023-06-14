import altair as alt

def colorMap(
    df,
    x='x:O', 
    y='y:O',
    color='value:Q',
    scheme='viridis'
):
    return (
        alt.Chart(df)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X(x),
            alt.Y(y),
            alt.Color(color, scale=alt.Scale(scheme=scheme)),
        )
        .interactive()
    )