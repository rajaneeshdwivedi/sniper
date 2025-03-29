import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from sqlalchemy import create_engine
import argparse

# Define the app
app = dash.Dash(__name__, title="Trading Dashboard", 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Define colors
colors = {
    'background': '#F5F5F5',
    'text': '#333333',
    'grid': '#DDDDDD',
    'profit': '#4CAF50',
    'loss': '#F44336',
    'neutral': '#2196F3',
}

# Define layout with tabs
app.layout = html.Div([
    html.H1('Trading Dashboard', style={'textAlign': 'center', 'margin': '20px 0px'}),
    
    # Tabs layout
    dcc.Tabs([
        # Overview tab
        dcc.Tab(label='Overview', children=[
            html.Div([
                # Performance metrics
                html.Div([
                    html.H3('Performance Metrics'),
                    html.Div(id='performance-metrics')
                ], className='six columns'),
                
                # Latest trades
                html.Div([
                    html.H3('Latest Trades'),
                    html.Div(id='latest-trades')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                # Equity curve
                html.Div([
                    html.H3('Equity Curve'),
                    dcc.Graph(id='equity-curve')
                ], className='twelve columns')
            ], className='row'),
            
            html.Div([
                # PnL distribution
                html.Div([
                    html.H3('PnL Distribution'),
                    dcc.Graph(id='pnl-distribution')
                ], className='six columns'),
                
                # Win rate by symbol
                html.Div([
                    html.H3('Win Rate by Symbol'),
                    dcc.Graph(id='win-rate-by-symbol')
                ], className='six columns')
            ], className='row')
        ]),
        
        # Trading tab
        dcc.Tab(label='Trading Activity', children=[
            html.Div([
                # Date range filter
                html.Div([
                    html.H3('Date Range'),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=datetime(2020, 1, 1),
                        max_date_allowed=datetime.now(),
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now()
                    )
                ], className='twelve columns')
            ], className='row'),
            
            html.Div([
                # Trades table
                html.Div([
                    html.H3('Trade Details'),
                    html.Div(id='trades-table')
                ], className='twelve columns')
            ], className='row'),
            
            html.Div([
                # Trade duration chart
                html.Div([
                    html.H3('Trade Duration Distribution'),
                    dcc.Graph(id='trade-duration')
                ], className='six columns'),
                
                # Exit reasons chart
                html.Div([
                    html.H3('Exit Reasons'),
                    dcc.Graph(id='exit-reasons')
                ], className='six columns')
            ], className='row')
        ]),
        
        # Analysis tab
        dcc.Tab(label='Analysis', children=[
            html.Div([
                # Confidence vs. PnL
                html.Div([
                    html.H3('Confidence vs. PnL'),
                    dcc.Graph(id='confidence-vs-pnl')
                ], className='six columns'),
                
                # Entry hour performance
                html.Div([
                    html.H3('Performance by Entry Hour'),
                    dcc.Graph(id='hour-performance')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                # Symbol comparison
                html.Div([
                    html.H3('Symbol Comparison'),
                    dcc.Graph(id='symbol-comparison')
                ], className='twelve columns')
            ], className='row'),
            
            html.Div([
                # Strategy evaluation
                html.Div([
                    html.H3('Strategy Evaluation'),
                    dcc.Graph(id='strategy-evaluation')
                ], className='twelve columns')
            ], className='row')
        ]),
        
        # Live trading tab
        dcc.Tab(label='Live Trading', children=[
            html.Div([
                # Current portfolio
                html.Div([
                    html.H3('Current Portfolio'),
                    html.Div(id='current-portfolio')
                ], className='six columns'),
                
                # Open positions
                html.Div([
                    html.H3('Open Positions'),
                    html.Div(id='open-positions')
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                # Recent signals
                html.Div([
                    html.H3('Recent Signals'),
                    html.Div(id='recent-signals')
                ], className='twelve columns')
            ], className='row'),
            
            html.Div([
                # Market overview
                html.Div([
                    html.H3('Market Overview'),
                    dcc.Graph(id='market-overview')
                ], className='twelve columns')
            ], className='row')
        ])
    ], id='tabs'),
    
    # Hidden div for storing data
    html.Div(id='trades-data', style={'display': 'none'}),
    html.Div(id='metrics-data', style={'display': 'none'}),
    html.Div(id='portfolio-data', style={'display': 'none'}),
    
    # Update interval
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # in milliseconds (1 minute)
        n_intervals=0
    )
], style={'backgroundColor': colors['background'], 'margin': '0px', 'padding': '20px'})

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

# Connect to database
def connect_to_db(connection_string):
    return create_engine(connection_string)

# Load trades from database
def load_trades(db_engine):
    query = """
        SELECT * FROM paper_trades
        ORDER BY entry_time DESC
    """
    
    try:
        df = pd.read_sql_query(query, db_engine)
        
        # Convert datetime strings to datetime objects
        for col in ['entry_time', 'exit_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Parse additional_data JSON if present
        if 'additional_data' in df.columns:
            # Function to safely parse JSON
            def parse_json(json_str):
                if pd.isna(json_str):
                    return {}
                try:
                    return json.loads(json_str)
                except:
                    return {}
            
            # Apply parsing and extract common fields
            additional_data = df['additional_data'].apply(parse_json)
            
            # Extract probability if it exists
            if any('probability' in data for data in additional_data if data):
                df['probability'] = additional_data.apply(lambda x: x.get('probability', None))
        
        return df
    except Exception as e:
        print(f"Error loading trades: {e}")
        return pd.DataFrame()

# Load metrics from database
def load_metrics(db_engine):
    query = """
        SELECT * FROM trading_metrics
        ORDER BY timestamp DESC
    """
    
    try:
        df = pd.read_sql_query(query, db_engine)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Parse additional_data JSON if present
        if 'additional_data' in df.columns:
            def parse_json(json_str):
                if pd.isna(json_str):
                    return {}
                try:
                    return json.loads(json_str)
                except:
                    return {}
            
            df['additional_data_parsed'] = df['additional_data'].apply(parse_json)
        
        return df
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return pd.DataFrame()

# Calculate performance metrics
def calculate_metrics(trades_df):
    if trades_df.empty:
        return {}
    
    # Filter to closed trades only
    closed_trades = trades_df[trades_df['status'] == 'closed']
    
    if closed_trades.empty:
        return {}
    
    # Calculate metrics
    total_trades = len(closed_trades)
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    losing_trades = closed_trades[closed_trades['pnl'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    total_pnl = closed_trades['pnl'].sum()
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    avg_win = winning_trades['pnl'].mean() if win_count > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if loss_count > 0 else 0
    
    # Calculate profit factor
    gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Calculate holding times
    closed_trades['holding_time'] = (closed_trades['exit_time'] - closed_trades['entry_time'])
    avg_holding_time = closed_trades['holding_time'].mean().total_seconds() / 3600  # In hours
    
    # Calculate current equity from most recent metrics
    # This would normally come from the metrics database table
    current_equity = total_pnl + 10000  # Assuming 10k starting capital
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_holding_time': avg_holding_time,
        'current_equity': current_equity
    }

# Update trades data from database
@app.callback(
    Output('trades-data', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_trades_data(n):
    global db_engine
    df = load_trades(db_engine)
    return df.to_json(date_format='iso', orient='split')

# Update metrics data from database
@app.callback(
    Output('metrics-data', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_metrics_data(n):
    global db_engine
    df = load_metrics(db_engine)
    return df.to_json(date_format='iso', orient='split')

# Update performance metrics
@app.callback(
    Output('performance-metrics', 'children'),
    Input('trades-data', 'children'),
    Input('metrics-data', 'children')
)
def update_performance_metrics(trades_json, metrics_json):
    # Parse trades data
    if not trades_json:
        return html.Div("No trading data available")
    
    trades_df = pd.read_json(trades_json, orient='split')
    
    # Calculate metrics or get from metrics table
    if metrics_json:
        metrics_df = pd.read_json(metrics_json, orient='split')
        if not metrics_df.empty:
            metrics = {
                'total_trades': metrics_df['total_trades'].iloc[0] if 'total_trades' in metrics_df.columns else 0,
                'win_rate': metrics_df['win_rate'].iloc[0] if 'win_rate' in metrics_df.columns else 0,
                'total_pnl': (metrics_df['portfolio_value'].iloc[0] - 10000) if 'portfolio_value' in metrics_df.columns else 0,
                'current_equity': metrics_df['portfolio_value'].iloc[0] if 'portfolio_value' in metrics_df.columns else 10000,
                'profit_factor': metrics_df['profit_factor'].iloc[0] if 'profit_factor' in metrics_df.columns else 0,
                'expectancy': metrics_df['expectancy'].iloc[0] if 'expectancy' in metrics_df.columns else 0,
                'avg_holding_time': metrics_df['mean_holding_time'].iloc[0] if 'mean_holding_time' in metrics_df.columns else 0,
            }
        else:
            metrics = calculate_metrics(trades_df)
    else:
        metrics = calculate_metrics(trades_df)
    
    # Create metrics cards
    cards = html.Div([
        html.Div([
            html.Div([
                html.H4('Portfolio Value'),
                html.H2(f"${metrics.get('current_equity', 0):.2f}", style={'color': colors['neutral']})
            ], className='metric-card'),
            html.Div([
                html.H4('Total P&L'),
                html.H2(f"${metrics.get('total_pnl', 0):.2f}", 
                        style={'color': colors['profit'] if metrics.get('total_pnl', 0) > 0 else colors['loss']})
            ], className='metric-card'),
            html.Div([
                html.H4('Win Rate'),
                html.H2(f"{metrics.get('win_rate', 0):.1%}", 
                        style={'color': colors['neutral']})
            ], className='metric-card'),
            html.Div([
                html.H4('Profit Factor'),
                html.H2(f"{metrics.get('profit_factor', 0):.2f}", 
                        style={'color': colors['neutral']})
            ], className='metric-card'),
            html.Div([
                html.H4('Total Trades'),
                html.H2(f"{metrics.get('total_trades', 0)}", 
                        style={'color': colors['neutral']})
            ], className='metric-card'),
            html.Div([
                html.H4('Avg Holding Time'),
                html.H2(f"{metrics.get('avg_holding_time', 0):.1f} hrs", 
                        style={'color': colors['neutral']})
            ], className='metric-card')
        ], className='metrics-container')
    ])
    
    return cards

# Update latest trades table
@app.callback(
    Output('latest-trades', 'children'),
    Input('trades-data', 'children')
)
def update_latest_trades(trades_json):
    if not trades_json:
        return html.Div("No trades available")
    
    trades_df = pd.read_json(trades_json, orient='split')
    
    if trades_df.empty:
        return html.Div("No trades available")
    
    # Get latest 5 trades
    latest_trades = trades_df.head(5)
    
    # Format for display
    display_df = latest_trades[['symbol', 'entry_price', 'exit_price', 'pnl', 'status', 'entry_time']].copy()
    
    # Format values
    display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['entry_price'] = display_df['entry_price'].map('${:.2f}'.format)
    display_df['exit_price'] = display_df['exit_price'].map(lambda x: '${:.2f}'.format(x) if pd.notna(x) else '')
    display_df['pnl'] = display_df['pnl'].map(lambda x: '${:.2f}'.format(x) if pd.notna(x) else '')
    
    # Rename columns
    display_df.columns = ['Symbol', 'Entry Price', 'Exit Price', 'P&L', 'Status', 'Entry Time']
    
    # Create data table
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in display_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_header={
            'backgroundColor': colors['neutral'],
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'left'
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{Status} = "open"'},
                'backgroundColor': 'rgba(33, 150, 243, 0.1)'
            },
            {
                'if': {'filter_query': '{P&L} contains "-"'},
                'color': colors['loss']
            },
            {
                'if': {'filter_query': '{P&L} contains "$" && !({P&L} contains "-")'},
                'color': colors['profit']
            }
        ]
    )
    
    return table

# Update equity curve
@app.callback(
    Output('equity-curve', 'figure'),
    Input('metrics-data', 'children'),
    Input('trades-data', 'children')
)
def update_equity_curve(metrics_json, trades_json):
    if metrics_json:
        try:
            metrics_df = pd.read_json(metrics_json, orient='split')
            
            if not metrics_df.empty and 'timestamp' in metrics_df.columns and 'portfolio_value' in metrics_df.columns:
                # Sort by timestamp
                metrics_df = metrics_df.sort_values('timestamp')
                
                # Create figure
                fig = go.Figure()
                
                # Add equity curve
                fig.add_trace(go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=colors['neutral'], width=2)
                ))
                
                # Add starting equity
                fig.add_trace(go.Scatter(
                    x=metrics_df['timestamp'],
                    y=[10000] * len(metrics_df),  # Assuming 10k starting capital
                    mode='lines',
                    name='Initial Capital',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                # If we have trades data, add trade markers
                if trades_json:
                    trades_df = pd.read_json(trades_json, orient='split')
                    
                    if not trades_df.empty and 'exit_time' in trades_df.columns and 'pnl' in trades_df.columns:
                        # Get closed trades
                        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
                        
                        if not closed_trades.empty:
                            # Add winning trades
                            winning_trades = closed_trades[closed_trades['pnl'] > 0]
                            if not winning_trades.empty:
                                fig.add_trace(go.Scatter(
                                    x=winning_trades['exit_time'],
                                    y=[0] * len(winning_trades),  # Y position will be adjusted in update_layout
                                    mode='markers',
                                    name='Winning Trades',
                                    marker=dict(
                                        symbol='triangle-up',
                                        size=10,
                                        color=colors['profit'],
                                        line=dict(width=1, color='white')
                                    ),
                                    hoverinfo='text',
                                    hovertext=[f"{row['symbol']}: +${row['pnl']:.2f}" for _, row in winning_trades.iterrows()]
                                ))
                            
                            # Add losing trades
                            losing_trades = closed_trades[closed_trades['pnl'] <= 0]
                            if not losing_trades.empty:
                                fig.add_trace(go.Scatter(
                                    x=losing_trades['exit_time'],
                                    y=[0] * len(losing_trades),  # Y position will be adjusted in update_layout
                                    mode='markers',
                                    name='Losing Trades',
                                    marker=dict(
                                        symbol='triangle-down',
                                        size=10,
                                        color=colors['loss'],
                                        line=dict(width=1, color='white')
                                    ),
                                    hoverinfo='text',
                                    hovertext=[f"{row['symbol']}: ${row['pnl']:.2f}" for _, row in losing_trades.iterrows()]
                                ))
                
                # Update layout
                fig.update_layout(
                    title='Portfolio Equity Curve',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='white',
                    height=400
                )
                
                fig.update_xaxes(
                    showgrid=True,
                    gridcolor=colors['grid'],
                    zeroline=False
                )
                
                fig.update_yaxes(
                    showgrid=True,
                    gridcolor=colors['grid'],
                    zeroline=False
                )
                
                return fig
        except Exception as e:
            print(f"Error updating equity curve: {e}")
    
    # If we get here, create empty figure
    fig = go.Figure()
    
    fig.update_layout(
        title='Portfolio Equity Curve (No Data)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

# Update P&L distribution
@app.callback(
    Output('pnl-distribution', 'figure'),
    Input('trades-data', 'children')
)
def update_pnl_distribution(trades_json):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='P&L Distribution (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty or 'pnl' not in trades_df.columns:
            raise ValueError("No P&L data available")
        
        # Filter to closed trades only
        closed_trades = trades_df[trades_df['status'] == 'closed']
        
        if closed_trades.empty:
            raise ValueError("No closed trades available")
        
        # Create P&L bins
        pnl_values = closed_trades['pnl'].dropna()
        
        if len(pnl_values) < 2:
            # Not enough data for a proper histogram
            bins = 5
        else:
            # Determine appropriate number of bins
            range_val = pnl_values.max() - pnl_values.min()
            bins = min(max(5, int(range_val / 10)), 20)  # Between 5 and 20 bins
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pnl_values,
            nbinsx=bins,
            marker_color=colors['neutral'],
            name='P&L Distribution'
        ))
        
        # Add vertical line at zero
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=0, y1=1,
            yref='paper',
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='P&L Distribution',
            xaxis_title='P&L ($)',
            yaxis_title='Count',
            height=400,
            plot_bgcolor='white',
            bargap=0.1
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating P&L distribution: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='P&L Distribution (Error)',
            height=400,
            plot_bgcolor='white'
        )
        return fig

# Update win rate by symbol
@app.callback(
    Output('win-rate-by-symbol', 'figure'),
    Input('trades-data', 'children')
)
def update_win_rate_by_symbol(trades_json):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='Win Rate by Symbol (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty or 'pnl' not in trades_df.columns:
            raise ValueError("No P&L data available")
        
        # Filter to closed trades only
        closed_trades = trades_df[trades_df['status'] == 'closed']
        
        if closed_trades.empty:
            raise ValueError("No closed trades available")
        
        # Calculate win rate by symbol
        symbol_stats = {}
        
        for symbol in closed_trades['symbol'].unique():
            symbol_trades = closed_trades[closed_trades['symbol'] == symbol]
            wins = sum(symbol_trades['pnl'] > 0)
            total = len(symbol_trades)
            win_rate = wins / total if total > 0 else 0
            
            symbol_stats[symbol] = {
                'wins': wins,
                'total': total,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100
            }
        
        # Create DataFrame for plotting
        stats_df = pd.DataFrame.from_dict(symbol_stats, orient='index')
        stats_df = stats_df.sort_values('win_rate_pct', ascending=False)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=stats_df.index,
            x=stats_df['win_rate_pct'],
            orientation='h',
            marker_color=colors['neutral'],
            text=[f"{wr:.1f}% ({w}/{t})" for wr, w, t in 
                  zip(stats_df['win_rate_pct'], stats_df['wins'], stats_df['total'])],
            textposition='auto'
        ))
        
        # Add 50% line
        fig.add_shape(
            type='line',
            x0=50, y0=-0.5,
            x1=50, y1=len(stats_df) - 0.5,
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Win Rate by Symbol',
            xaxis_title='Win Rate (%)',
            yaxis_title='Symbol',
            height=400,
            plot_bgcolor='white',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        fig.update_xaxes(
            range=[0, 100],
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        fig.update_yaxes(
            showgrid=False,
            zeroline=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating win rate by symbol: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='Win Rate by Symbol (Error)',
            height=400,
            plot_bgcolor='white'
        )
        return fig

# Update trades table based on date range
@app.callback(
    Output('trades-table', 'children'),
    Input('trades-data', 'children'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_trades_table(trades_json, start_date, end_date):
    if not trades_json:
        return html.Div("No trades available")
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty:
            return html.Div("No trades available")
        
        # Convert string dates to datetime if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date
        
        # Filter by date range
        filtered_trades = trades_df[
            (trades_df['entry_time'] >= start_date) & 
            (trades_df['entry_time'] <= end_date)
        ]
        
        if filtered_trades.empty:
            return html.Div("No trades in selected date range")
        
        # Format for display
        display_df = filtered_trades[
            ['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 
             'pnl', 'pnl_percent', 'status', 'exit_reason', 'confidence']
        ].copy()
        
        # Format values
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['exit_time'] = display_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['entry_price'] = display_df['entry_price'].map('${:.2f}'.format)
        display_df['exit_price'] = display_df['exit_price'].map(lambda x: '${:.2f}'.format(x) if pd.notna(x) else '')
        display_df['pnl'] = display_df['pnl'].map(lambda x: '${:.2f}'.format(x) if pd.notna(x) else '')
        display_df['pnl_percent'] = display_df['pnl_percent'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else '')
        display_df['confidence'] = display_df['confidence'].map(lambda x: f"{x:.2f}" if pd.notna(x) else '')
        
        # Rename columns
        display_df.columns = ['Symbol', 'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 
                             'P&L', 'P&L %', 'Status', 'Exit Reason', 'Confidence']
        
        # Create data table
        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in display_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': colors['neutral'],
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'left'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Status} = "open"'},
                    'backgroundColor': 'rgba(33, 150, 243, 0.1)'
                },
                {
                    'if': {'filter_query': '{P&L} contains "-"'},
                    'color': colors['loss']
                },
                {
                    'if': {'filter_query': '{P&L} contains "$" && !({P&L} contains "-")'},
                    'color': colors['profit']
                }
            ],
            sort_action='native',
            filter_action='native',
            page_size=20
        )
        
        return table
    
    except Exception as e:
        print(f"Error updating trades table: {e}")
        return html.Div(f"Error loading trades: {str(e)}")

# Update trade duration chart
@app.callback(
    Output('trade-duration', 'figure'),
    Input('trades-data', 'children'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_trade_duration(trades_json, start_date, end_date):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='Trade Duration Distribution (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty:
            raise ValueError("No trades available")
        
        # Convert string dates to datetime if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date
        
        # Filter by date range and closed status
        filtered_trades = trades_df[
            (trades_df['entry_time'] >= start_date) & 
            (trades_df['entry_time'] <= end_date) &
            (trades_df['status'] == 'closed')
        ]
        
        if filtered_trades.empty:
            raise ValueError("No closed trades in selected date range")
        
        # Calculate durations in hours
        filtered_trades['duration'] = (filtered_trades['exit_time'] - filtered_trades['entry_time']).dt.total_seconds() / 3600
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=filtered_trades['duration'],
            marker_color=colors['neutral'],
            nbinsx=10,
            name='Trade Duration'
        ))
        
        # Add vertical line at mean
        mean_duration = filtered_trades['duration'].mean()
        fig.add_shape(
            type='line',
            x0=mean_duration, y0=0,
            x1=mean_duration, y1=1,
            yref='paper',
            line=dict(
                color='red',
                width=2,
                dash='dash'
            )
        )
        
        # Add annotation for mean
        fig.add_annotation(
            x=mean_duration,
            y=1,
            yref='paper',
            text=f"Mean: {mean_duration:.1f} hrs",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
        )
        
        # Update layout
        fig.update_layout(
            title='Trade Duration Distribution',
            xaxis_title='Duration (hours)',
            yaxis_title='Count',
            height=400,
            plot_bgcolor='white',
            bargap=0.1
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating trade duration chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='Trade Duration Distribution (Error)',
            height=400,
            plot_bgcolor='white'
        )
        return fig

# Update exit reasons chart
@app.callback(
    Output('exit-reasons', 'figure'),
    Input('trades-data', 'children'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_exit_reasons(trades_json, start_date, end_date):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='Exit Reasons (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty:
            raise ValueError("No trades available")
        
        # Convert string dates to datetime if needed
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date) + timedelta(days=1)  # Include the end date
        
        # Filter by date range and closed status
        filtered_trades = trades_df[
            (trades_df['entry_time'] >= start_date) & 
            (trades_df['entry_time'] <= end_date) &
            (trades_df['status'] == 'closed')
        ]
        
        if filtered_trades.empty:
            raise ValueError("No closed trades in selected date range")
        
        # Count exit reasons
        exit_reasons = filtered_trades['exit_reason'].value_counts().reset_index()
        exit_reasons.columns = ['exit_reason', 'count']
        
        # Replace None/NaN with 'unknown'
        exit_reasons['exit_reason'] = exit_reasons['exit_reason'].fillna('unknown')
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=exit_reasons['exit_reason'],
            y=exit_reasons['count'],
            marker_color=colors['neutral'],
            text=exit_reasons['count'],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title='Exit Reasons',
            xaxis_title='Exit Reason',
            yaxis_title='Count',
            height=400,
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(
            showgrid=False,
            zeroline=False
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating exit reasons chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='Exit Reasons (Error)',
            height=400,
            plot_bgcolor='white'
        )
        return fig

# Update confidence vs P&L chart
@app.callback(
    Output('confidence-vs-pnl', 'figure'),
    Input('trades-data', 'children')
)
def update_confidence_vs_pnl(trades_json):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='Confidence vs P&L (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty or 'confidence' not in trades_df.columns:
            raise ValueError("No confidence data available")
        
        # Filter to closed trades only
        closed_trades = trades_df[
            (trades_df['status'] == 'closed') & 
            (trades_df['confidence'].notna()) & 
            (trades_df['pnl'].notna())
        ]
        
        if closed_trades.empty:
            raise ValueError("No closed trades with confidence data available")
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add winning trades
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        if not winning_trades.empty:
            fig.add_trace(go.Scatter(
                x=winning_trades['confidence'],
                y=winning_trades['pnl_percent'],
                mode='markers',
                name='Winning Trades',
                marker=dict(
                    size=10,
                    color=colors['profit'],
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                hoverinfo='text',
                hovertext=[f"{row['symbol']}: +{row['pnl_percent']:.2f}%" for _, row in winning_trades.iterrows()]
            ))
        
        # Add losing trades
        losing_trades = closed_trades[closed_trades['pnl'] <= 0]
        if not losing_trades.empty:
            fig.add_trace(go.Scatter(
                x=losing_trades['confidence'],
                y=losing_trades['pnl_percent'],
                mode='markers',
                name='Losing Trades',
                marker=dict(
                    size=10,
                    color=colors['loss'],
                    symbol='circle',
                    line=dict(width=1, color='white')
                ),
                hoverinfo='text',
                hovertext=[f"{row['symbol']}: {row['pnl_percent']:.2f}%" for _, row in losing_trades.iterrows()]
            ))
        
        # Add trend line
        if len(closed_trades) >= 5:
            x = closed_trades['confidence']
            y = closed_trades['pnl_percent']
            
            # Simple linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            x_range = np.linspace(x.min(), x.max(), 100)
            y_range = slope * x_range + intercept
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                name='Trend',
                line=dict(color='black', width=2, dash='dash')
            ))
            
            # Add correlation coefficient
            correlation = np.corrcoef(x, y)[0, 1]
            
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref='paper',
                yref='paper',
                text=f"Correlation: {correlation:.2f}",
                showarrow=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1
            )
        
        # Add horizontal line at 0
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=1, y1=0,
            xref='paper',
            line=dict(
                color='black',
                width=1,
                dash='dot'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Confidence vs P&L %',
            xaxis_title='Confidence',
            yaxis_title='P&L %',
            height=400,
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False,
            range=[0, 1]
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating confidence vs P&L chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='Confidence vs P&L (Error)',
            height=400,
            plot_bgcolor='white'
        )
        return fig

# Update hour performance chart
@app.callback(
    Output('hour-performance', 'figure'),
    Input('trades-data', 'children')
)
def update_hour_performance(trades_json):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='Performance by Entry Hour (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty:
            raise ValueError("No trades available")
        
        # Filter to closed trades only
        closed_trades = trades_df[trades_df['status'] == 'closed']
        
        if closed_trades.empty:
            raise ValueError("No closed trades available")
        
        # Extract hour of day
        closed_trades['hour'] = closed_trades['entry_time'].dt.hour
        
        # Calculate metrics by hour
        hour_stats = []
        
        for hour in range(24):
            hour_trades = closed_trades[closed_trades['hour'] == hour]
            
            if len(hour_trades) > 0:
                win_rate = hour_trades[hour_trades['pnl'] > 0].shape[0] / hour_trades.shape[0]
                avg_pnl = hour_trades['pnl'].mean()
                count = len(hour_trades)
                
                hour_stats.append({
                    'hour': hour,
                    'win_rate': win_rate * 100,
                    'avg_pnl': avg_pnl,
                    'count': count
                })
        
        hour_stats_df = pd.DataFrame(hour_stats)
        
        if hour_stats_df.empty:
            raise ValueError("No hour statistics available")
        
        # Create figure with two y-axes
        fig = go.Figure()
        
        # Add win rate
        fig.add_trace(go.Bar(
            x=hour_stats_df['hour'],
            y=hour_stats_df['win_rate'],
            name='Win Rate (%)',
            marker_color=colors['neutral'],
            opacity=0.7,
            text=[f"{wr:.1f}%" for wr in hour_stats_df['win_rate']],
            textposition='auto'
        ))
        
        # Create second y-axis for trade count
        fig.add_trace(go.Scatter(
            x=hour_stats_df['hour'],
            y=hour_stats_df['count'],
            name='Trade Count',
            mode='lines+markers',
            yaxis='y2',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        # Update layout with second y-axis
        fig.update_layout(
            title='Performance by Entry Hour',
            xaxis_title='Hour of Day (UTC)',
            yaxis_title='Win Rate (%)',
            yaxis2=dict(
                title='Trade Count',
                overlaying='y',
                side='right'
            ),
            height=400,
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            dtick=2,  # Show every 2 hours
            range=[-0.5, 23.5]
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False,
            range=[0, 100]
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating hour performance chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='Performance by Entry Hour (Error)',
            height=400,
            plot_bgcolor='white'
        )
        return fig

# Update symbol comparison chart
@app.callback(
    Output('symbol-comparison', 'figure'),
    Input('trades-data', 'children')
)
def update_symbol_comparison(trades_json):
    if not trades_json:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title='Symbol Comparison (No Data)',
            height=400,
            plot_bgcolor='white'
        )
        return fig
    
    try:
        trades_df = pd.read_json(trades_json, orient='split')
        
        if trades_df.empty:
            raise ValueError("No trades available")
        
        # Filter to closed trades only
        closed_trades = trades_df[trades_df['status'] == 'closed']
        
        if closed_trades.empty:
            raise ValueError("No closed trades available")
        
        # Calculate metrics by symbol
        symbol_stats = []
        
        for symbol in closed_trades['symbol'].unique():
            symbol_trades = closed_trades[closed_trades['symbol'] == symbol]
            
            if len(symbol_trades) >= 3:  # Require at least 3 trades for meaningful stats
                win_rate = symbol_trades[symbol_trades['pnl'] > 0].shape[0] / symbol_trades.shape[0]
                avg_pnl = symbol_trades['pnl'].mean()
                avg_pnl_pct = symbol_trades['pnl_percent'].mean()
                count = len(symbol_trades)
                total_pnl = symbol_trades['pnl'].sum()
                
                symbol_stats.append({
                    'symbol': symbol,
                    'win_rate': win_rate * 100,
                    'avg_pnl': avg_pnl,
                    'avg_pnl_pct': avg_pnl_pct,
                    'count': count,
                    'total_pnl': total_pnl
                })
        
        symbol_stats_df = pd.DataFrame(symbol_stats)
        
        if symbol_stats_df.empty:
            raise ValueError("No symbol statistics available")
        
        # Sort by total PnL
        symbol_stats_df = symbol_stats_df.sort_values('total_pnl', ascending=False)
        
        # Create bubble chart
        fig = go.Figure()
        
        # Add bubbles
        fig.add_trace(go.Scatter(
            x=symbol_stats_df['win_rate'],
            y=symbol_stats_df['avg_pnl_pct'],
            mode='markers',
            marker=dict(
                size=symbol_stats_df['count'] * 5,  # Scale bubble size by trade count
                sizemode='area',
                sizeref=2.*max(symbol_stats_df['count'])/(40.**2),
                sizemin=5,
                color=symbol_stats_df['total_pnl'],
                colorscale='RdYlGn',
                colorbar=dict(
                    title='Total P&L ($)'
                ),
                line=dict(width=1, color='white')
            ),
            text=[f"{s}: {c} trades, ${p:.2f}" for s, c, p in 
                  zip(symbol_stats_df['symbol'], symbol_stats_df['count'], symbol_stats_df['total_pnl'])],
            hoverinfo='text'
        ))
        
        # Add quadrant lines
        fig.add_shape(
            type='line',
            x0=50, y0=min(symbol_stats_df['avg_pnl_pct'].min(), 0),
            x1=50, y1=max(symbol_stats_df['avg_pnl_pct'].max(), 0),
            line=dict(
                color='black',
                width=1,
                dash='dash'
            )
        )
        
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=100, y1=0,
            line=dict(
                color='black',
                width=1,
                dash='dash'
            )
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=25, y=symbol_stats_df['avg_pnl_pct'].max() * 0.8,
            text="Low Win Rate,<br>High Avg P&L",
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=75, y=symbol_stats_df['avg_pnl_pct'].max() * 0.8,
            text="High Win Rate,<br>High Avg P&L",
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=25, y=symbol_stats_df['avg_pnl_pct'].min() * 0.8,
            text="Low Win Rate,<br>Low Avg P&L",
            showarrow=False,
            font=dict(size=10)
        )
        
        fig.add_annotation(
            x=75, y=symbol_stats_df['avg_pnl_pct'].min() * 0.8,
            text="High Win Rate,<br>Low Avg P&L",
            showarrow=False,
            font=dict(size=10)
        )
        
        # Update layout
        fig.update_layout(
            title='Symbol Comparison',
            xaxis_title='Win Rate (%)',
            yaxis_title='Avg P&L (%)',
            height=500,
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False,
            range=[0, 100]
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor=colors['grid'],
            zeroline=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error updating symbol comparison chart: {e}")
        fig = go.Figure()
        fig.update_layout(
            title='Symbol Comparison (Error)',
            height=500,
            plot_bgcolor='white'
        )
        return fig

# Initialize global variables
db_engine = None
config = None

def main():
    global db_engine, config
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Dashboard')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default configuration
        config = {
            'database': {
                'connection_string': 'mysql+pymysql://ctUser:-023poqw-023@127.0.0.1/ct'
            }
        }
    
    # Connect to database
    db_engine = connect_to_db(config['database']['connection_string'])
    
    # Run dashboard
    app.run_server(debug=True, port=args.port)

if __name__ == '__main__':
    main()