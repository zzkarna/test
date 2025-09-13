# MVP Quant Dashboard

A containerized quantitative finance dashboard that streams live crypto data from Binance, pulls equity and options data from yfinance, and computes comprehensive financial metrics in real-time.

## Features

- **Real-time Data Streaming**: Live crypto data from Binance WebSocket
- **Equity & Options Data**: Historical and current data via yfinance
- **Comprehensive Analytics**: 50+ financial metrics and indicators
- **Risk Management**: VaR, Expected Shortfall, drawdown analysis
- **Options Analytics**: Greeks, implied volatility surfaces, put-call parity
- **Technical Indicators**: SMA/EMA, MACD, RSI, Bollinger Bands, and more
- **Portfolio Analytics**: Performance attribution, correlation analysis
- **Real-time Alerts**: Crossover signals, volatility regime changes

## Architecture

The application consists of 4 microservices:

- **Ingestor**: Streams data from Binance WebSocket and yfinance
- **Analytics**: FastAPI service computing financial metrics
- **Dashboard**: Streamlit web interface
- **Scheduler**: Background jobs for data refresh and maintenance

Data is persisted in DuckDB and Parquet files for fast analytics.

## Quick Start

1. **Clone and Setup**:
   \`\`\`bash
   git clone <repository>
   cd quant-dashboard
   \`\`\`

2. **Start Services**:
   \`\`\`bash
   docker compose up
   \`\`\`

3. **Access Dashboard**:
   Open http://localhost:8501 in your browser

## Configuration

Edit `config.yaml` to customize:

- Symbols to track (crypto and equities)
- Options underlyings and expiries
- Risk parameters (VaR confidence, lookback periods)
- Data refresh intervals

## Data Storage

- **Parquet Files**: `./data/parquet/` - Partitioned by symbol/source/date
- **DuckDB**: `./data/duckdb/market.duckdb` - Fast analytical queries
- **Provenance**: All records include source, fetched_at, ingest_run_id

## API Endpoints

Analytics service (http://localhost:8000):

- `GET /health` - Service health check
- `GET /metrics/{symbol}` - Time-series metrics
- `GET /risk/{symbol}` - Risk analytics
- `GET /options/{underlying}` - Options analytics
- `GET /signals` - Trading signals and alerts

## Development

### Local Development

1. **Install Dependencies**:
   \`\`\`bash
   poetry install
   \`\`\`

2. **Run Services Individually**:
   \`\`\`bash
   # Analytics API
   cd analytics && uvicorn api:app --reload --port 8000
   
   # Dashboard
   cd dashboard && streamlit run app.py --server.port 8501
   
   # Ingestor
   cd ingestor && python -m crypto_binance
   \`\`\`

### Testing

\`\`\`bash
poetry run pytest tests/
\`\`\`

## Financial Metrics Computed

### Time-Series Analysis
- Simple/Log returns with annualization
- Rolling volatility (realized, EWMA, Garman-Klass, Parkinson)
- Autocorrelation and stationarity tests

### Risk Metrics
- Sharpe, Sortino, Calmar, Information Ratios
- Value-at-Risk (VaR) and Expected Shortfall
- Maximum Drawdown and recovery analysis
- Beta and alpha vs benchmarks

### Technical Indicators
- Moving averages (SMA/EMA/WMA)
- Momentum indicators (MACD, RSI, ATR)
- Volatility bands (Bollinger Bands)
- Support/resistance levels

### Options Analytics
- Black-Scholes Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility surfaces and term structures
- Put-call parity validation
- Moneyness and skew analysis

### Portfolio Analytics
- Equal-weight portfolio construction
- Performance attribution by asset
- Correlation matrices and factor exposure
- Kelly criterion position sizing

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 8000 and 8501 are available
2. **Data Directory**: Verify `./data/` directory has write permissions
3. **Memory Usage**: DuckDB may require 2GB+ RAM for large datasets
4. **API Limits**: yfinance has rate limits; scheduler respects these

### Logs

View service logs:
\`\`\`bash
docker compose logs -f [service_name]
\`\`\`

### Health Checks

Check service status:
\`\`\`bash
curl http://localhost:8000/health
curl http://localhost:8501/health
\`\`\`

## License

MIT License - see LICENSE file for details.
