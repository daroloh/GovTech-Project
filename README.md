# HDB BTO Pricing System (Backend Only)

This project ingests HDB resale CSVs into DuckDB, trains a price prediction model (RandomForest), and serves an API that:
- Predicts resale price for a given unit configuration
- Converts predicted resale price to BTO prices via a configurable discount
- Estimates household income bands
- Recommends towns with limited recent activity (proxy for fewer BTO launches)
- Optionally generates short natural-language explanations using an LLM (OpenAI)
- Provides a composite endpoint to answer the prompt: recommend estates with limited BTO launches and analyze 3- and 4-room prices by floor bands and income.

## Data
Place the provided CSV files in the project root (already present). The ETL detects schema differences and standardizes columns.

## Requirements
- Python 3.10+ on Windows
- PowerShell

## Quickstart (Windows PowerShell)

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Set your OpenAI key via .env (see below)

# 4) Run ETL into DuckDB and build features
python cli.py etl

# 5) Train the model
python cli.py train

# 6) Start the API
python cli.py serve --host 0.0.0.0 --port 8000
```

Open your browser at `http://localhost:8000/docs` for the interactive API UI.

## API Endpoints
- `GET /health` — basic health check
- `GET /metrics` — returns training metrics
- `POST /predict` — body:
  ```json
  {
    "town": "ANG MO KIO",
    "flat_type": "4 ROOM",
    "floor_area_sqm": 95,
    "storey_mid": 12,
    "flat_model": "Improved",
    "lease_commence_date": 1990,
    "year": 2023,
    "month_num": 6
  }
  ```
- `GET /recommend?limit=5&flat_types=3%20ROOM&flat_types=4%20ROOM` — suggests candidate towns with lower recent activity
- `POST /bto_analysis` — body:
  ```json
  {
    "towns": ["ANG MO KIO", "BEDOK"],
    "flat_types": ["3 ROOM", "4 ROOM"],
    "low_floor": 5,
    "mid_floor": 12,
    "high_floor": 25
  }
  ```
- `GET /report_md` — returns a Markdown report; query example:
  ```
  /report_md?flat_types=3%20ROOM,4%20ROOM&low_floor=5&mid_floor=12&high_floor=25&limit=5
  ```

## PowerShell request examples
- Health:
  ```powershell
  Invoke-RestMethod "http://localhost:8000/health"
  ```
- BTO analysis (PowerShell-native JSON):
  ```powershell
  $body = @{ towns=@("ANG MO KIO","BEDOK"); flat_types=@("3 ROOM","4 ROOM"); low_floor=5; mid_floor=12; high_floor=25 } | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Uri "http://localhost:8000/bto_analysis" -Method POST -ContentType "application/json" -Body $body
  ```
- If you prefer curl.exe with a file:
  ```powershell
  @'
  {
    "towns": ["ANG MO KIO", "BEDOK"],
    "flat_types": ["3 ROOM", "4 ROOM"],
    "low_floor": 5,
    "mid_floor": 12,
    "high_floor": 25
  }
  '@ | Set-Content -Encoding UTF8 .\bto.json
  curl.exe -s -X POST "http://localhost:8000/bto_analysis" -H "Content-Type: application/json" --data-binary "@bto.json"
  ```

## CLI commands
- ETL: `python cli.py etl`
- Train: `python cli.py train`
- Serve API: `python cli.py serve --host 0.0.0.0 --port 8000`
- Generate Markdown report (auto-select towns):
  ```powershell
  python cli.py report
  ```
- Generate report with specified towns and floors:
  ```powershell
  python cli.py report --towns "ANG MO KIO,BEDOK,QUEENSTOWN" --flat-types "3 ROOM,4 ROOM" --low-floor 5 --mid-floor 12 --high-floor 25 --limit 5 --output artifacts/bto_report.md
  ```

## Configuration
Edit `config.yaml` to change paths, model hyperparameters, and discount rate. Artifacts are stored under `artifacts/` and `data/`.

### LLM API key via .env
Create a `.env` file in the project root (same folder as `cli.py`) with:
```
OPENAI_API_KEY=sk-your-key-here
```
The app auto-loads `.env` on startup. Alternatively, set it in the shell:
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```
If not set, a deterministic fallback explanation string is used.

### Tuning training speed vs accuracy
In `config.yaml`:
```yaml
training:
  n_estimators: 100   # increase to 300+ for higher accuracy
  max_depth: 12       # set to null for deeper trees (slower)
```
Then re-run:
```powershell
python cli.py train
```

## Outputs
- DuckDB database: `data/hdb.duckdb`
- Model artifact: `artifacts/models/rf_pipeline.joblib`
- Metrics: `artifacts/metrics.json`
- Logs (if any): `artifacts/logs/`
- Markdown report: `artifacts/bto_report.md`

## Troubleshooting
- Port already in use (500/addr-in-use): either stop the old server or run a different port:
  ```powershell
  python cli.py serve --host 0.0.0.0 --port 8001
  ```
- PowerShell quoting issues: prefer `Invoke-RestMethod` with `ConvertTo-Json` as shown above.
- 500 Internal Server Error: check the server console for the traceback. The API returns a concise error message; share it for quick fixes.
- No LLM key: responses use fallback explanations; set `OPENAI_API_KEY` to enable LLM.

## Testing
After running ETL and training, run:
```powershell
pytest -q
```

## Notes and Next Steps
- The income estimation uses a simple heuristic; replace with a proper mortgage affordability calculator if needed.
- The recommendation endpoint uses recent resale transaction counts as a proxy; refine with actual BTO launch data if available.
- Add CI, model versioning strategy, and data drift monitoring in production.