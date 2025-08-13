import os
import typer
import uvicorn

from hdb.config import load_config
from hdb.etl import load_csvs_to_duckdb
from hdb.train import train_model
from hdb.report import generate_bto_report

app = typer.Typer(help="HDB BTO pricing pipeline")


@app.command()
def etl():
	"""Load CSVs into DuckDB and build features."""
	load_csvs_to_duckdb()
	typer.echo("ETL completed.")


@app.command()
def train():
	"""Train model and save metrics."""
	path, metrics = train_model()
	typer.echo(f"Model saved: {path}")
	typer.echo(metrics)


@app.command()
def serve(host: str = None, port: int = None):
	"""Start FastAPI server."""
	cfg = load_config()
	host = host or cfg.api.host
	port = port or cfg.api.port
	uvicorn.run("hdb.serve:app", host=host, port=port, reload=False)


@app.command()
def report(
	owns: str = typer.Option(None, help="Comma-separated towns. If omitted, auto-recommend."),
	flat_types: str = typer.Option("3 ROOM,4 ROOM", help="Comma-separated flat types."),
	low_floor: float = typer.Option(5, help="Low floor midpoint."),
	mid_floor: float = typer.Option(12, help="Mid floor midpoint."),
	high_floor: float = typer.Option(25, help="High floor midpoint."),
	limit: int = typer.Option(5, help="If no towns provided, number of towns to auto-select."),
	output: str = typer.Option("artifacts/bto_report.md", help="Output Markdown path."),
):
	"""Generate a Markdown report with BTO recommendations and price analysis."""
	town_list = [t.strip() for t in owns.split(",")] if owns else None
	ft_list = [t.strip() for t in flat_types.split(",")]
	md = generate_bto_report(
		towns=town_list,
		flat_types=ft_list,
		low_floor=low_floor,
		mid_floor=mid_floor,
		high_floor=high_floor,
		limit_if_recommend=limit,
		output_path=output,
	)
	typer.echo(f"Report generated at {output}")


if __name__ == "__main__":
	app()
