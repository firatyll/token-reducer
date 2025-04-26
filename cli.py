import sys
import typer
from rules import trim

app = typer.Typer(help="Token-reducer CLI")

@app.command()
def run() -> None:
    for line in sys.stdin:
        print(trim(line.rstrip()))

if __name__ == "__main__":
    app()
