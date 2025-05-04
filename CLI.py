from rich import print
import pyfiglet
from tfIdf import *

ascii_banner = pyfiglet.figlet_format("Politeness\nStripper", font="larry3d")
print(f"[purple]{ascii_banner}[/purple]")

while True:
    print("\n[bold yellow]Enter a sentence (or type 'exit' to quit):[/bold yellow]")
    sentence = input("> ").strip()

    if sentence.lower() == "exit":
        print("\n[bold green]Goodbye![/bold green]")
        break
    cleaned_sentence , probab = tfIdf(sentence)
    print(f"\n[bold cyan]Politeness Score:[/bold cyan] [bold white]{probab}[/bold white]")
    if probab > THRESHOLD:
        print("\n[bold magenta]Cleaned Sentence:[/bold magenta]")
        print(f"[white]{cleaned_sentence}[/white]")
    else:
        print("\n[bold magenta]Sentence does not include polite expressions:[/bold magenta]")
        print(f"[white]{cleaned_sentence}[/white]")