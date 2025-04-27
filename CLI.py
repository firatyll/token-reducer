from rich import print
import pyfiglet
from strip_polite import strip_polite

ascii_banner = pyfiglet.figlet_format("Politeness\nStripper", font="larry3d")
print(f"[purple]{ascii_banner}[/purple]")

while True:
    print("\n[bold yellow]Enter a sentence (or type 'exit' to quit):[/bold yellow]")
    sentence = input("> ").strip()

    if sentence.lower() == "exit":
        print("\n[bold green]Goodbye![/bold green]")
        break
    cleaned_sentence = strip_polite(sentence)
    
    print("\n[bold magenta]Cleaned Sentence:[/bold magenta]")
    print(f"[white]{cleaned_sentence}[/white]")
