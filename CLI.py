from rich import print
import pyfiglet
from strip_polite import remove_polite_features
from spell_correct import correct_spell
from utils import load_resources

ascii_banner = pyfiglet.figlet_format("Politeness\nStripper", font="larry3d")
print(f"[purple]{ascii_banner}[/purple]")

print("[bold blue]Loading resources...[/bold blue]")
polite_features_dict, nlp = load_resources()
print("[bold green]Resources loaded successfully![/bold green]")

THRESHOLD = 0.3

while True:
    print("\n[bold yellow]Enter a sentence (or type 'exit' to quit):[/bold yellow]")
    sentence = input("> ").strip()

    if sentence.lower() == "exit":
        print("\n[bold green]Goodbye![/bold green]")
        break
    
    corrected_sentence = correct_spell(sentence)
    if corrected_sentence != sentence:
        print(f"\n[bold cyan]Text has been spell-corrected.[/bold cyan]")
    cleaned_sentence, removed_features = remove_polite_features(corrected_sentence, polite_features_dict, nlp, THRESHOLD)
    
    politeness_score = len(removed_features) / 10.0 if removed_features else 0.0
    politeness_score = min(politeness_score, 1.0) 
    
    print(f"\n[bold cyan]Politeness Score:[/bold cyan] [bold white]{politeness_score:.4f}[/bold white]")
    
    if removed_features:
        print("\n[bold magenta]Cleaned Sentence:[/bold magenta]")
        print(f"[white]{cleaned_sentence}[/white]")
        
        print("\n[bold magenta]Removed Features:[/bold magenta]")
        for feature in removed_features:
            print(f"[italic white]- {feature}[/italic white]")
    else:
        print("\n[bold magenta]Sentence does not include polite expressions:[/bold magenta]")
        print(f"[white]{cleaned_sentence}[/white]")