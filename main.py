from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import run_tournament, play

# Read all files within subfolder of "groups"
participants = find_importable_classes("groups", Policy)

# Build a participant list (name, class) for GUTI vs JUAN
players = [
    ("Group B GUTI", participants["Group B GUTI"]),
    ("Group A JUAN", participants["Group A JUAN"]),
]

# Run the tournament
champion = run_tournament(
    players,
    play,  # You could also create your own play function for testing purposes
    shuffle=False,
)
print("Champion:", champion)
