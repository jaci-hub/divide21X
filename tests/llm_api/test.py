from divide21x.llm_api.llm_client import ModelClient

client = ModelClient(provider="openai", model="gpt-4o")

prompt = """
Given the Divide21 game state:
{
    "static_number": 19,
    "dynamic_number": 59,
    "available_digits_per_rindex": {0: [1, 2, 3], 1: [4, 5, 6]},
    "players": [{"id": 0, "score": 0, "is_current_turn": 1}],
    "player_turn": 0
}
Suggest the next best action as a JSON object with keys: division, digit, rindex.
"""

print(client.chat(prompt))
