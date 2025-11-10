import json
from divide21x.challenge_maker.challenge_maker import ChallengeMaker
from divide21x.llm_api.client_class import ModelClient
from divide21x.utils.util import get_utc_date, get_utc_datetime, get_utc_hour

if __name__ == "__main__":
    # create challenge if it does not exist
    challenge_maker = ChallengeMaker()
    challenge_maker.make_challenge()
    
    # Load LLM registry
    with open("divide21x/llm_api/registry.json", "r") as f:
        registry = json.load(f)

    # Pick a model by ID
    model_info = next(item for item in registry if item["id"] == "openai-o1")

    client = ModelClient(
        model_id=model_info["id"]
    )

    # Load the challenge JSON
    utc_datetime = get_utc_datetime()
    hour = str(get_utc_hour())
    hour_2 = int(hour) + 2
    date = str(get_utc_date())
    
    with open(f"divide21x/challenges/{date}/{hour}.json", "r") as f:
        challenge_data = json.load(f)

    # Construct the few-shot + challenge prompt
    prompt_lines = []
    for example_key in ["example_1", "example_2"]:
        ex = challenge_data[example_key]
        prompt_lines.append(f"Example:\nInitial state: {json.dumps(ex['initial_state'])}\n"
                            f"Action: {json.dumps(ex['action'])}\n"
                            f"Final state: {json.dumps(ex['final_state'])}\n")

    # Add the challenge (no final_state)
    challenge = challenge_data["challenge"]
    prompt_lines.append(f"Challenge:\nInitial state: {json.dumps(challenge['initial_state'])}\n"
                        f"Action: {json.dumps(challenge['action'])}\n"
                        f"Final state: ? (compute this and return as JSON)")

    prompt = "\n\n".join(prompt_lines)

    system_prompt = (
        "Given an initial state and an action, "
        "compute the resulting final state. ONLY return a valid JSON object."
    )

    print(prompt)
    # Ask the LLM
    result = client.chat(prompt, system_prompt=system_prompt)
    print("LLM predicted final_state:\n", result)
