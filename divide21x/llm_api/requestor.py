import json
from divide21x.challenge_maker.challenge_maker import ChallengeMaker
from divide21x.llm_api.client_class import ModelClient
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_llm_registry, get_utc_date, get_utc_datetime, get_utc_hour


# base dir
BASE_DIR = './divide21x/llm_api/logs'
# categories
REQUESTOR = 'requestor'
API = 'api'
CHAT = 'chat'
NOTE = 'note'
PROMPT = 'prompt'
SYSTEM_PROMPT = 'system_prompt'
ANSWER = 'answer'
# types
CRITICAL = 'critical'
WARNING = 'warning'

# Logging
logger = EpisodeLogger(BASE_DIR)


def get_prompt():
    # get date and time
    date = str(get_utc_date())
    hour = str(get_utc_hour())
    
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
    
    # log
    logger.add_info(REQUESTOR, PROMPT, prompt)
    logger.add_info(REQUESTOR, SYSTEM_PROMPT, system_prompt)
    
    return prompt, system_prompt
    

def start_request(registry_entry, prompt, system_prompt):
    client = ModelClient(
        registry_entry=registry_entry
    )
    if client.client is None:
        return

    # Request the LLM   
    answer = client.chat(prompt, system_prompt=system_prompt) 
    logger.add_info(client.model_alias, ANSWER, answer)



if __name__ == "__main__":
    # Load LLM registry
    registry = get_llm_registry()
    
    if registry:
        # get prompt
        prompt, system_prompt = get_prompt()
        
        # start the requests
        for registry_entry in registry:
            start_request(registry_entry, prompt, system_prompt)
        
        # log
        if logger.info not in logger.episode_log:
            logger.episode_log.append(logger.info)
        logger.save_episode()
        