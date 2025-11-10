import json
from divide21x.challenge_maker.challenge_maker import ChallengeMaker
from divide21x.llm_api.client_class import ModelClient
from divide21x.llm_api.requestor import Requestor
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_llm_registry


# base dir
BASE_DIR = './tests/llm_api/logs'
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

# requestor
requestor = Requestor()


def test_llm(registry_entry):
    client = ModelClient(
        registry_entry=registry_entry
    )
    if client.client is None:
        return
    
    # get prompt
    prompt, system_prompt = requestor.get_prompt()

    # Ask the LLM
    answer = client.chat(prompt, system_prompt=system_prompt)
    logger.add_info(client.model_alias, ANSWER, answer)



if __name__ == "__main__":
    # create challenge if it does not exist
    challenge_maker = ChallengeMaker()
    challenge_maker.make_challenge()
    
    # Load LLM registry
    llm_id = "google-gemini25-pro"
    registry = get_llm_registry()
    if registry:
        # start the requests
        for registry_entry in registry:
            if registry_entry["id"] == llm_id:
                test_llm(registry_entry)
                break
            