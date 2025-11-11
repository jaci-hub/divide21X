import hashlib
import json
import os
from divide21x.llm_api.client_class import ModelClient
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_llm_registry, get_utc_date, get_utc_datetime, get_utc_hour


RESULTS_DIR = './divide21x/results'
BASE_DIR = './divide21x/llm_api/logs'
# categories
REQUESTOR = 'requestor'
API = 'api'
CHAT = 'chat'
NOTE = 'note'
PROMPT = 'prompt'
SYSTEM_PROMPT = 'system_prompt'
ANSWER = 'answer'
RESULTS = 'results'
ID = 'id'
HASH = 'hash'
# types
CRITICAL = 'critical'
WARNING = 'warning'


class Requestor():
    def __init__(self):
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
        
        self.registry = None
        self.prompt = None
        self.system_prompt = None
        
        self.result_dir = None
        
        # get date and time
        self.date = str(get_utc_date())
        self.hour = str(get_utc_hour())
        
        self.results = {}
        
    def get_prompt(self):
        try:
            with open(f"divide21x/challenges/{self.date}/{self.hour}.json", "r") as f:
                challenge_data = json.load(f)
        except FileNotFoundError:
            challenge_data = None
            message = "Challenge has not been created yet!"
            self.logger.add_info(REQUESTOR, CRITICAL, message)
            return self.prompt, self.system_prompt

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

        self.prompt = "\n\n".join(prompt_lines)

        self.system_prompt = (
            "Given an initial state and an action, "
            "compute the resulting final state. ONLY return a valid JSON object."
        )
        
        # log
        self.logger.add_info(REQUESTOR, PROMPT, self.prompt)
        self.logger.add_info(REQUESTOR, SYSTEM_PROMPT, self.system_prompt)
        
        return self.prompt, self.system_prompt
        

    def prompt_llm(self, registry_entry):
        client = ModelClient(
            registry_entry=registry_entry
        )
        if client.client is None:
            return

        # Request the LLM   
        answer = client.chat(self.prompt, system_prompt=self.system_prompt)
        
        # record results
        if client.model_alias not in self.results:
            self.results[client.model_alias] = {}
            self.results[client.model_alias][ANSWER] = answer
        
        # log
        self.logger.add_info(client.model_alias, ANSWER, answer)


    def start_request(self):
        self.registry = get_llm_registry()
        
        if self.registry:
            # get prompt
            self.prompt, self.system_prompt = self.get_prompt()
            
            if self.prompt and self.system_prompt:
                # start the requests
                for registry_entry in self.registry:
                    self.prompt_llm(registry_entry)
            
                # write to results dir
                if self.results:
                    self.result_dir = os.path.join(RESULTS_DIR, self.date)
                    os.makedirs(self.result_dir, exist_ok=True)
                    
                    result_name = str(self.hour) + '.json'
                    result_file = os.path.join(self.result_dir, result_name)
                    result_name_tmp = str(self.hour) + '.json.tmp'
                    result_file_tmp = os.path.join(self.result_dir, result_name_tmp)
                    
                    # make the results file
                    with open(result_file_tmp, 'w') as tmp_file:
                        json.dump(self.results, tmp_file, indent=4)
                    os.rename(result_file_tmp, result_file)
                    
                    # log
                    message = f'Created results for today [{self.date}]'
                    self.logger.add_info(REQUESTOR, RESULTS, message)
                    # log a unique challenge ID and hash
                    utc_datetime = get_utc_datetime()
                    date_hour_str = utc_datetime[:13]
                    self.results_id = date_hour_str
                    to_hash = self.results_id + str(self.results)
                    self.results_hash = hashlib.sha256(to_hash.encode()).hexdigest()
                    self.logger.add_info(REQUESTOR, ID, self.results_id)
                    self.logger.add_info(REQUESTOR, HASH, self.results_hash)
                else:
                    message = f'No results recorded!'
                    self.logger.add_info(REQUESTOR, CRITICAL, message)
        else:
            message = f'Registry is empty!'
            self.logger.add_info(REQUESTOR, CRITICAL, message)
        
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
        self.logger.save_episode()
        


if __name__ == "__main__":
    requestor = Requestor()
    requestor.start_request()
        