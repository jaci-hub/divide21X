import hashlib
import json
import os
import re
from divide21x.llm_api.client_class import ModelClient
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_llm_registry, get_utc_date, get_utc_datetime, get_utc_day, get_utc_hour


RESULTS_DIR = './divide21x/results'
BASE_DIR = './divide21x/llm_api/logs'
# categories
REQUESTOR = 'requestor'
API = 'api'
CHAT = 'chat'
NOTE = 'note'
PROMPT = 'prompt'
ANSWER = 'answer'
RESULTS = 'results'
ID = 'id'
HASH = 'hash'
# types
CRITICAL = 'critical'
WARNING = 'warning'

Z = "z"
A = "a"
O = "o"


class Requestor():
    def __init__(self):
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
        
        self.registry = None
        self.prompt = None
                
        # get date
        self.date = str(get_utc_date())
        self.day = str(get_utc_day())
        
        self.results = {}
        
    def get_prompt(self):
        try:
            with open(f"divide21x/challenges/{self.date}.json", "r") as f:
                challenge_data = json.load(f)
        except FileNotFoundError:
            challenge_data = None
            message = "Challenge has not been created yet!"
            self.logger.add_info(REQUESTOR, CRITICAL, message)
            return self.prompt

        # Construct the few-shot + challenge prompt
        prompt_lines = []
        for example_key in ["example_1", "example_2"]:
            ex = challenge_data[example_key]
            prompt_lines.append(f"Example:\n{Z}: {json.dumps(ex[Z])}\n"
                                f"{A}: {json.dumps(ex[A])}\n"
                                f"{O}: {json.dumps(ex[O])}\n")

        # Add the challenge (no final_state)
        challenge = challenge_data["challenge"]
        prompt_lines.append(f"Challenge:\n{Z}: {json.dumps(challenge[Z])}\n"
                            f"{A}: {json.dumps(challenge[A])}\n"
                            f"{O}: ? (compute this and return as JSON)")

        prompt_lines.append(f"Given '{Z}' and '{A}', compute '{O}'. You must ONLY return a valid JSON object.")
        
        self.prompt = "\n\n".join(prompt_lines)
        
        # log
        self.logger.add_info(REQUESTOR, PROMPT, self.prompt)
        
        return self.prompt
        

    def prompt_llm(self, registry_entry):
        client = ModelClient(
            registry_entry=registry_entry
        )
        if client.client is None:
            return

        # Request the LLM   
        answer = client.chat(prompt=self.prompt)
        
        # clean the answer - although it might be json, it still might need to be polished as it is gotten from a chat
        # --- Clean the answer safely ---
        if not answer or not isinstance(answer, str):
            self.logger.add_info(CHAT, "ERROR", f"Empty or invalid answer: {answer}")
            return {"error": "empty_answer"}

        # (1) Remove Markdown code fences, with optional language tag and newlines
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.DOTALL)

        # (2) Remove any leading non-JSON text before the first brace/bracket
        json_start = re.search(r"[\{\[]", answer)
        if json_start:
            answer = answer[json_start.start():].strip()

        # (3) Unescape backslashes (common in JSON returned as string)
        try:
            answer = answer.encode('utf-8').decode('unicode_escape')
        except Exception:
            pass  # only warn if necessary

        # (4) Try to parse JSON safely
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError as e:
            # fallback: try a cleanup for double quotes or trailing commas
            cleaned = answer.strip().strip('"').strip("'").rstrip(',')
            try:
                answer = json.loads(cleaned)
            except Exception:
                self.logger.add_info(CHAT, "WARN", f"Invalid JSON: {answer[:150]} | Error: {e}")
                answer = {"error": "invalid_json", "raw": answer}
        
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
            self.prompt = self.get_prompt()
            
            if self.prompt:
                # start the requests
                for registry_entry in self.registry:
                    self.prompt_llm(registry_entry)
            
                # write to results dir
                if self.results:
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    
                    result_name = str(self.date) + '.json'
                    result_file = os.path.join(RESULTS_DIR, result_name)
                    result_name_tmp = result_name + '.tmp'
                    result_file_tmp = os.path.join(RESULTS_DIR, result_name_tmp)
                    
                    # make the results file
                    with open(result_file_tmp, 'w') as tmp_file:
                        json.dump(self.results, tmp_file, indent=4)
                    os.rename(result_file_tmp, result_file)
                    
                    # log
                    message = f'Results for today [{self.date}] are in.'
                    self.logger.add_info(REQUESTOR, RESULTS, message)
                    # log a unique challenge ID and hash
                    self.results_id = self.date
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
        