import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


RESULTS_DIR = './divide21x/results'
RESULT = 'result'


if __name__ == "__main__":
    # get all result files
    files = [file for file in os.listdir(RESULTS_DIR)]
    files.sort()
    
    llm_scores = {}
    # go through each file
    for file in files:
        data = None
        # get the data
        with open(file, "r") as f:
            data = json.load(f)
        
            for llm_alias, info in data.items():
                if RESULT in info:
                    if llm_alias not in llm_scores:
                        llm_scores[llm_alias] = []
                    llm_scores[llm_alias].append(info[RESULT])
    
    # plot
    for llm, scores in llm_scores.items():
        days = list(range(1, len(scores)+1))
        plt.plot(days, scores, marker='o', label=llm)
        plt.xticks(days)
    
    plt.title("LLM Scores Over the Days")
    plt.xlabel("Day Number")
    plt.ylabel("Score (%)")
    plt.legend(title="LLM Alias")
    plt.grid(True)
    plt.tight_layout()
    plt.show()