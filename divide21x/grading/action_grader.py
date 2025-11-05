"""
Author: Jacinto Jeje Matamba Quimua
Date: 11/01/2025

Description:
------------
Evaluation module for Divide21X Phase 1: Action-only benchmark environment
for faithful strategic reasoning.

This module compares an LLM's submitted action and resulting game state
against the ground-truth Divide21 agent to produce a numerical fidelity score.

Scoring is based on:
    (1) Action Fidelity — how closely the LLM's chosen action matches the ground-truth action.
    (2) State Fidelity  — how similar the resulting next-state is to the ground-truth next-state.

Both are combined to form an overall Divide21X Phase 1 score.
"""
import divide21env
from divide21env.envs.divide21_env import Divide21Env
from divide21x.inspection.inspector import Inspector
import numpy as np
import math
from typing import Dict, Any, Tuple


class ActionGrader():
    """
    Evaluates action-only submissions (no explanations) from LLMs against
    the Divide21 ground-truth agent.

    Usage:
    -------
    >>> actionGrader = ActionGrader()
    >>> result = actionGrader.evaluate_submission(llm_action, ground_truth_action, llm_state, gt_state)
    >>> print(result)
    {'action_fidelity': 0.8, 'state_fidelity': 0.9, 'overall_score': 0.85}
    """

    def __init__(self, digits: int = 3, players: list = None):
        self.digits = digits
        self.players = players or [{"id": 0, "score": 0, "is_current_turn": 1}]
        self.gt_env = Divide21Env(digits=self.digits, players=self.players)

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the action dict is in canonical form."""
        return {
            "division": bool(action.get("division", 0)),
            "digit": int(action.get("digit", 0)),
            "rindex": int(action.get("rindex", 0)),
        }

    def _action_distance(self, a1: Dict[str, Any], a2: Dict[str, Any]) -> float:
        """
        Compute normalized distance between two actions.
        Returns value in [0,1], where 1.0 = identical, 0.0 = completely different.
        """
        keys = ["division", "digit", "rindex"]
        matches = sum(1 for k in keys if a1.get(k) == a2.get(k))
        return matches / len(keys)

    def _state_similarity(self, s1: Dict[str, Any], s2: Dict[str, Any]) -> float:
        """
        Computes state similarity between two Divide21 states.
        Each state is expected to contain:
            - 'static_number'
            - 'dynamic_number'
            - 'available_digits_per_rindex'
            - 'players'
            - 'player_turn'
        """
        score = 0
        total = 0

        # Compare static_number
        n1 = "".join(map(str, s1.get("static_number", [])))
        n2 = "".join(map(str, s2.get("static_number", [])))
        total += 1
        score += 1 if n1 == n2 else 1 - min(1.0, abs(int(n1) - int(n2)) / 999)
        
        # Compare dynamic_number
        n1 = "".join(map(str, s1.get("dynamic_number", [])))
        n2 = "".join(map(str, s2.get("dynamic_number", [])))
        total += 1
        score += 1 if n1 == n2 else 1 - min(1.0, abs(int(n1) - int(n2)) / 999)

        # Compare player_turn
        total += 1
        score += 1 if s1.get("player_turn") == s2.get("player_turn") else 0

        # Compare players' scores (normalized)
        total += 1
        p1_scores = np.array(s1.get("players", []))[1::3] if len(s1.get("players", [])) > 0 else np.zeros(1)
        p2_scores = np.array(s2.get("players", []))[1::3] if len(s2.get("players", [])) > 0 else np.zeros(1)
        diff = np.abs(p1_scores - p2_scores).mean() if len(p1_scores) == len(p2_scores) else 1
        score += 1 - min(1.0, diff / 10)

        # Compare available digits mask (Jaccard similarity)
        total += 1
        d1 = set(np.where(np.array(s1.get("available_digits_per_rindex", [])) == 1)[0])
        d2 = set(np.where(np.array(s2.get("available_digits_per_rindex", [])) == 1)[0])
        intersection = len(d1.intersection(d2))
        union = len(d1.union(d2)) or 1
        score += intersection / union

        return score / total

    # ------------------------------------------------------------------
    # Main evaluation function
    # ------------------------------------------------------------------
    def evaluate_submission(
        self,
        llm_action: Dict[str, Any],
        gt_action: Dict[str, Any],
        llm_state: Dict[str, Any],
        gt_state: Dict[str, Any],
        alpha: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate one LLM submission (action + state) against ground truth.

        Parameters
        ----------
        llm_action : dict
            Model's submitted action (division, digit, rindex)
        gt_action : dict
            Ground-truth agent's action
        llm_state : dict
            Resulting state after model's action
        gt_state : dict
            Resulting state after ground-truth action
        alpha : float
            Weighting factor between action and state fidelity (default = 0.5)

        Returns
        -------
        dict
            {
                "action_fidelity": float,
                "state_fidelity": float,
                "overall_score": float
            }
        """
        llm_action = self._normalize_action(llm_action)
        gt_action = self._normalize_action(gt_action)

        # Compute fidelities
        action_fidelity = self._action_distance(llm_action, gt_action)
        state_fidelity = self._state_similarity(llm_state, gt_state)

        # Combine scores
        overall_score = alpha * action_fidelity + (1 - alpha) * state_fidelity

        return {
            "action_fidelity": round(float(action_fidelity), 4),
            "state_fidelity": round(float(state_fidelity), 4),
            "overall_score": round(float(overall_score), 4),
        }
