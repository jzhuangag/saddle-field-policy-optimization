"""Self-contained game and neural-policy definitions used by the journal paper.

Only the four neural Markov games reported in the manuscript are exposed here.
The tabular suite additionally constructs the one-state RPS benchmark directly
in :mod:`markov_game_suite`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


ACTIONS = 3
INPUT_DIM = 4
HIDDEN = 8


@dataclass(frozen=True)
class Game:
    """Finite zero-sum Markov game used by the population oracle."""

    name: str
    features: torch.Tensor
    rewards: torch.Tensor
    transitions: torch.Tensor
    rho: torch.Tensor


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    return array / array.sum(axis=-1, keepdims=True)


def _features(states: int, third: np.ndarray | None = None) -> np.ndarray:
    phase = 2.0 * np.pi * np.arange(states) / states
    if third is None:
        third = np.arange(states) / max(states - 1, 1)
    return np.stack(
        (np.sin(phase), np.cos(phase), third, np.ones(states)), axis=1
    )


def _game(
    name: str,
    rewards: np.ndarray,
    transitions: np.ndarray,
    features: np.ndarray,
) -> Game:
    states = len(features)
    return Game(
        name,
        torch.tensor(features),
        torch.tensor(rewards),
        torch.tensor(_normalize_rows(transitions)),
        torch.ones(states) / states,
    )


def cyclic_control() -> Game:
    """Five-state cyclic-control game from Appendix B of the manuscript."""

    states = 5
    phase = 2.0 * np.pi * np.arange(states) / states
    rps = np.array(
        [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
    )
    features = _features(states)
    rewards = np.empty((states, ACTIONS, ACTIONS))
    transitions = np.full(
        (states, ACTIONS, ACTIONS, states), 0.02 / (states - 1)
    )
    for state in range(states):
        permutation = np.roll(np.arange(ACTIONS), state % ACTIONS)
        rewards[state] = (
            (1.0 + 0.12 * np.cos(phase[state]))
            * rps[np.ix_(permutation, permutation)]
        )
        for action_max in range(ACTIONS):
            for action_min in range(ACTIONS):
                next_state = (
                    state + 1 + ((action_max - action_min) % ACTIONS)
                ) % states
                transitions[state, action_max, action_min, next_state] = 0.98
    return _game("CyclicControl", rewards, transitions, features)


def frequency_hopping() -> Game:
    """Eight-state anti-jamming frequency-hopping game."""

    states = 8
    phase = 2.0 * np.pi * np.arange(states) / states
    quality = np.stack(
        [
            0.25 * np.cos(phase + 2.0 * np.pi * action / ACTIONS)
            for action in range(ACTIONS)
        ],
        axis=1,
    )
    rewards = np.empty((states, ACTIONS, ACTIONS))
    transitions = np.full(
        (states, ACTIONS, ACTIONS, states), 0.08 / (states - 1)
    )
    for state in range(states):
        for action_max in range(ACTIONS):
            for action_min in range(ACTIONS):
                rewards[state, action_max, action_min] = quality[
                    state, action_max
                ] + (0.55 if action_max != action_min else -1.10)
                next_state = (
                    state + 1 + action_max - action_min
                ) % states
                transitions[state, action_max, action_min, next_state] = 0.92
    return _game(
        "FrequencyHopping",
        rewards,
        transitions,
        _features(states, quality.max(1) - quality.min(1)),
    )


def routing_interdiction() -> Game:
    """Six-state routing-versus-interdiction game."""

    states = 6
    phase = 2.0 * np.pi * np.arange(states) / states
    capacities = np.stack(
        [
            1.0 + 0.20 * np.sin(phase + 2.0 * np.pi * action / ACTIONS)
            for action in range(ACTIONS)
        ],
        axis=1,
    )
    rewards = np.empty((states, ACTIONS, ACTIONS))
    transitions = np.full(
        (states, ACTIONS, ACTIONS, states), 0.10 / (states - 1)
    )
    for state in range(states):
        for action_max in range(ACTIONS):
            for action_min in range(ACTIONS):
                rewards[state, action_max, action_min] = (
                    capacities[state, action_max]
                    - (1.35 if action_max == action_min else 0.15)
                    - 0.55
                )
                next_state = (
                    state + 1 + int(action_max == action_min) + action_max
                ) % states
                transitions[state, action_max, action_min, next_state] = 0.90
    return _game(
        "RoutingInterdiction",
        rewards,
        transitions,
        _features(states, capacities.std(1)),
    )


def security_patrol() -> Game:
    """Eight-state patrol-versus-attack game."""

    states = 8
    values = 0.8 + 0.25 * np.cos(
        2.0 * np.pi * np.arange(states)[:, None] / states
        + 2.0 * np.pi * np.arange(ACTIONS)[None, :] / ACTIONS
    )
    rewards = np.empty((states, ACTIONS, ACTIONS))
    transitions = np.full(
        (states, ACTIONS, ACTIONS, states), 0.08 / (states - 1)
    )
    for state in range(states):
        for action_max in range(ACTIONS):
            for action_min in range(ACTIONS):
                rewards[state, action_max, action_min] = values[
                    state, action_min
                ] * (1.0 if action_max == action_min else -0.55)
                next_state = (
                    state + 1 + action_min + int(action_max != action_min)
                ) % states
                transitions[state, action_max, action_min, next_state] = 0.92
    return _game(
        "SecurityPatrol",
        rewards,
        transitions,
        _features(states, values.mean(1)),
    )


def journal_games() -> tuple[Game, ...]:
    """Return the four neural-policy games reported in the journal paper."""

    return (
        cyclic_control(),
        frequency_hopping(),
        routing_interdiction(),
        security_patrol(),
    )


def policy_dim() -> int:
    """Number of parameters in one 4-8-3 tanh-softmax policy."""

    return INPUT_DIM * HIDDEN + HIDDEN + HIDDEN * ACTIONS + ACTIONS


def policy(block: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """Evaluate a state-conditioned 4-8-3 tanh-softmax policy."""

    cursor = 0
    w1 = block[cursor : cursor + INPUT_DIM * HIDDEN].reshape(INPUT_DIM, HIDDEN)
    cursor += INPUT_DIM * HIDDEN
    b1 = block[cursor : cursor + HIDDEN]
    cursor += HIDDEN
    w2 = block[cursor : cursor + HIDDEN * ACTIONS].reshape(HIDDEN, ACTIONS)
    cursor += HIDDEN * ACTIONS
    b2 = block[cursor : cursor + ACTIONS]
    logits = torch.tanh(features @ w1 + b1) @ w2 + b2
    logits = logits - logits.mean(dim=1, keepdim=True)
    return torch.softmax(logits, dim=1)
