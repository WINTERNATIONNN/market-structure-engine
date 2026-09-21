"""Scoring 层 —— 把 Wyckoff 引擎产物打分成可排序的候选档案。

红线: 纯计算, 零 LLM; 输出非布尔 (composite/分项分数 + 方向 + tags)。
分层: scoring 依赖 wyckoff/core, 不得反向 (见 pyproject.toml importlinter)。
"""

from mse.scoring.candidate import CandidateProfile, EventRef, TransitionRef
from mse.scoring.params import ScoringParams
from mse.scoring.score import score_candidate

__all__ = [
    "ScoringParams",
    "CandidateProfile",
    "EventRef",
    "TransitionRef",
    "score_candidate",
]
