# ============================================================
# config.py — Full configuration with detailed English comments
# ============================================================

from pathlib import Path  # Used to build filesystem paths
import os                # Used to access environment variables (fallbacks)

# ====================== BASE PATHS / HTTP ======================

ROOT_DIR = Path(__file__).resolve().parent        # Root folder of the project
TEMP_AUDIO_DIR = ROOT_DIR / "temp_audio"          # Temporary folder for saving audio files
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True) # Create temp_audio folder if not present

LOCAL_HTTP_BASE = "http://127.0.0.1:8008"  # Base URL for local HTTP requests

# ============================ FURHAT ===========================

FURHAT_HOST = "localhost"  # Furhat robot host address
FURHAT_PORT = 54321        # Furhat robot port

# ============================== ASR ============================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
# Read from the environment. The previous hard-coded key was exposed and must be
# revoked. Set it in PowerShell before running (see user_study/PROTOCOL.md).
# AssemblyAI API key used for speech recognition

SAMPLE_RATE = 16000         # Audio sample rate in Hz
ASR_DEVICE_INDEX = 1        # Microphone device index (choose correct input device)
ASR_BLOCKSIZE = 800         # Number of audio samples per processing block
ASR_PARTIAL_DEBOUNCE_MS = 80  # Debounce time for partial ASR results (to avoid flicker)

PREP_REFRESH_ON_CANCEL = True   # Whether to refresh PREP when canceling due to partial mismatch

# ============================== VAP ============================



VAP_TORCH_PT_PATH = (
    ROOT_DIR
    / "VoiceActivityProjection"
    / "example"
    / "VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
)

VAP_TORCH_CLASS = "vap.model:VapGPT"

VAP_TORCH_HALF = False
VAP_TORCH_NUM_THREADS = 1

VAP_ONNX = ROOT_DIR / "onnx" / "vap.onnx"

VAP_ONNX_PROVIDERS = ["CPUExecutionProvider"]
VAP_ONNX_INTRA = 4
VAP_ONNX_INTER = 1
# --- Misc
VAP_WARMUP_S = 4.0
VAP_IDLE_WIN = 0.18
VAP_SPEAK_WIN = 0.22

# =============================== TTS ===========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Read from the environment. The previous hard-coded key was exposed and must be
# revoked. Condition B and OpenAI TTS both depend on this.
# OpenAI API key for TTS and LLM

OPENAI_TTS_MODEL = "gpt-4o-mini-tts"   # OpenAI TTS model
OPENAI_TTS_VOICE = "alloy"             # Voice type for TTS

TTS_FRAME_MS = 20                      # Frame size in milliseconds for TTS streaming
TTS_RING_CAP_SEC = 10.0                # Maximum audio buffer capacity in seconds
TTS_STREAM_TIMEOUT_S = 12.0            # Timeout for TTS stream in seconds

TTS_FIXED_BASENAME = "robot_turn"      # Fixed filename base for TTS outputs

# =============================== LLM ===========================

OPENAI_MODEL = "gpt-4o-mini"   # OpenAI LLM model used for generating responses
OPENAI_MAX_TOKENS = 350        # Max tokens per LLM response
OPENAI_TEMPERATURE = 0.5       # Sampling temperature for LLM (higher = more random)
OPENAI_TOP_P = None            # Nucleus sampling parameter (None = disabled)
OPENAI_REQUEST_TIMEOUT = 12.0  # Timeout for LLM requests in seconds.
                               # MUST stay below PREP_STALL_MS (see below), otherwise the
                               # controller abandons a request that is still alive and
                               # stacks overlapping OpenAI streams.
OPENAI_MAX_RETRIES = 2         # Maximum retries if LLM request fails
LLM_PROMPT_BUDGET_TOKENS = 8000 # Max tokens reserved for conversation context   

SEPARATOR = "|||"             # Special delimiter used to split bot turns.
                              # Matches the multi-agent SEP so both study
                              # conditions use the same separator on gemma3:1b.
CURRENT_TIME = 0 
# ============================== Job Data ========================


job_description = """\
Junior Data Scientist role at DataTech Berlin. Support the data team in building, evaluating and deploying data-driven solutions.
Responsibilities: collect, clean and wrangle data; build and evaluate machine learning models; create visualizations and communicate insights to stakeholders; contribute to real-world data science projects; collaborate within a cross-functional team.
Knowledge of Python, SQL, common ML libraries (scikit-learn, pandas, NumPy) and model evaluation metrics is valued.
Qualifications: Bachelor/Master in Data Science, Computer Science, Statistics or related; strong analytical and communication skills; fluent English.
"""

company_information = """\
DataTech Berlin is a data and analytics company based in Berlin, Germany.
We focus on building data-driven products and helping clients turn data into actionable insights.
Our mission is to combine rigorous data science, teamwork, and clear communication to deliver real-world impact.
For more information, please contact us at info@datatech-berlin.de
or visit www.datatech-berlin.de.
"""

gpt_system_prompt = f"""
# Job Interview Simulation — No Repeats & Natural Turn-taking

## Your Role
You are the hiring manager at DataTech Berlin, interviewing for a Junior Data Scientist role.

## Hard Rules
- Ask exactly ONE question per turn.
- Each turn = max 1 sentences (6–9 words each).
- Separate each sentence with this token: {SEPARATOR}.
- Do NOT invent new topics beyond the list below but ask deep questions about the topics below.
- If there are no remaining topics, end the interview naturally.
- Never answers to unrelated questions except company information and data science.
- If the candidate cannot answer, move directly to the next unasked topic.
- The Interview must be 10 minutes.
- Conduct the interview naturally and professionally.
- Whenever you do not understand the Candidate quesries ask for repeatation or confirmation.
- Focus on relevant data science topics.
- If the user interrupts while you are speaking:
  1) Stop immediately (do NOT finish previous sentence).
  2) Respond to the interruption in ONE short sentence (6–9 words).
  3) After your response ask if user is ready for continue. Output format:<brief answer>{SEPARATOR}<shall we continue?>
  4) Filler is optional, but total must remain max two short sentences.

## Topic Control (no repeat logic)
Topics (ask each at most 1):
education and academic background; programming skills and tools; machine learning and modelling experience;
data wrangling and preprocessing; model evaluation and metrics; data visualization and storytelling;
real-world data science projects; teamwork and communication; problem solving approach; career goals in data science;


At every turn, follow this deterministic logic:
1) Read the full chat history.
2) Identify which topics from the list have ALREADY been asked (semantic match allowed).
3) Let Asked = those topics.
4) Let Next = the first topic from the list that is NOT in Asked.
5) If Next exists → ask EXACTLY ONE question about **Next**.


## Speaking Style
- Begin each turn (if not interrupted) with a short natural filler such as:
  "Alright,", "I see,", "Sounds good,", "Okay,", "Interesting,", "Thanks for sharing,", "That’s great,".
- Then ask the question.
- Format example:
  Okay,{SEPARATOR}Which programming languages do you use most for data science?


## Interview Context
### Job Description
{job_description}

### Company Information
{company_information}
""".strip()





# ============================== TurnGPT =========================

TURNGPT_DIR = str((ROOT_DIR / "Trained_TurnGPT").resolve()) # Path to TurnGPT trained model
TURNGPT_MAX_CTX_TOKENS = 256   # Max context tokens for TurnGPT
TURNGPT_DEBUG = False          # Enable TurnGPT debug logging

TURNGPT_PARTIAL_CAP_YIELD = 0.40    # Max probability for partial yield during ASR
TURNGPT_PARTIAL_CAP_PREPARE = 0.58  # Max probability for partial prepare

TURNGPT_PREPARE_THRESHOLD = 0.20    # Probability threshold for TurnGPT prepare

TURNGPT_YIELD_TIMEOUTS = [          # Yield timeouts based on probability bins
    (0.50, 200),    # If prob ≥ 0.50 → wait 250ms
    (0.30, 400),    # If prob ≥ 0.30 → wait 500ms
    (0.20, 800),   # If prob ≥ 0.20 → wait 1s
    (0.10, 1600),   # If prob ≥ 0.10 → wait 2s
    (0.00, 2500),   # Otherwise wait max 3s
]

# ============================== Similarity ======================

SIM_THREADS = 4                 # Number of threads for similarity model
SIM_MAX_LENGTH = 128            # Max token length for similarity input
CANCEL_ON_SIMILARITY = True     # Cancel if similarity is too high
SIMILARITY_PREPARE_THRESHOLD = 0.80  # Threshold for preparing based on similarity
SIMILARITY_CANCEL_THRESHOLD  = 0.75
# ======================= Controller Policy ======================

VAP_PNOW_YIELD_THRESHOLD = 0.48   # Threshold for robot-likely current prob
VAP_PFUT_YIELD_THRESHOLD = 0.48   # Threshold for robot-likely future prob
USER_LOW_NOW_FOR_YIELD = 0.47     # Threshold for user-unlikely current prob
USER_LOW_FUT_FOR_YIELD = 0.46     # Threshold for user-unlikely future prob
MIN_GAP_TIME = 500                # Minimum silence gap before yielding
YIELD_COOLDOWN_MS = 400           # Cooldown after yielding before next action

BARGE_IN_GRACE_MS = 350           # Grace period to avoid false barge-ins
BARGE_IN_HOLD_MS = 300            # Minimum hold duration before confirming barge-in
VAP_USER_PNOW_INTERRUPT = 0.48    # Threshold for user barge-in current prob
VAP_USER_PFUT_INTERRUPT = 0.48    # Threshold for user barge-in future prob
ENABLE_BARGE_IN = False

BACKCHANNEL_DEADZONE_MS = 800     # Deadzone where backchannels cannot happen
BACK_HOLD_MS = 200                # Duration to hold backchannel state
BACK_PNOW_MIN = 0.33              # Minimum user probability for backchannel
BACK_PNOW_MAX = 0.65              # Maximum user probability for backchannel
BACK_PFUT_MAX = 0.35              # Maximum user future probability for backchannel

# ======================= Controller Helpers =====================

AUDIO_CHUNK_MS = 30            # Audio chunk size in ms
ASR_PREPARE_TIMEOUT_MS = 200   # Timeout for ASR prepare stage
PREP_STALL_MS = 15000          # Backstop only. MUST stay above OPENAI_REQUEST_TIMEOUT
                               # (12 s) so a live request is never abandoned as
                               # "stalled". Real failures are caught far sooner by
                               # PreparedReply.failed(), not by this timer.
PREP_DRAIN_MS = 90            # Time before draining PREP state

# ---------------- Legacy participant-facing controller (main.py) ----------------

ENABLE_EARLY_LLM_PREFETCH = False
# Speculative LLM generation from evolving ASR partials.
# Disabled for the user study for two reasons:
#   1) Stability — long answers produced many partials, and each one could cancel
#      and restart the in-flight OpenAI stream, so no reply ever completed.
#   2) Comparability — the SLM arm (rq3b_user_eval_main.py) has
#      partial_asr_llm_prefetch = False in its frozen design, so both study arms
#      now begin generating only after a settled final transcript.
# NOTE: this flag is read by main.py's EARLY PREPARE block. Setting it here alone
# does nothing unless that guard is present.

RQ3B_USER_EVAL_MIN_GAP_MS = 1000
# Minimum silence hold for the SLM arm's participant-facing controller
# (rq3b_user_eval_policy.py). Previously relied on an implicit getattr default,
# which made it invisible to the freeze manifest. Stated explicitly so the value
# governing Condition A's turn-taking is recorded and verifiable.
# Matches LEGACY_MIN_HOLD_MS below so both study arms wait equally long.
# NOT read by the frozen RQ3-B experiment files.

LEGACY_MIN_HOLD_MS = 1000
# Minimum silence the legacy controller must observe before taking the floor.
# TurnGPT's shaped probability sits at ~0.89, which always selects the top rung
# of TURNGPT_YIELD_TIMEOUTS (200 ms), so without this floor the robot cut in on
# ordinary thinking pauses. Set to match RQ3B_USER_EVAL_MIN_GAP_MS (1000 ms) in
# the SLM arm, so both study conditions wait equally long before responding --
# otherwise questionnaire items Q1/Q2/Q5 would measure a threshold difference
# rather than a difference between the systems.
# READ ONLY BY main.py. Not used by the multi-agent stack or any frozen module.

LLM_FALLBACK_TEXT = "Sorry, I did not catch that. Could you say it again?"
# Spoken when a prepared reply finishes with no usable text, so the participant
# never faces open-ended silence after a backend failure.

# ======================= Conversation Prompts ==================

INTERVIEW_GREETING = "Hi, I am Furhat, your interviewer at DataTech Berlin. We are interviewing for a Junior Data Scientist position. Are you ready to begin?"   # Initial greeting (matches multi_agent_main.GREETING)
INACTIVITY_REPROMPT_TEXT = "are you there ? shall we continue?" # Prompt if silence too long
INACTIVITY_REPROMPT_MS = 30000   # Time in ms before reprompt

