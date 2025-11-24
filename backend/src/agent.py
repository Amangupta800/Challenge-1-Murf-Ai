import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ----------------------------------------------------
# Setup & shared content
# ----------------------------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")

CONTENT_PATH = Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json"


def _default_tutor_content() -> list[dict]:
    """Fallback content if JSON file is missing or invalid."""
    return [
        {
            "id": "variables",
            "title": "Variables",
            "summary": "Variables store values so you can reuse them later. Think of them as labeled boxes that hold data like numbers or text.",
            "sample_question": "What is a variable and why is it useful?",
        },
        {
            "id": "loops",
            "title": "Loops",
            "summary": "Loops let you repeat an action multiple times without copying code, like running something for each item in a list.",
            "sample_question": "Explain the difference between a for loop and a while loop.",
        },
    ]


def load_tutor_content() -> list[dict]:
    """Load course concepts from JSON, with safe fallback."""
    if not CONTENT_PATH.exists():
        logger.warning("Tutor content file not found at %s, using defaults", CONTENT_PATH)
        return _default_tutor_content()

    try:
        with CONTENT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Content JSON must be a list of concepts")
        return data
    except Exception as e:
        logger.warning("Failed to read tutor content JSON (%s), using defaults", e)
        return _default_tutor_content()


def make_tts_for_mode(mode: str) -> murf.TTS:
    """Return the correct Murf Falcon voice for a given mode."""
    mode = mode.lower().strip()
    voice_id = "Matthew"

    if mode == "quiz":
        voice_id = "Alicia"
    elif mode == "teach_back":
        voice_id = "Ken"
    else:
        voice_id = "Matthew"  # learn / default

    return murf.TTS(
        voice=voice_id,
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )


# ----------------------------------------------------
# Single TutorAgent class with 3 modes + handoffs
# ----------------------------------------------------


class TutorAgent(Agent):
    """
    Day 4 – Teach-the-Tutor: Active Recall Coach

    Modes:
      - learn      → explains a concept (Murf Falcon: Matthew)
      - quiz       → asks questions (Murf Falcon: Alicia)
      - teach_back → user explains; agent gives feedback (Murf Falcon: Ken)

    This class is instantiated with a specific (mode, concept_id).
    Switching modes returns a NEW instance via a tool → triggers agent handoff.
    """

    def __init__(
        self,
        mode: str = "intro",
        concept_id: Optional[str] = None,
    ) -> None:
        self.mode = mode.strip().lower()
        concepts = load_tutor_content()
        self.concepts_by_id = {c["id"]: c for c in concepts}

        # Choose default concept if none given
        if concept_id is None and concepts:
            concept_id = concepts[0]["id"]
        self.concept_id = concept_id

        concept_list_text = "\n".join(
            f"- id: {c['id']} | title: {c['title']}"
            for c in concepts
        ) or "None (no concepts defined)."

        # Safe lookup for current concept
        concept = self.concepts_by_id.get(self.concept_id) if self.concept_id else None
        concept_title = concept["title"] if concept else "Unknown concept"
        concept_summary = concept["summary"] if concept else ""
        concept_question = concept.get("sample_question", "") if concept else ""

        # Different behavior hints per mode
        if self.mode == "learn":
            persona = "You are Matthew, a calm explainer."
            mode_instructions = """
LEARN MODE:
- Explain the current concept clearly using the summary from content.
- Break explanation into 2–3 short chunks.
- After each chunk, ask a quick check like: "Does that make sense?" or "Should I give an example?".
- Keep it concise and interactive.
"""
        elif self.mode == "quiz":
            persona = "You are Alicia, a friendly quiz master."
            mode_instructions = """
QUIZ MODE:
- Ask ONE question at a time about the current concept.
- Use the sample_question as a starting point, then vary it.
- Wait for the user's answer, then briefly evaluate:
  - What they got right.
  - What is missing or slightly off.
- Stay encouraging; you can ask simple follow-ups.
"""
        elif self.mode == "teach_back":
            persona = "You are Ken, a thoughtful coach."
            mode_instructions = """
TEACH_BACK MODE:
- Ask the user to explain the current concept in their own words.
- Let them talk and finish.
- Then give QUALITATIVE FEEDBACK:
  - What they explained well.
  - 1–2 gaps or suggestions for improvement.
- Give a simple mastery score 1–5.
- Encourage them that teaching back is part of learning.
"""
        else:
            # Intro / neutral mode
            persona = "You are a neutral tutor orchestrator."
            mode_instructions = """
INTRO MODE:
- Greet the user briefly.
- Explain that you are an active recall tutor with three modes:
  - learn (Matthew) – explanation
  - quiz (Alicia) – questions
  - teach_back (Ken) – they explain, you score
- Show the user the available concept ids and titles.
- Ask them TWO things:
  1) Which concept id they want to study.
  2) Which mode they want: learn, quiz, or teach_back.
- Then call the `switch_mode` tool with the chosen mode + concept_id.
"""

        instructions = f"""
You are an ACTIVE RECALL PROGRAMMING TUTOR.

Personality:
{persona}

AVAILABLE CONCEPTS (from JSON):
{concept_list_text}

Current concept:
- id: {self.concept_id}
- title: {concept_title}
- summary: {concept_summary}
- sample_question: {concept_question}

CURRENT MODE: {self.mode.upper()}

{mode_instructions}

GENERAL RULES:
- Ask one clear question at a time.
- Keep turns short and interactive.
- Use concrete examples when helpful.
- You can always ask which mode or concept the user wants next.
- To change mode or concept, use the `switch_mode` tool.

TOOLS:
- `switch_mode(mode, concept_id)`:
   Use this whenever the user says things like:
   - "switch to quiz"
   - "now teach back loops"
   - "explain variables again"
   It will create a new specialized tutor instance with the right voice.
"""

        # Choose TTS per mode (Murf Falcon voices)
        # Intro uses Matthew by default
        tts_plugin = make_tts_for_mode(self.mode if self.mode != "intro" else "learn")

        super().__init__(
            instructions=instructions,
            tts=tts_plugin,
        )

    async def on_enter(self) -> None:
        """
        Called when this TutorAgent instance becomes active (after handoff).
        We give a short mode-specific greeting prompt.
        """
        if self.mode == "intro":
            prompt = (
                "Greet the user, explain the three modes and available concepts, "
                "then ask which concept and mode they want to start with."
            )
        elif self.mode == "learn":
            prompt = (
                "Briefly greet the user as Matthew and start explaining the current concept in one or two sentences, "
"then ask if they'd like more detail or an example."

            )
        elif self.mode == "quiz":
            prompt = (
                "Introduce yourself as Alicia and ask one quiz question about the current concept."
            )
        elif self.mode == "teach_back":
            prompt = (
                "Introduce yourself as Ken and ask the user to explain the current concept "
                "in their own words."
            )
        else:
            prompt = "Greet the user and ask how they want to study."

        await self.session.generate_reply(instructions=prompt)

    # ---------------- Tool: switch_mode → handoff ----------------

    @function_tool
    async def switch_mode(
        self,
        context: RunContext,
        mode: str,
        concept_id: Optional[str] = None,
    ):
        """
        Switch to a different learning mode (and optionally concept) by returning
        a NEW TutorAgent instance → triggers an agent handoff.

        Args:
            mode: "learn", "quiz", or "teach_back".
            concept_id: Optional concept id. If omitted, keeps current concept.
        """
        mode = mode.strip().lower()
        if mode not in {"learn", "quiz", "teach_back"}:
            raise ValueError("mode must be one of: learn, quiz, teach_back")

        if concept_id is None:
            concept_id = self.concept_id

        if concept_id not in self.concepts_by_id:
            raise ValueError(
                f"Unknown concept_id '{concept_id}'. "
                f"Valid ids: {', '.join(self.concepts_by_id.keys())}"
            )

        logger.info("Handoff: switching to mode=%s, concept=%s", mode, concept_id)

        # IMPORTANT: returning a new Agent → AGENT HANDOFF (per docs)
        # We are NOT passing chat_ctx here to avoid the session access error.
        return TutorAgent(
            mode=mode,
            concept_id=concept_id,
        )


# ----------------------------------------------------
# LiveKit session wiring
# ----------------------------------------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        # STT
        stt=deepgram.STT(model="nova-3"),
        # LLM
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        # Default TTS (will be overridden by TutorAgent’s own tts)
        tts=make_tts_for_mode("learn"),
        # Turn detection / VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start in INTRO mode so it asks for mode + concept first
    await session.start(
        agent=TutorAgent(mode="intro", concept_id=None),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
