import logging
import json
from datetime import datetime
from pathlib import Path

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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------
# Setup
# -------------------------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")

LOG_PATH = Path(__file__).parent.parent / "wellness_log.json"


def load_wellness_log() -> list[dict]:
    """Load history from JSON."""
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read wellness log: {e}")
        return []


def summarize_last_entry(entries: list[dict]) -> str:
    """Summarize the latest check-in."""
    if not entries:
        return "This is your very first check-in with me."

    last = entries[-1]
    mood = last.get("mood", "unspecified mood")
    energy = last.get("energy", "unspecified energy")
    goals = ", ".join(last.get("goals", [])) or "no specific goals"

    return (
        f"Last time, you said your mood was '{mood}', "
        f"your energy was '{energy}', and your goals were: {goals}."
    )


# -------------------------------------------------
# Assistant Class
# -------------------------------------------------
class Assistant(Agent):
    def __init__(self) -> None:
        # Load past wellness data
        self.log_entries = load_wellness_log()

        last_summary = summarize_last_entry(self.log_entries)

        # Extract last values (for better referencing)
        if self.log_entries:
            last = self.log_entries[-1]
            last_mood = last.get("mood", "unspecified")
            last_energy = last.get("energy", "unspecified")
            last_goals = ", ".join(last.get("goals", [])) if last.get("goals") else "no goals"
        else:
            last_mood = None
            last_energy = None
            last_goals = None

        instructions = f"""
You are a supportive, realistic health & wellness voice companion.
You help the user reflect on their day, mood, energy, and simple objectives.

SAFETY:
- You are NOT a doctor or therapist.
- Do NOT diagnose or give medical instructions.
- If the user expresses crisis-level distress, encourage them to reach out to a trusted contact or local professional.

PAST SESSION MEMORY:
Here is what you know from previous check-ins:
{last_summary}

Last mood: {last_mood}
Last energy: {last_energy}
Last goals: {last_goals}

Use this info naturally.  
Example:
- “Last time you felt {last_mood}. How does today compare?”
- “Previously you mentioned goals like {last_goals}. Want to set something similar today?”

TODAY’S CHECK-IN FLOW:
1. Warm greeting.
2. Ask about:
   - today's emotional mood,
   - today's energy,
   - any stress,
3. Ask for 1–3 simple intentions / goals for today.
4. Offer simple, practical, NON-MEDICAL advice.
5. Give a short recap.
6. CONFIRM with: “Does this sound right?”
7. ALWAYS call save_checkin at the end, even if the user does not confirm.

   With:
   - mood
   - energy
   - goals
   - a short 1-2 sentence summary

STYLE:
- Warm, concise, calm.
- One question at a time.
- Supportive and grounded.
"""

        super().__init__(instructions=instructions)

    # ---------- TOOL: save daily check-in ----------
    @function_tool
    async def save_checkin(
        self,
        context: RunContext,
        mood: str,
        energy: str,
        goals: list[str],
        summary: str,
    ):
        """
        Save the daily wellness check-in to wellness_log.json.
        """

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mood": mood,
            "energy": energy,
            "goals": goals,
            "summary": summary,
        }

        # Load, append, save
        log = load_wellness_log()
        log.append(entry)

        with LOG_PATH.open("w") as f:
            json.dump(log, f, indent=2)

        self.log_entries = log

        return "Daily wellness check-in saved."



# -------------------------------------------------
# Prewarm
# -------------------------------------------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _m(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    # -------------------------------------------------
    # Instantiate Assistant & Attach Transcript Listener
    # -------------------------------------------------

    assistant = Assistant()

    @session.on("transcription")
    def on_user_text(ev):
        text = ev.text.lower()

        # detect mood
        if assistant.mood is None and (
            "feel" in text or "mood" in text or "emotion" in text
        ):
            assistant.mood = ev.text

        # detect energy
        if assistant.energy is None and (
            "energy" in text or "tired" in text or "low" in text or "high" in text
        ):
            assistant.energy = ev.text

        # detect goals
        if "goal" in text or "want to" in text or "plan to" in text:
            items = [g.strip() for g in ev.text.replace("and", ",").split(",")]
            assistant.goals = [i for i in items if len(i) > 2]

        # if all info is collected → save
        if assistant.is_ready_to_save():
            summary = (
                f"Today you said your mood is '{assistant.mood}', "
                f"your energy is '{assistant.energy}', and your goals are {assistant.goals}."
            )

            session.invoke_tool(
                "save_checkin",
                mood=assistant.mood,
                energy=assistant.energy,
                goals=assistant.goals,
                summary=summary,
            )

    # Start agent session
    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
