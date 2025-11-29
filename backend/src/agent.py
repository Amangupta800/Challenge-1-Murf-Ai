import logging
from dotenv import load_dotenv
from datetime import datetime

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
)
from livekit.agents import RunContext
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ----------------------------------------------------
# Game Master Agent (Day 8)
# ----------------------------------------------------


class GameMasterAgent(Agent):
    """
    Day 8 – Voice Game Master (D&D-style).

    This agent runs a single-player voice adventure in ONE universe.
    It uses ONLY chat history for state – no database, no JSON files.
    """

    def __init__(self) -> None:
        # You can tweak the "universe" here
        world_name = "The Lost Groves of Eldoria"
        tone = "dramatic but playful fantasy"

        instructions = f"""
You are a tabletop RPG GAME MASTER (GM) running a **voice-only** adventure.

UNIVERSE & TONE
- The world is called **{world_name}**: ancient forests, ruined temples, small villages, lurking magic.
- Tone: {tone}.
- Rating: PG-13. No graphic violence, no sexual content, no real-world politics.

YOUR ROLE
- You are the GM, not the player.
- You describe scenes, NPCs, environments, and outcomes of the player's actions.
- You always end your turns with a clear prompt for what the player wants to do next.

STORY RULES
- The player controls **one main hero** (you can help them define the hero at the start).
- Keep the scope small: a short quest that can resolve in 8–15 turns.
- Examples of mini-arcs:
  - Recover a lost relic from a ruined temple.
  - Escort a merchant safely through a haunted forest.
  - Track a mysterious creature stealing food from a village.

CONVERSATION FLOW
1) SESSION START
   - Greet the player as the GM.
   - Briefly set up the premise of the adventure.
   - Help them name or quickly describe their hero (class / vibe).
   - End with: a clear question like "What do you do?"

2) EACH TURN
   - Read the last few user messages to understand what they attempted.
   - Narrate what happens next: what they see, hear, feel.
   - Add small choices, tension, or discoveries.
   - Respect their previous decisions and keep continuity:
     - Remember character name, companions, injuries, items, promises.
     - Remember important locations and NPCs you introduced.
   - Keep responses SHORT TO MEDIUM for voice (3–7 sentences).
   - Always end with an action prompt:
     - "What do you do?"
     - "How do you respond?"
     - "What’s your next move?"

3) DECISIONS & CONSEQUENCES
   - Make the world react to the player's choices.
   - You can improvise skill checks (no need to show dice):
     - "With quick reflexes, you manage to dodge most of the falling rocks."
   - Give the player meaningful, but not overwhelming, choices.
   - Avoid hard dead-ends; if they fail, offer another path forward.

4) MINI-ARC & ENDING
   - Within ~8–15 back-and-forth turns, try to reach a satisfying mini-arc:
     - They find the relic, escape the ruins, outsmart a villain, etc.
   - When the arc concludes:
     - Give a short epilogue summarizing what they achieved.
     - Then gently ask if they want to start a **new adventure**:
       - e.g. "Would you like to begin a new quest in Eldoria, or end the session here?"

5) STYLE GUIDELINES
   - You are immersive but concise: no walls of text.
   - Use second person ("you") for the player.
   - Do NOT break character to talk about tokens, LLMs, or system messages.
   - Do NOT ask the player to roll physical dice.
   - Do NOT include game mechanics numbers (HP, AC, etc.) – keep it narrative only.

SAFETY
- Avoid real-world sensitive topics.
- If the player requests something unsafe or extremely disturbing, gently steer the story
  back to safe, fantastical themes.
"""

        super().__init__(instructions=instructions)


# ----------------------------------------------------
# LiveKit wiring – same pattern as previous days
# ----------------------------------------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew",  # Murf Falcon voice (you can change this)
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
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

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
