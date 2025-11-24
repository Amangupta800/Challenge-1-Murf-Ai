import json
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, RoomInputOptions, JobContext, WorkerOptions, cli
from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

CONTENT_PATH = Path(__file__).parent.parent / "shared-data/day4_tutor_content.json"


class TeachBackAgent(Agent):
    def __init__(self):
        instructions = """
You are Ken (Murf Falcon voice).
Ask the user to explain the concept.
Then give friendly feedback and a mastery score.
"""
        super().__init__(instructions=instructions)

    async def on_handoff(self, ctx, state):
        concept_id = state["concept_id"]
        content = json.load(open(CONTENT_PATH))
        for item in content:
            if item["id"] == concept_id:
                await ctx.send_text(f"Teach me **{item['title']}** in your own words. I'm listening.")
                return


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="Ken", style="Conversation"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=TeachBackAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
