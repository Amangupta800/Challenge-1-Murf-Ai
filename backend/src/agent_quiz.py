import json
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, RoomInputOptions, JobContext, WorkerOptions, cli
from livekit.plugins import murf, google, deepgram, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

CONTENT_PATH = Path(__file__).parent.parent / "shared-data/day4_tutor_content.json"


class QuizAgent(Agent):
    def __init__(self):
        instructions = """
You are Alicia (Murf Falcon voice).
You ask the user questions to test understanding.
One question at a time.
"""
        super().__init__(instructions=instructions)

    async def on_handoff(self, ctx, state):
        concept_id = state["concept_id"]
        content = json.load(open(CONTENT_PATH))
        for item in content:
            if item["id"] == concept_id:
                await ctx.send_text(f"Let’s quiz **{item['title']}**.\nQuestion: {item['sample_question']}")
                return


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="Alicia", style="Conversation"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=QuizAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
