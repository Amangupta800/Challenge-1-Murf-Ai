import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent_learn import LearnAgent
from agent_quiz import QuizAgent
from agent_teachback import TeachBackAgent

logger = logging.getLogger("router")
load_dotenv(".env.local")


# Decide which agent + which Murf voice to use
def build_agent_for_mode(mode: str):
    mode = mode.lower().strip()

    if mode == "learn":
        # Murf Falcon Matthew
        return LearnAgent(), murf.TTS(
            voice="Matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        )

    elif mode == "quiz":
        # Murf Falcon Alicia
        return QuizAgent(), murf.TTS(
            voice="Alicia",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        )

    elif mode == "teach_back":
        # Murf Falcon Ken
        return TeachBackAgent(), murf.TTS(
            voice="Ken",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # ✅ Read mode from ENV instead of ctx.job.vars (which crashes in dev)
    mode = os.getenv("TUTOR_MODE", "learn")  # default: learn
    logger.info(f"[Router] Starting in mode: {mode}")

    agent, tts_model = build_agent_for_mode(mode)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=tts_model,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # optional metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
