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
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ----------------------------------------------------
# Setup
# ----------------------------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")

BASE_DIR = Path(__file__).parent.parent
FAQ_PATH = BASE_DIR / "shared-data" / "day5_blink_faq.json"
LEADS_DIR = BASE_DIR / "leads"
LEADS_DIR.mkdir(exist_ok=True)


def load_faq() -> list[dict]:
    """Load Blink Digital FAQ / info from JSON."""
    if not FAQ_PATH.exists():
        logger.warning("FAQ file not found at %s, using empty FAQ", FAQ_PATH)
        return []
    try:
        with FAQ_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("FAQ JSON must be a list")
        return data
    except Exception as e:
        logger.warning("Failed to read FAQ JSON (%s), using empty FAQ", e)
        return []


def find_best_faq(question: str, faqs: list[dict]) -> dict | None:
    """
    Very simple keyword-based FAQ lookup.
    Checks the question against each FAQ's 'keywords' list.
    """
    q = question.lower()
    best = None
    best_score = 0

    for item in faqs:
        keywords = item.get("keywords", [])
        score = 0
        for kw in keywords:
            if kw.lower() in q:
                score += 1
        if score > best_score:
            best_score = score
            best = item

    if best_score == 0:
        return None
    return best


# ----------------------------------------------------
# SDR Agent
# ----------------------------------------------------


class BlinkSDRAgent(Agent):
    """
    Day 5 – Simple FAQ SDR + Lead Capture for Blink Digital.
    """

    def __init__(self) -> None:
        self.faq_data = load_faq()

        # Lead state in memory
        self.lead: dict[str, str | None] = {
            "name": None,
            "company": None,
            "email": None,
            "role": None,
            "use_case": None,
            "team_size": None,
            "timeline": None,
            "notes": None,
        }

        faq_overview_lines = []
        for item in self.faq_data:
            faq_overview_lines.append(
                f"- {item.get('id', '')}: {item.get('question', '')}"
            )
        faq_overview = "\n".join(faq_overview_lines) if faq_overview_lines else "No FAQ loaded."

        instructions = f"""
You are a Sales Development Representative (SDR) name Mathew for Blink Digital, a digital-first creative and technology agency in India.

Your goals:
1) Greet visitors warmly and professionally.
2) Quickly understand what they are working on and why they are interested.
3) Answer basic questions about Blink Digital using the provided FAQ content.
4) Collect key lead details in a natural, conversational way.
5) When the user is done, summarize the lead and call the save_lead tool.

Company context (Blink Digital):
- Digital-first creative & technology agency in India.
- Services: web & app development, UX/UI design, digital campaigns, content & social, performance marketing, creative tech.
- Typical clients: startups, high-growth tech companies, established brands.
- Pricing: project / retainer based, depends on scope and complexity (you CANNOT give exact quotes).
- You must NOT invent detailed pricing, contract terms, or legal / financial promises.

FAQ content available (for lookup_faq tool):
{faq_overview}

FAQ behavior:
- When the user asks about Blink Digital, services, who it's for, or pricing:
  - Use the lookup_faq tool with the user question.
  - Answer based ONLY on the content returned.
  - If the FAQ doesn't cover something, say you are not sure and that a human will follow up.

Lead capture:
- Over the course of the conversation, politely collect:
  - name
  - company
  - email
  - role
  - use_case (what they want to build or solve)
  - team_size
  - timeline (now / soon / later)
- Ask for these naturally and not all at once.
- Do NOT make up values. Only use what the user actually said.

End of call:
- Detect when the user is done (phrases like "that's all", "I'm done", "thanks, that's it").
- Before ending:
  - Briefly summarize who they are, what they want, and rough timeline.
  - Then call save_lead with the fields you have.
  - If some fields are missing, pass an empty string for them.
- After calling save_lead, close politely.

Style:
- Friendly, concise, slightly consultative.
- Ask one clear question at a time.
- Keep answers grounded in the FAQ and company description.
"""
        super().__init__(instructions=instructions)

    # ---------------- FAQ tool ----------------

    @function_tool
    async def lookup_faq(self, context: RunContext, question: str) -> dict:
        """
        Look up the most relevant FAQ entry for a given user question.

        Args:
            question: The user's question in natural language.

        Returns:
            A dict with 'question' and 'answer' fields (or a message if nothing matched).
        """
        if not self.faq_data:
            return {
                "found": False,
                "message": "No FAQ content is loaded.",
                "question": "",
                "answer": "",
            }

        best = find_best_faq(question, self.faq_data)
        if not best:
            return {
                "found": False,
                "message": "No matching FAQ entry was found for this question.",
                "question": "",
                "answer": "",
            }

        return {
            "found": True,
            "message": "FAQ match found.",
            "question": best.get("question", ""),
            "answer": best.get("answer", ""),
        }

    # ---------------- Lead save tool ----------------

    @function_tool
    async def save_lead(
        self,
        context: RunContext,
        name: str,
        company: str,
        email: str,
        role: str,
        use_case: str,
        team_size: str,
        timeline: str,
        notes: str = "",
    ) -> str:
        """
        Save the collected lead to a JSON file.

        Args:
            name: Person's name (or empty string if not provided).
            company: Company name.
            email: Email address.
            role: Their role or title.
            use_case: Short description of what they want Blink Digital for.
            team_size: Approximate team size.
            timeline: When they want to start (now / soon / later).
            notes: Any extra notes or summary.
        """
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": name,
            "company": company,
            "email": email,
            "role": role,
            "use_case": use_case,
            "team_size": team_size,
            "timeline": timeline,
            "notes": notes,
        }

        # Save per-lead file
        file_path = LEADS_DIR / f"lead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

        logger.info("Saved lead to %s", file_path)

        return f"Lead saved to {file_path.name}"
    

# ----------------------------------------------------
# LiveKit wiring (similar to previous days)
# ----------------------------------------------------


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
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
        agent=BlinkSDRAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
