import logging
from datetime import datetime

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
)
from livekit.agents import function_tool, RunContext
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import SQLite-backed helpers
from fraud_db import load_fraud_case, save_fraud_case

# ----------------------------------------------------
# Setup
# ----------------------------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# ----------------------------------------------------
# Fraud Alert Agent
# ----------------------------------------------------


class FraudAlertAgent(Agent):
    """
    Day 6 – Fraud Alert Voice Agent.

    Behavior:

    - Introduce as fictional bank fraud department.
    - Verify customer with a *non-sensitive* security question from the case.
    - Read suspicious transaction (amount, merchant, masked card, time, location).
    - Ask if the transaction is legitimate.
    - Update the case status in SQLite as:
        - confirmed_safe
        - confirmed_fraud
        - verification_failed
      and add an outcome note.
    """

    def __init__(self) -> None:
        # Load the single fraud case from SQLite "database"
        self.case = load_fraud_case()

        # Never expose verificationAnswer directly.
        verification_question = self.case.get("verificationQuestion", "a security question")
        bank_name = "ICC Bank"  # you can rename the fictional bank

        instructions = f"""
You are a FRAUD DETECTION VOICE AGENT for a fictional bank called "{bank_name}".

SAFETY:
- DO NOT ask for full card numbers, PINs, OTPs, passwords, or any sensitive credentials.
- ONLY use the NON-SENSITIVE security question stored in the fraud case to verify the user.


CURRENT FRAUD CASE (DO NOT READ VERBATIM, USE IT TO GUIDE CONVERSATION):
- Customer name: {self.case.get("userName", "Aman Gupta")}
- Security identifier (internal only): {self.case.get("securityIdentifier", "N/A")}
- Card ending: **** {self.case.get("cardEnding", "XXXX")}
- Merchant: {self.case.get("transactionName", "")}
- Amount: {self.case.get("transactionAmount", "")}
- Time: {self.case.get("transactionTime", "")}
- Category: {self.case.get("transactionCategory", "")}
- Source: {self.case.get("transactionSource", "")}
- Location: {self.case.get("transactionLocation", "")}
- Current status: {self.case.get("status", "")}
- Verification question: "{verification_question}"

FLOW YOU MUST FOLLOW:

1) GREETING & CONTEXT
   - Politely introduce yourself:
     e.g. "Hello, this is the fraud monitoring team from {bank_name}."
   - Explain this is about a suspicious transaction on a card ending with the last digits.
 

2) BASIC VERIFICATION
   - Use ONLY the verification question:
     "{verification_question}"
   - Ask the user this question in your own words.
   - Compare their answer USING the `check_verification_answer` tool.
   - If the tool result says verification_failed:
       - Politely say you cannot continue without verification.
       - Call `mark_verification_failed` tool.
       - End the call.

3) SUSPICIOUS TRANSACTION DETAILS
   - If verification passes:
       - Briefly describe the transaction:
         - The amount,
         - The merchant,
         - The approximate time/date,
         - That it was an e-commerce transaction on the card ending with the last 4 digits.
       - Ask clearly:
         "Did you make this transaction yourself?" (YES/NO style question).

4) DECISION LOGIC
   - If the user confirms they DID make the transaction:
       - Reassure them that the card is safe.
       - Call `mark_case_safe` tool with a short summary note.
       - Give a short verbal summary: "We have marked this as a safe, legitimate transaction."
       - End politely.

   - If the user says they DID NOT make the transaction or expresses clear doubt:
       - Call `mark_case_fraudulent` tool with a short summary note:
         e.g. "Customer denies transaction; treat as fraud in this demo."
       - Explain that in a real system the card would be blocked and a dispute started
         (but make clear this is a demo).
       - Reassure them.
       - End politely.

5) STYLE:
   - Calm, professional, reassuring.
   - Short, simple sentences.
   - One question at a time.
   - No medical, legal, or financial promises – it's a demo.
"""

        super().__init__(instructions=instructions)

    # ---------------- Tools ----------------

    @function_tool
    async def check_verification_answer(
        self,
        context: RunContext,
        user_answer: str,
    ) -> dict:
        """
        Compare the user's answer to the stored verification answer (case-insensitive).

        Returns:
          {
            "matched": bool,
            "message": short_text_explaining_match
          }
        """
        expected = (self.case.get("verificationAnswer") or "").strip().lower()
        got = (user_answer or "").strip().lower()

        matched = bool(expected) and (expected == got)
        if matched:
            msg = "Verification answer matched. You may proceed with discussing the transaction."
        else:
            msg = "Verification answer did not match. You should not proceed."

        return {"matched": matched, "message": msg}

    @function_tool
    async def mark_case_safe(
        self,
        context: RunContext,
        note: str,
    ) -> dict:
        """
        Mark the fraud case as CONFIRMED SAFE.

        Args:
          note: Short explanation of why it was marked safe.
        """
        self.case["status"] = "confirmed_safe"
        self.case["outcomeNote"] = note
        self.case["lastUpdated"] = datetime.now().isoformat(timespec="seconds")
        save_fraud_case(self.case)
        return {
            "status": self.case["status"],
            "outcomeNote": self.case["outcomeNote"],
        }

    @function_tool
    async def mark_case_fraudulent(
        self,
        context: RunContext,
        note: str,
    ) -> dict:
        """
        Mark the fraud case as CONFIRMED FRAUDULENT.

        Args:
          note: Short explanation of why it was marked as fraud.
        """
        self.case["status"] = "confirmed_fraud"
        self.case["outcomeNote"] = note
        self.case["lastUpdated"] = datetime.now().isoformat(timespec="seconds")
        save_fraud_case(self.case)
        return {
            "status": self.case["status"],
            "outcomeNote": self.case["outcomeNote"],
        }

    @function_tool
    async def mark_verification_failed(
        self,
        context: RunContext,
        note: str,
    ) -> dict:
        """
        Mark the fraud case as VERIFICATION FAILED and stop the call.

        Args:
          note: Short explanation (e.g. wrong answer to security question).
        """
        self.case["status"] = "verification_failed"
        self.case["outcomeNote"] = note
        self.case["lastUpdated"] = datetime.now().isoformat(timespec="seconds")
        save_fraud_case(self.case)
        return {
            "status": self.case["status"],
            "outcomeNote": self.case["outcomeNote"],
        }


# ----------------------------------------------------
# LiveKit wiring (same pattern as previous days)
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
            voice="en-US-matthew",  # choose any Murf Falcon voice you like
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
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
