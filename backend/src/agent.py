import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

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

# ----------------------------------------------------
# Setup & JSON "Database"
# ----------------------------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")

BASE_PATH = Path(__file__).parent.parent
SHARED_DATA = BASE_PATH / "shared-data"
CATALOG_PATH = SHARED_DATA / "day7_catalog.json"
ORDERS_PATH = SHARED_DATA / "day7_orders.json"

SHARED_DATA.mkdir(parents=True, exist_ok=True)


def _default_catalog() -> Dict[str, Any]:
    """Default catalog used if day7_catalog.json doesn't exist yet."""
    return {
        "items": [
            # Groceries
            {
                "id": 1,
                "name": "Whole Wheat Bread",
                "category": "groceries",
                "price": 45,
                "tags": ["bread", "sandwich", "wheat"],
            },
            {
                "id": 2,
                "name": "Eggs (12 pack)",
                "category": "groceries",
                "price": 80,
                "tags": ["eggs", "protein", "breakfast"],
            },
            {
                "id": 3,
                "name": "Milk 1L",
                "category": "groceries",
                "price": 60,
                "tags": ["milk", "dairy"],
            },
            {
                "id": 4,
                "name": "Peanut Butter (Large)",
                "category": "groceries",
                "price": 180,
                "tags": ["peanut_butter", "spread"],
            },
            {
                "id": 5,
                "name": "Butter 200g",
                "category": "groceries",
                "price": 90,
                "tags": ["butter", "dairy"],
            },
            # Snacks
            {
                "id": 6,
                "name": "Potato Chips (Classic)",
                "category": "snacks",
                "price": 35,
                "tags": ["chips", "snack"],
            },
            {
                "id": 7,
                "name": "Chocolate Bar",
                "category": "snacks",
                "price": 25,
                "tags": ["chocolate", "snack"],
            },
            {
                "id": 8,
                "name": "Salted Peanuts",
                "category": "snacks",
                "price": 40,
                "tags": ["peanuts", "snack"],
            },
            # Prepared food
            {
                "id": 9,
                "name": "Veg Sandwich (Ready to Eat)",
                "category": "prepared_food",
                "price": 120,
                "tags": ["sandwich", "veg"],
            },
            {
                "id": 10,
                "name": "Cheese Pizza (Medium)",
                "category": "prepared_food",
                "price": 250,
                "tags": ["pizza", "cheese"],
            },
            {
                "id": 11,
                "name": "Pasta Penne 500g",
                "category": "groceries",
                "price": 110,
                "tags": ["pasta", "penne"],
            },
            {
                "id": 12,
                "name": "Tomato Pasta Sauce Jar",
                "category": "groceries",
                "price": 130,
                "tags": ["sauce", "pasta"],
            },
        ]
    }


def load_catalog() -> Dict[str, Any]:
    """Load catalog JSON, creating a default one if needed."""
    if not CATALOG_PATH.exists():
        catalog = _default_catalog()
        with CATALOG_PATH.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        logger.info("Created default Day 7 catalog at %s", CATALOG_PATH)
        return catalog

    try:
        with CATALOG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "items" not in data:
            raise ValueError("Catalog JSON must contain an 'items' list")
        return data
    except Exception as e:
        logger.error("Failed to load catalog JSON: %s", e)
        # fallback to default in-memory catalog
        return _default_catalog()


def load_orders() -> List[Dict[str, Any]]:
    """Load existing orders from JSON; if none, return empty list."""
    if not ORDERS_PATH.exists():
        return []
    try:
        with ORDERS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("Orders JSON was not a list; resetting.")
        return []
    except Exception as e:
        logger.error("Failed to load orders JSON: %s", e)
        return []


def save_orders(orders: List[Dict[str, Any]]) -> None:
    """Persist all orders to JSON."""
    with ORDERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d orders to %s", len(orders), ORDERS_PATH)


# Simple recipe mapping for "ingredients for X" behavior
RECIPES = {
    "peanut butter sandwich": [
        "Whole Wheat Bread",
        "Peanut Butter (Large)",
    ],
    "pasta for two": [
        "Pasta Penne 500g",
        "Tomato Pasta Sauce Jar",
    ],
    "simple pasta": [
        "Pasta Penne 500g",
        "Tomato Pasta Sauce Jar",
    ],
}


# ----------------------------------------------------
# Grocery Ordering Agent
# ----------------------------------------------------


class GroceryOrderingAgent(Agent):
    """
    Day 7 – Food & Grocery Ordering Voice Agent.

    Behavior:
    - Acts as a friendly grocery & food ordering assistant for a Swiggy App.
    - Loads a product catalog from JSON.
    - Maintains a cart (items + quantities).
    - Supports "ingredients for X" by mapping recipes to multiple catalog items.
    - When user is done, saves final order as JSON.
    """

    def __init__(self) -> None:
        self.catalog = load_catalog()
        # flatten items for easier lookup
        self.items = self.catalog.get("items", [])
        self.cart: List[Dict[str, Any]] = []

        brand_name = "Blinkit"

        instructions = f"""
You are a FOOD & GROCERY ORDERING VOICE ASSISTANT for a fictional store called "{brand_name}".

YOUR GOAL:
- Help the user order groceries, snacks, and simple meal ingredients from a catalog.
- Maintain a cart in memory.
- When the user is done, confirm the order and place it (using tools).

CATALOG:
- You have access to a catalog via tools (you do NOT invent new items that are clearly not present).
- Items include groceries (bread, eggs, milk, pasta, sauce), snacks (chips, chocolate), and prepared food (sandwiches, pizza).

CART BEHAVIOR:
- When user asks for items (e.g. "add 2 milk and a bread"), call `add_item_to_cart` tool.
- When user asks for "ingredients for X" (e.g. peanut butter sandwich), call `add_recipe_to_cart`.
- When user wants to remove/change things, call `remove_item_from_cart` or `update_item_quantity`.
- When asked "what's in my cart", call `get_cart_summary`.

ORDER PLACEMENT:
- When user says they are done (e.g. "that's all", "place my order", "checkout"):
  1. Use `get_cart_summary` to understand the cart.
  2. Confirm the final items and total verbally.
  3. Call `place_order` tool to persist the order.
  4. Tell the user the order has been placed.

STYLE:
- Friendly, concise, like a quick commerce app assistant.
- Ask clarifying questions if item or quantity is ambiguous.
- Always confirm important cart changes.
"""

        super().__init__(instructions=instructions)

    # --------------- Helper Methods ---------------

    def _find_item(self, query: str) -> Optional[Dict[str, Any]]:
        """Very simple fuzzy match by name or tags."""
        q = query.strip().lower()
        if not q:
            return None

        # Exact name match first
        for item in self.items:
            if item["name"].lower() == q:
                return item

        # Contains match in name
        for item in self.items:
            if q in item["name"].lower():
                return item

        # Match by tag
        for item in self.items:
            for tag in item.get("tags", []):
                if q in tag.lower():
                    return item

        return None

    def _add_to_cart(self, item: Dict[str, Any], quantity: int) -> None:
        if quantity <= 0:
            return
        for c in self.cart:
            if c["id"] == item["id"]:
                c["quantity"] += quantity
                return
        self.cart.append(
            {
                "id": item["id"],
                "name": item["name"],
                "category": item.get("category", ""),
                "price": item["price"],
                "quantity": quantity,
            }
        )

    def _total(self) -> int:
        return sum(int(i["price"]) * int(i["quantity"]) for i in self.cart)

    # --------------- Tools ---------------

    @function_tool
    async def add_item_to_cart(
        self,
        context: RunContext,
        item_name: str,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """
        Add a specific item from the catalog to the cart.

        Args:
          item_name: Name or rough description of the item.
          quantity: How many units to add (default 1).
        """
        quantity = max(1, int(quantity or 1))
        item = self._find_item(item_name)
        if not item:
            return {
                "success": False,
                "message": f"Could not find any item matching '{item_name}' in the catalog.",
            }

        self._add_to_cart(item, quantity)
        return {
            "success": True,
            "message": f"Added {quantity} x {item['name']} to the cart.",
            "cart_size": len(self.cart),
            "cart_total": self._total(),
        }

    @function_tool
    async def add_recipe_to_cart(
        self,
        context: RunContext,
        recipe_name: str,
        servings: int = 1,
    ) -> Dict[str, Any]:
        """
        Add ingredients for a simple recipe (e.g. peanut butter sandwich, pasta for two).

        Args:
          recipe_name: Name like 'peanut butter sandwich' or 'pasta for two'.
          servings: Optional multiplier (simple scaling).
        """
        servings = max(1, int(servings or 1))
        key = recipe_name.strip().lower()
        # Try exact key, then some light normalization
        recipe = RECIPES.get(key)
        if not recipe:
            # simple normalization for "pasta" or "pasta for two"
            if "peanut butter" in key and "sandwich" in key:
                recipe = RECIPES["peanut butter sandwich"]
            elif "pasta" in key and "two" in key:
                recipe = RECIPES["pasta for two"]
            elif "pasta" in key:
                recipe = RECIPES["simple pasta"]

        if not recipe:
            return {
                "success": False,
                "message": f"I don't have a recipe mapping for '{recipe_name}' yet.",
            }

        added_items = []
        for name in recipe:
            item = self._find_item(name)
            if item:
                self._add_to_cart(item, servings)
                added_items.append(item["name"])

        if not added_items:
            return {
                "success": False,
                "message": f"Recipe '{recipe_name}' matched no items in the catalog.",
            }

        return {
            "success": True,
            "message": f"Added ingredients for {recipe_name}: {', '.join(added_items)}.",
            "cart_size": len(self.cart),
            "cart_total": self._total(),
        }

    @function_tool
    async def remove_item_from_cart(
        self,
        context: RunContext,
        item_name: str,
    ) -> Dict[str, Any]:
        """
        Remove an item from the cart by name.
        """
        item = self._find_item(item_name)
        if not item:
            return {
                "success": False,
                "message": f"Could not find any item matching '{item_name}' to remove.",
            }

        before = len(self.cart)
        self.cart = [c for c in self.cart if c["id"] != item["id"]]
        after = len(self.cart)

        if before == after:
            return {
                "success": False,
                "message": f"'{item['name']}' was not in the cart.",
            }

        return {
            "success": True,
            "message": f"Removed {item['name']} from the cart.",
            "cart_size": len(self.cart),
            "cart_total": self._total(),
        }

    @function_tool
    async def update_item_quantity(
        self,
        context: RunContext,
        item_name: str,
        quantity: int,
    ) -> Dict[str, Any]:
        """
        Update the quantity of an item in the cart.

        Args:
          item_name: The item to update.
          quantity: New quantity (if 0 or less, item is removed).
        """
        item = self._find_item(item_name)
        if not item:
            return {
                "success": False,
                "message": f"Could not find any item matching '{item_name}' to update.",
            }

        qty = int(quantity)
        updated = False
        for c in self.cart:
            if c["id"] == item["id"]:
                if qty <= 0:
                    self.cart.remove(c)
                    msg = f"Removed {item['name']} from the cart."
                else:
                    c["quantity"] = qty
                    msg = f"Updated {item['name']} quantity to {qty}."
                updated = True
                break

        if not updated:
            if qty <= 0:
                return {
                    "success": False,
                    "message": f"{item['name']} was not in the cart.",
                }
            # add if not present
            self._add_to_cart(item, qty)
            msg = f"{item['name']} was not in the cart, so I added {qty}."

        return {
            "success": True,
            "message": msg,
            "cart_size": len(self.cart),
            "cart_total": self._total(),
        }

    @function_tool
    async def get_cart_summary(
        self,
        context: RunContext,
    ) -> Dict[str, Any]:
        """
        Return the current cart contents and total amount.
        """
        return {
            "items": self.cart,
            "total": self._total(),
        }

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        customer_name: str = "",
        delivery_note: str = "",
    ) -> Dict[str, Any]:
        """
        Place the current order and save it into a JSON file.

        Args:
          customer_name: Optional free-text name.
          delivery_note: Optional delivery instructions or address text.
        """
        if not self.cart:
            return {
                "success": False,
                "message": "Your cart is empty. There is nothing to place.",
            }

        orders = load_orders()
        order_id = len(orders) + 1
        now = datetime.now().isoformat(timespec="seconds")

        order = {
            "order_id": order_id,
            "created_at": now,
            "customer_name": customer_name or "Guest",
            "delivery_note": delivery_note,
            "items": self.cart,
            "total": self._total(),
            "status": "placed",
        }

        orders.append(order)
        save_orders(orders)

        # clear cart after placing
        self.cart = []

        return {
            "success": True,
            "message": f"Order #{order_id} has been placed.",
            "order_id": order_id,
            "total": order["total"],
            "created_at": order["created_at"],
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
            voice="en-US-matthew",  # Murf Falcon voice
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
        agent=GroceryOrderingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
