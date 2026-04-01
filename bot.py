import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import agent
import db
import embeddings
import linkedin_client

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

db.init_db()

MAX_TG_LENGTH = 4000  # leave headroom below 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str) -> str:
    if len(text) <= MAX_TG_LENGTH:
        return text
    return text[:MAX_TG_LENGTH] + "\n\n…_(truncated)_"


async def _safe_reply(msg, text: str):
    """Send reply, falling back to plain text if Markdown parse fails."""
    try:
        await msg.reply_text(_truncate(text), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await msg.reply_text(_truncate(text))


async def _safe_edit(msg, text: str):
    """Edit message, falling back to plain text if Markdown parse fails."""
    try:
        await msg.edit_text(_truncate(text), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        try:
            await msg.edit_text(_truncate(text))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = db.get_connection_count()
    text = (
        "*LinkedIn AI Assistant* 🔎\n\n"
        f"Local cache: *{count} connections*\n\n"
        "*Commands*\n"
        "/sync — pull your LinkedIn connections into cache\n"
        "/status — check cache size\n"
        "/clear — reset conversation history\n\n"
        "*Just ask me anything:*\n"
        "• _Find SaaS founders in India_\n"
        "• _Who in my network works at Stripe?_\n"
        "• _Show me investors in my 1st-degree connections_\n"
        "• _Find ex-Google people who started companies_"
    )
    await _safe_reply(update.message, text)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = db.get_connection_count()
    await update.message.reply_text(f"Cache: {count} connections stored locally.")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    agent.clear_history(update.effective_chat.id)
    await update.message.reply_text("Conversation history cleared.")


async def cmd_embed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = db.get_connection_count()
    status_msg = await update.message.reply_text(
        f"🧠 Building semantic index for {count} connections...\n"
        "First run downloads ~80MB model. Give it a minute."
    )

    loop = asyncio.get_event_loop()

    def progress_cb(text: str):
        asyncio.run_coroutine_threadsafe(status_msg.edit_text(text), loop)

    try:
        indexed = await asyncio.to_thread(embeddings.build_index, progress_cb)
        await _safe_edit(status_msg, f"✅ Semantic index built — *{indexed}* connections indexed.\n\nYour searches now use meaning, not just keywords.")
    except Exception as e:
        logger.error(f"Embed failed: {e}", exc_info=True)
        await status_msg.edit_text(f"❌ Indexing failed: {str(e)[:300]}")


async def cmd_sync(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text(
        "🔄 Connecting to LinkedIn — this may take a few minutes..."
    )

    loop = asyncio.get_event_loop()

    def progress_cb(text: str):
        asyncio.run_coroutine_threadsafe(status_msg.edit_text(text), loop)

    try:
        count = await asyncio.to_thread(linkedin_client.sync_connections, progress_cb)
        await _safe_edit(
            status_msg,
            f"✅ Sync complete! *{count}* connections now in local cache.\n\n"
            "You can now ask me to find people in your network.",
        )
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        await status_msg.edit_text(
            f"❌ Sync failed: {str(e)[:300]}\n\n"
            "Check your LINKEDIN_EMAIL and LINKEDIN_PASSWORD in .env"
        )


# ---------------------------------------------------------------------------
# Message handler (main agent entrypoint)
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    thinking_msg = await update.message.reply_text("🤔 Thinking...")

    async def progress_cb(status: str):
        try:
            await thinking_msg.edit_text(status)
        except Exception:
            pass

    try:
        response = await agent.run_agent(chat_id, user_text, progress_cb)
        await _safe_edit(thinking_msg, response)
    except Exception as e:
        logger.error(f"Agent error for chat {chat_id}: {e}", exc_info=True)
        await thinking_msg.edit_text(f"❌ Error: {str(e)[:300]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

    # Pre-initialize ChromaDB and the embedding model in the main thread.
    # ChromaDB's Rust backend fails to initialize when first called from a
    # thread pool thread (via asyncio.to_thread), producing "bindings" errors.
    try:
        logger.info("Initialising semantic index...")
        embeddings.get_collection()
        embeddings.get_model()
        logger.info("Semantic index ready.")
    except Exception as e:
        logger.warning(f"Semantic index unavailable at startup: {e}")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("sync", cmd_sync))
    app.add_handler(CommandHandler("embed", cmd_embed))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
