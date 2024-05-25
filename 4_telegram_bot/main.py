from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, JobQueue
import json

TOKEN: Final = "TOKEN"
BOT_USERNAME: Final = "@hackai_firefighter_bot"

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! Thanks for chatting with me.')

def get_notification_message():
    with open('config.json', 'r') as file:
        data = json.load(file)
        return data['NOTIFICATION_MESSAGE']

async def send_automatic_notification(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = '-4268098032'  # Specify the chat ID where the message should be sent
    message = get_notification_message()  # Fetch the message from config.json
    await context.bot.send_message(chat_id=chat_id, text=message)

async def send_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    file_path = 'test.pf'  # Specify the path to the file you want to send
    await context.bot.send_document(chat_id=chat_id, document=open(file_path, 'rb'))

if __name__ == "__main__":
    print('Starting bot...')
    app = Application.builder().token(TOKEN).build()

    # Add command handler
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("sendfile", send_file_command))  # Add handler for sending file


    job_queue = app.job_queue

    #job_queue.run_repeating(send_automatic_notification, interval=10, first=0)

    app.run_polling()