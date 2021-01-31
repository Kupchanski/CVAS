import logging

from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
import cv2
from utils import get_object_coordinates, adjust_coordinates, generate_result_frame
from segmentation import get_segmentation
from main import Args, generate_objects
import numpy as np
import tempfile
import uuid
import os
import socket
from google.cloud import storage
from google.oauth2 import service_account
from utils import get_parent_dir_path
import random
from telegram import MessageEntity
import redis

r = redis.Redis(connection_pool=redis.BlockingConnectionPool(max_connections=30))

# Enable logging
logging.basicConfig(
    filename=f'{get_parent_dir_path()}/logs.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def target_dir():
    return f'{get_parent_dir_path()}/temp/targets'

def set_target_flag(key, flag):
    global r

    try:
        r.set(key, flag)
    except:
        pass


def get_target_flag(key):
    global r

    try:
        return int(r.get(key))
    except:
        pass

    return 0

def get_reference_target_path():
    return 'data/raw/target3.jpeg'

def error_handler(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'Exception {context.error}')

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def help(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! Send me a video (<20mb), youtuve link or photo and I will handle it. Use `/status` to check my status.')
    update.message.reply_text(
        'Use /set_image for using your image as replacement.')

def status(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(f'I am ok, thank you! I am here `{socket.gethostname()}`')

def funny_video_names():
    return [
            'granny_going_to_gym_',
            'my_porn_',
            'xxx_withoutsms_',
            'first_jump_',
            'parachutes_'
        ]

def get_file_names(ext):
    temp_dir = tempfile.gettempdir()

    file_name = f'{random.choice(funny_video_names())}{uuid.uuid4()}'
    original_file_name = f"{temp_dir}/{file_name}.{ext}"
    generated_name = f"{file_name}_final.mp4"
    generated_file_name = f"{temp_dir}/{generated_name}"

    return (file_name, original_file_name, generated_name, generated_file_name)

def youtube_handler(update: Update, context: CallbackContext) -> None:
    message = update.message['text'].split(' ')

    url = None

    for text in message:
    	if text.startswith('https://') or text.startswith('http://') or text.startswith('www.') or text.startswith('youtu'):
    		url = text

    if url is not None:
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=f'Ok, I found youtube link {url}. I will try to download it if it\'s smaller than 100Mb.')

        ext = os.popen(f'youtube-dl --get-filename -o "%(ext)s" {url}').read().strip()

        file_name, original_file_name, generated_name, generated_file_name = get_file_names(ext)

        result = os.system(f'youtube-dl  -f \'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=webm]+bestaudio[ext=webm]/best\' --max-filesize 100M {url} -o {original_file_name}')

        if result == 0:
            video_processing(update, context, file_name, original_file_name, generated_name, generated_file_name)
        else:
            context.bot.send_message(chat_id=update.message.chat_id, text=f'I can\'t upload your video')
    else:
        context.bot.send_message(parse_mode='markdown', chat_id=update.message.chat_id,
                                 text=f'I didn\'t find youtube links')

def video_processing(update: Update, context: CallbackContext, file_name, original_file_name, generated_name, generated_file_name) -> None:
    user_id = update.effective_user.id

    temp_dir = tempfile.gettempdir()
    generated_file_with_audio_name = f"{temp_dir}/with_audio_{generated_name}"

    context.bot.send_message(parse_mode='markdown', chat_id=update.message.chat_id, text=f'Nice, I received your video `{file_name}`')

    input_video_cap = cv2.VideoCapture(original_file_name)
    frames_counts = int(input_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_video_cap.release()

    context.bot.send_message(parse_mode='markdown', chat_id=update.message.chat_id, text=f'Stay tune, your video `{file_name}` has {frames_counts} frames, I need ~{round((frames_counts)/ 60)} minutes to process it.')

    os.system(f'python {get_parent_dir_path()}/cvas/main.py -s 0 -r {get_replacement_path(user_id)} -i {original_file_name} -o {generated_file_name}')
    os.system(f'ffmpeg -i {generated_file_name} -i {original_file_name} -map 0:v:0 -map 1:a:0 -shortest {generated_file_with_audio_name}; mv {generated_file_with_audio_name} {generated_file_name}')

    logger.info(f'Uploading generated video {generated_file_name} to gcs')

    credentials = service_account.Credentials.from_service_account_file(f'{get_parent_dir_path()}/creds_st.json')
    bucket_name = 'cvas'

    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(generated_name)
    blob.upload_from_filename(generated_file_name)

    context.bot.send_message(parse_mode='markdown', chat_id=update.message.chat_id, text=f'I have processed your video `{file_name}`.')
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=f'You can download it here https://storage.googleapis.com/{bucket_name}/{generated_name}.')
    logger.info(f'Finished with {generated_file_name} video')

    os.remove(original_file_name)
    os.remove(generated_file_name)

    logger.info(f"Dropped temp files {original_file_name}, {generated_file_name}")

def video_handling(update: Update, context: CallbackContext) -> None:
    file = context.bot.get_file(update.message.video.file_id)
    logger.info(f'Received video {file}')

    file_name, original_file_name, generated_name, generated_file_name = get_file_names('mp4')

    file.download(original_file_name)
    logger.info(f'Video downloaded to {original_file_name}')

    video_processing(update, context, file_name, original_file_name, generated_name, generated_file_name)


def set_image(update: Update, context: CallbackContext) -> None:
    context.bot.send_message(chat_id=update.message.chat_id, text='Wow, you want a customization. Ok, send me an image that you want usig for the replacement.')

    set_target_flag(update.effective_user.id, -1)

def set_image_impl(file, update: Update, context: CallbackContext) -> None:
    user_id = update.effective_user.id
    logger.info(f"Start updaeting replacement for {user_id} with {file}")
    context.bot.send_message(chat_id=update.message.chat_id, text='Nice, save your image.')

    temp_dir = target_dir()
    original_file_name = f"{temp_dir}/{user_id}"
    file.download(original_file_name)

    # replacement = cv2.imread(get_target_path())
    context.bot.send_message(chat_id=update.message.chat_id, text='Ok, I have saved your image. Now you can send me another image to check.')
    set_target_flag(update.effective_user.id, 1)

def get_replacement_path(key):
    path = get_reference_target_path()
    if get_target_flag(key) == 1:
        user_path = f'{target_dir()}/{key}'

        if os.path.isfile(user_path):
            path = user_path

    return path

def get_replacement(key):
    return cv2.imread(get_replacement_path(key))

def image_handling(update: Update, context: CallbackContext) -> None:
    file = context.bot.get_file(update.message.photo[-1].file_id)
    user_id = update.effective_user.id

    if get_target_flag(update.effective_user.id) == -1:
        set_image_impl(file, update, context)
        return

    logger.info(f"Start processing image {file}")
    context.bot.send_message(chat_id=update.message.chat_id, text='Nice photo, let me replace advs there')

    temp_dir = tempfile.gettempdir()
    original_file_name = f"{temp_dir}/{uuid.uuid4()}"
    file.download(original_file_name)
    frame = cv2.imread(original_file_name)

    replacement = get_replacement(user_id)

    logger.info(f"Read innputs {frame.shape} {replacement.shape}")

    args = Args().extract([
      '-i', 'data/raw/reference.mkv',
      '-o', 'data/processed/out.mp4',
      '-r', 'data/raw/target.jpg',
      '-segm', '0'
    ])

    segment_frame = get_segmentation(frame, args)

    logger.info(f"Segmented image {segment_frame.shape}")

    object_coordinates = get_object_coordinates(segment_frame)
    object_coordinates = adjust_coordinates(object_coordinates)
    frames_objects = [object_coordinates]
    global_objects = generate_objects(frames_objects)

    if len(global_objects) == 0:
        context.bot.send_message(chat_id=update.message.chat_id, text='Sorry, I did\'t find any billboards in your image.')
    else:

        generated_frame = frame.copy()
        generated_frame = generate_result_frame(generated_frame, 0, replacement, global_objects)

        logger.info(f"Generated result {generated_frame.shape}")


        # debug

        context.bot.send_message(chat_id=update.message.chat_id, text='Here is a debug.')

        send_img = np.concatenate((frame, segment_frame, generated_frame), axis=1)

        generated_file_name = f"{temp_dir}/gen_debug_{uuid.uuid4()}.png"

        cv2.imwrite(generated_file_name, send_img)

        context.bot.send_photo(
            chat_id=update.effective_chat.id, photo=open(generated_file_name, "rb")
        )

        ## debug

        context.bot.send_message(chat_id=update.message.chat_id, text='Here is a result.')


        generated_file_name = f"{temp_dir}/gen_{uuid.uuid4()}.png"

        cv2.imwrite(generated_file_name, generated_frame)

        context.bot.send_photo(
            chat_id=update.effective_chat.id, photo=open(generated_file_name, "rb")
        )

        logger.info("Sent result")

        os.remove(generated_file_name)
        logger.info(f"Dropped temp files {generated_file_name}")

    os.remove(original_file_name)

    logger.info(f"Dropped temp files {original_file_name}")

def main():
    os.system(f'mkdir -p {target_dir()}')

    """Run bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    # updater = Updater("some_token", use_context=True, workers=16)

    updater = Updater('some_token', use_context=True)
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("help", help, run_async=True))
    dispatcher.add_handler(CommandHandler("start", help, run_async=True))
    dispatcher.add_handler(CommandHandler("status", status, run_async=True))
    dispatcher.add_handler(CommandHandler("set_image", set_image, run_async=True))

    dispatcher.add_handler(MessageHandler(Filters.photo, image_handling, run_async=True))
    dispatcher.add_handler(MessageHandler(Filters.video, video_handling, run_async=True))
    dispatcher.add_handler(MessageHandler(
        Filters.text & (Filters.entity(MessageEntity.URL) | Filters.entity(MessageEntity.TEXT_LINK)),youtube_handler, run_async=True))

    dispatcher.add_error_handler(error_handler)

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()