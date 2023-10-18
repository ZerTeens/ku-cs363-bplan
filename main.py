from keras.models import load_model
from flask import Flask, request, abort
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pythainlp.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from conf import *
from service import Spreadsheet
import time
import service
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import re
import pytz
import User


scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    "id-credentials-for-spreadsheet.json", scope)
client = gspread.authorize(credentials)

spreadsheet = client.open("ku-cs363-bplan-database")

worksheet = spreadsheet.get_worksheet(0)


model = load_model('model.keras')
thailand = pytz.timezone('Asia/Bangkok')


app = Flask(__name__)

conn = Spreadsheet(
    "1RO_bhae8ID7yrdFCysKirNi46123JgciCsmQ67uFSyE", "datasets!A:B")
sheet = conn.read()

data = pd.DataFrame(sheet[1:], columns=sheet[0])

X = data.text

X_tokens = X.apply(word_tokenize, keep_whitespace=False)

maxlen = X_tokens.apply(len).max()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_tokens)


global timeFormat, user_data

user_data = {}

timeFormat = "%m/%d/%Y, %H:%M:%S"
time_pattern = r'^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$'


@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    if request.method == 'POST':
        payload = request.json
        replyToken = payload['events'][0]['replyToken']
        message = payload['events'][0]['message']['text']

        events = payload['events']
        user_id = None

        for event in events:  # get ID form message
            if event['type'] == 'message':
                user_id = event['source']['userId']
            if user_id not in user_data:
                user_data[user_id] = User.User(user_id)
        user = user_data[user_id]

    
        if user.state == 1:
            user.title = message
            user.state = -1
            
            
        elif user.state == 2:
            user.startTime = convert_speech_to_time(message)
            if (user.startTime != None):
                user.startTime = user.startTime.strftime(timeFormat)
                user.state = -1
            else:
                user.startTime = ""
        elif user.state == 3:
            user.endTime = convert_speech_to_time(message)
            if (user.endTime != None):
                user.endTime = user.endTime.strftime(timeFormat)
                user.state = -1
            else:
                user.endTime = ""
        elif user.state == 4:
            user.description = message
            user.state = -1

        if user.mode == -1:
            message = word_tokenize(message, keep_whitespace=False)
            message = tokenizer.texts_to_sequences([message])
            message = pad_sequences(message, maxlen=maxlen, padding="post")
            y_pred = model.predict(message)
            y = np.argmax(y_pred)

            user.mode = y

            if user.mode == 0 or user.mode == 1 or user.mode == 2:
                user.title = ""
                user.startTime = ""
                user.endTime = ""
                user.description = ""
                # load default value

        if user.mode == 0:
            if user.title == "":
                replyMessage(
                    replyToken, 'คุณต้องการเพิ่มกิจกรรมอะไร', channelAccessToken)
                user.state = 1
            elif user.startTime == "":
                replyMessage(replyToken, 'ขอทราบเวลาเริ่ม', channelAccessToken)
                user.state = 2
            elif user.endTime == "":
                replyMessage(replyToken, 'ขอทราบเวลาจบ', channelAccessToken)
                user.state = 3
            elif user.description == "":
                replyMessage(
                    replyToken, f'รายละเอียดเพิ่มเติมเกี่ยวกับ {user.title}', channelAccessToken)
                user.state = 4
            else:
                if user_id is not None:
                    replyMes = f'เพิ่มกิจกรรม {user.title} ในเวลา {user.startTime} ถึง {user.endTime} ในตารางเรียบร้อยแล้ว\n\n'
                    worksheet.append_row(
                        [user_id, user.title, user.startTime, user.endTime, user.description])
                    user.mode = -1
                    replyMes += "คุณต้องการให้ฉันช่วยอะไรอีกไหม"
                    replyMessage(replyToken, replyMes, channelAccessToken)
        elif user.mode == 1:
            if user.title == "":
                replyMes = showPlan(user_id) + '\nคุณต้องการแก้ไขกิจกรรมอะไร'
                replyMessage(
                    replyToken, replyMes, channelAccessToken)
                user.state = 1
            elif user.startTime == "":
                replyMessage(replyToken, 'ขอทราบเวลาเริ่ม', channelAccessToken)
                user.state = 2
            elif user.endTime == "":
                replyMessage(replyToken, 'ขอทราบเวลาจบ', channelAccessToken)
                user.state = 3
            elif user.description == "":
                replyMessage(
                    replyToken, f'รายละเอียดเพิ่มเติมเกี่ยวกับ {user.title}', channelAccessToken)
                user.state = 4
            else:
                if user_id is not None:
                    df = pd.DataFrame(data=worksheet.get_all_records())
                    if not df.empty:
                        filtered_df = df[(df["userid"] == user_id) & (
                            df["title"] == user.title)]

                        if not filtered_df.empty:
                            index_to_delete = filtered_df.index[0].item()
                            worksheet.delete_rows(
                                index_to_delete + 2, index_to_delete + 3)

                            worksheet.append_row(
                                [user_id, user.title, user.startTime, user.endTime, user.description])

                            replyMes = f'แก้ไขกิจกรรม {user.title} ในเวลา {user.startTime} ถึง {user.endTime} ในตารางเรียบร้อยแล้ว\n\n'
                        else:
                            replyMes = "ไม่พบกิจกรรมในตาราง"
                    else:
                        replyMes = f"ยังไม่มีกิจกรรมในตาราง โปรดเพิ่มกิจกรรมก่อนค่ะ\n"
                    replyMes += "คุณต้องการให้ฉันช่วยอะไรอีกไหม"
                    user.mode = -1
                    replyMessage(replyToken, replyMes, channelAccessToken)
        elif user.mode == 2:
            if user.title == "":
                replyMes = showPlan(user_id) + '\nคุณต้องการลบกิจกรรมอะไร'
                replyMessage(replyToken, replyMes, channelAccessToken)
                user.state = 1
            else:
                if user_id is not None:
                    df = pd.DataFrame(data=worksheet.get_all_records())
                    if not df.empty:
                        filtered_df = df[(df["userid"] == user_id) & (
                            df["title"] == user.title)]

                        if not filtered_df.empty:
                            index_to_delete = filtered_df.index[0].item()
                            worksheet.delete_rows(
                                index_to_delete + 2, index_to_delete + 3)
                            replyMes = f'ลบกิจกรรม {user.title} ในตารางเรียบร้อยแล้ว\n\n'
                        else:
                            replyMes = "ไม่พบกิจกรรมในตาราง\n"
                    else:
                        replyMes = f"ยังไม่มีกิจกรรมในตาราง โปรดเพิ่มกิจกรรมก่อนค่ะ\n"
                    replyMes += "คุณต้องการให้ฉันช่วยอะไรอีกไหม"
                    user.mode = -1
                    replyMessage(replyToken, replyMes, channelAccessToken)
        elif user.mode == 3:
            conn = service.Spreadsheet(
                "1RO_bhae8ID7yrdFCysKirNi46123JgciCsmQ67uFSyE", "datasets!G:H")
            sheet = conn.read()
            ans = pd.DataFrame(sheet[1:], columns=sheet[0])
            replyMessage(
                replyToken, f'{ans.loc[0, "ans"]}', channelAccessToken)
            user.mode = -1
            replyMessage(
                replyToken, 'คุณต้องการให้ฉันช่วยอะไรอีกไหม', channelAccessToken)
        elif user.mode == 4:
            conn = service.Spreadsheet(
                "1RO_bhae8ID7yrdFCysKirNi46123JgciCsmQ67uFSyE", "datasets!G:H")
            sheet = conn.read()
            ans = pd.DataFrame(sheet[1:], columns=sheet[0])
            replyMessage(
                replyToken, f'{ans.loc[1, "ans"]}', channelAccessToken)
            user.mode = -1
            replyMessage(
                replyToken, 'คุณต้องการให้ฉันช่วยอะไรอีกไหม', channelAccessToken)
        elif user.mode == 5:
            replyMes = showPlan(user_id)
            user.mode = -1
            replyMes += "\nคุณต้องการให้ฉันช่วยอะไรอีกไหม"
            replyMessage(replyToken, replyMes, channelAccessToken)

        return request.json, 200
    else:
        abort(400)


def replyMessage(replyToken, text, accessToken):
    LINE_API = 'https://api.line.me/v2/bot/message/reply/'

    authorization = 'Bearer {}'.format(accessToken)

    headers = {
        'Content-Type': 'application/json; char=UTF-8',
        'Authorization': authorization
    }

    data = {
        'replyToken': replyToken,
        'messages': [
            {
                'type': 'text',
                'text': text
            }
        ]
    }

    data = json.dumps(data)
    requests.post(LINE_API, headers=headers, data=data)


def convert_speech_to_time(text):
    if re.search(time_pattern, text):
        current_date = datetime.now(thailand)
        formatted_text = current_date.strftime(
            '%m/%d/%Y') + ", " + text + ":00"
        date = datetime.strptime(formatted_text, timeFormat)
        return date
    elif re.search(r'^\d{2}/\d{2}/\d{4}, \d{2}:\d{2}$', text):
        date = datetime.strptime(text+":00", timeFormat)
        return date
    else:
        now = datetime.now(thailand)

        if 'now' in text or 'ตอนนี้' in text:
            return now

        if 'tomorrow' in text:
            return now + timedelta(days=1)

        if 'next' in text:
            parts = text.split()
            index = parts.index('next')
            value = int(parts[index + 1])

            if 'days' in text:
                return now + timedelta(days=value)
            elif 'months' in text:
                return now + timedelta(days=30 * value)
            elif 'years' in text:
                return now + timedelta(days=365 * value)

        if 'อีก' in text:
            parts = text.split()
            value = int(parts[1])

            if 'วัน' in text:
                return now + timedelta(days=value)
            elif 'เดือน' in text:
                return now + timedelta(days=30 * value)
            elif 'ปี' in text:
                return now + timedelta(days=365 * value)
        if 'พรุ่งนี้' in text:
            return now + timedelta(days=1)
        if 'เดือนหน้า' in text:
            return now + timedelta(days=30)

        if 'ปีหน้า' in text:
            return now + timedelta(days=365)

    return None


def showPlan(user_id):
    df = pd.DataFrame(data=worksheet.get_all_records())
    if not df.empty:
        filtered_df = df[df["userid"] == user_id].reset_index()
        if not filtered_df.empty:
            filtered_df['start'] = pd.to_datetime(filtered_df['start'])
            filtered_df['start'] = filtered_df['start'].dt.strftime(timeFormat)
            filtered_df = filtered_df.sort_values(by='start')
            replyMes = "คุณมี\n\n"
            for i, row in filtered_df.iterrows():
                replyMes += f'{row["title"]} ({row["description"]}) ในเวลา {row["start"]} ถึง {row["end"]}\n'
            return replyMes
    return "โปรดเพิ่มกิจกรรมก่อนค่ะ"
    