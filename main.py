import asyncio
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from starlette import status
import uvicorn
from pydantic import BaseModel, Field
# from moderation import criteria_alert_from_prompt
from typing import Literal
import os
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from prompt import tori_system_prompt, summary_system_prompt


API_KEY = os.getenv('API_KEY')


client = OpenAI(
    api_key=API_KEY,
    project='proj_YA4wA5gFbCTSd8ImZ1UapNJN'
)
app = FastAPI()


# 로깅 설정 (시간 제거)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - %(message)s"
)
logger = logging.getLogger("fastapi-logger")

# OpenAI 및 HTTP 라이브러리 로그 수준 조정
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)









class MessageModel(BaseModel):
    user_uid: str = Field(..., min_length=27, description="Input user's user's uid")
    user_chat_date: str = Field(..., min_length=9, description="Input user's chat_date")
    new_user_message: str = Field(..., min_length=1, description="Input user prompt")

    model_config = {#docs에 보이는 예시
        "json_schema_extra": {
            "example": {
                "user_uid": "T6lA4m8FCUWiqk2EWqo9JFtvSfW2",
                "user_chat_date": "2024-07-07",
                "new_user_message": "오늘은 좋은 하루였어",
            }
        }
    }

class ToriResponseModel(BaseModel):
    tori_message: str





async def getPreviousChat(user_uid:str, user_chat_date:str, new_user_message:str) -> list:
    return await asyncio.to_thread(_getPreviousChat_sync, user_uid, user_chat_date, new_user_message)

def _getPreviousChat_sync(user_uid:str, user_chat_date:str, new_user_message:str) -> list:
    if not firebase_admin._apps:
        cred = credentials.Certificate("/etc/secrets/dotori-fd1b0-firebase-adminsdk-zzxxd-fb0e07e05e.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Get all top-level collections
    chat_doc_ref = db.collection(user_uid).document('chat')
    chat_doc = chat_doc_ref.get()
    
    if chat_doc.exists:
        # Fetch and store the data
        chat_info = chat_doc.to_dict().get(user_chat_date, '')
        
        start_key = "채팅_10001"
        # 결과를 저장할 리스트
        chat_sequence = [        
            {
                "role": "system",
                "content": f"{tori_system_prompt}"
            },
            {
                "role": "user",
                "content": "(대화시작)"
            },
            {
                "role": "assistant",
                "content": "오늘 뭐가 가장 기억에 남았어?"
            }
        ]

        # 현재 키를 초기값으로 설정
        current_key = start_key

        # 순서대로 채팅 내용을 리스트에 추가
        while current_key in chat_info:
            chat_sequence.append(chat_info[current_key])
            
            # 다음 키 생성
            next_number = int(current_key.split('_')[1]) + 1
            current_key = f"채팅_{next_number:05}"


        if chat_sequence[-1].get('role', '') != 'user':
            chat_sequence.append(
                {
                "role": "user",
                "content": f"{new_user_message}"
                }
            )
        
        return chat_sequence


@app.post(
    "/retrieve_tori_message",
    status_code=status.HTTP_201_CREATED,
    response_model=ToriResponseModel,
    responses={ #docs에 보이는 예시
        501: {
            "description": "Failed to fecth data from Firebase",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to fetch data from firebase console due to, <Error Explanation>"
                    }
                }
            },
        },
    },
)
async def getMessageFromTori(model: MessageModel) -> ToriResponseModel:
    try:
        previous_chat_list = await getPreviousChat(user_uid = model.user_uid, user_chat_date = model.user_chat_date, new_user_message = model.new_user_message) 
        logger.info(f'Fetch Successful from firebase')
    except Exception as e:
        logger.error(f'Failed to fetch data (userUID = {model.user_uid}, chat_date = {model.user_chat_date}) from firebase console due to {e}')
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f'Failed to fetch data from firebase console due to {e}'
        )
     
    max_try = 3
    current_try = 0

    while True:
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model="ft:gpt-4o-2024-08-06:personal:toriforest002:ATRZ3Q78",
                temperature = 0.21,
                messages=previous_chat_list,
            )
            response = completion.choices[0].message.content
            logger.info(f'Response succesfully generated, response = {response}, userUID = {model.user_uid}, chat_date = {model.user_chat_date}')

            return ToriResponseModel(tori_message=response)

        except Exception as e:
            logger.error(f'The conversation of (userUID = {model.user_uid}, chat_date = {model.user_chat_date}) has not been responsed intentionally due to {e}')
            if current_try<max_try : 
                current_try += 1
                logger.info('Retrying...')
                continue
            else:
                logger.error(f'Terminal Error and returned Hard-Coded Message, (userUID = {model.user_uid}, chat_date = {model.user_chat_date})')
                return ToriResponseModel(tori_message='(토리가 잠깐 딴 생각을 했나봐요! 다시 한번 토리를 불러주세요 ㅜㅜ)')




















class SummaryLine(BaseModel):
    content: str = Field(..., description='One of the lines in the Summary')


class SummaryModel(BaseModel):
    dotori_emotion: Literal['very_happy', 'happy', 'neutral', 'sad', 'very sad', 'angry']
    summary: list[SummaryLine] = Field(description='Only summarize what has been spoken by the user')


class Prompt(BaseModel):
    role: Literal['system', 'user', 'assistant'] = Field(
        ...,
        description='Input the prompt\'s owner. Only type one of the followings [system, user, assistant]'
    )
    content: str = Field(
        ...,
        min_length=1,
        description='Input prompt generated by \'role\' '
    )


class ConversationModel(BaseModel):
    messages: list[Prompt]

    model_config = {
        "json_schema_extra": {
            'example': {
                "messages": [
                    {
                        "role": "user",
                        "content": "오늘 하루는 그냥 그랬던거 같아"
                    },
                    {
                        "role": "assistant",
                        "content": "그래? 무슨 일 있었는데..?"
                    },
                    {
                        "role": "user",
                        "content": "코딩하고 밥먹고 코딩하고의 반복이었어"
                    },
                    {
                        "role": "assistant",
                        "content": "그럼 이제 집에가서 뭐하게?"
                    },
                    {
                        "role": "user",
                        "content": "넷플릭스 보고 자야지, 오늘은 기록 그만할래"
                    },
                    {
                        "role": "assistant",
                        "content": "그래 잘자 오늘 푹 쉬고!"
                    }
                ]
            }
        }
    }


@app.post(
    '/get-summary',
    status_code=status.HTTP_201_CREATED,
    response_model=SummaryModel,
    responses={
        451: {
            "description": "Gpt's Refusal",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Refused by GPT for inappropriate Word use, <refusal body>"
                    }
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Unidentified Errors with Structured Outputs completion : <Exception Code>"
                    }
                }
            }
        }
    }
)
async def getSummaryFromGpt(conversationModel: ConversationModel) -> SummaryModel:
    max_try = 3
    current_try = 0

    while True:
        try:
            completion = await asyncio.to_thread(
                client.beta.chat.completions.parse,
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": f"{summary_system_prompt}"
                    },
                    {
                        "role": "user",
                        "content": f"{conversationModel.messages}"
                    },
                ],
                response_format=SummaryModel,
            )
            response = completion.choices[0].message

            if response.parsed:
                response_json = response.parsed.model_dump()
                logger.info('Summary successfully generated')
                return SummaryModel(
                    dotori_emotion=response_json['dotori_emotion'],
                    summary=response_json['summary']
                )
            elif response.refusal:
                logger.error(f'Refused by GPT for inappropriate Word use, {response.refusal}')
                raise HTTPException(
                    status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
                    detail=f'Refused by GPT for inappropriate Word use, {response.refusal}'
                )

        except Exception as e:
            logger.error(f'The thread has not been created due to Internal Server Error, {e}')
            if current_try<max_try :
                current_try += 1
                logger.info(f'Retrying...')
                continue
            else:
                logger.error(f'Terminal Error and raised HTTP exception 500')
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f'Unidentified Errors with Structured Outputs completion : {e}'
                )






if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
