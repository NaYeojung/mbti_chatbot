# -*- coding: utf-8 -*-
import json
import requests
import streamlit as st 


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }
        ret = []
        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    ret.append(line.decode("utf-8"))
        return ret

if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key='NTA0MjU2MWZlZTcxNDJiY0Yx8UfE6zpKxwm+Gyd0YaxQin2S7mTJ6NKo02d0UiDK',
        api_key_primary_val='dlXNq6VvRKLSk01Bi609VBXeRmY5qQiepkkaoXo6',
        request_id='394bed83-dc4d-496f-968a-d9454bb33d2b'
    )
    
    st.title('MBTI 백과사전')
    # 대화들 쭉 나오게 만드는 코드
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    question = st.chat_input("질문을 입력하세요")
    
    if question:
        preset_text = [{"role":"system","content":"- MBTI에 대한 지식을 기반으로, MBTI 질문에 답해보세요.\n\n질문: ESFJ는 문제에 봉착했을때 어떻게 대응하는가?\n답: 현실적인 해결 방법을 찾기 위해 노력합니다.\n###\n질문: ISFJ는 연인에게 어떻게 대하는 편인가?\n답: 섬세하고 다정하게 케어해주는 편입니다.\n####\n질문: INTP는 사람들이 많은 곳에 가면 어떻게 행동하는가?\n답: 주변의 상황을 파악하기 위해 관찰하는 편입니다.\n###\n질문: ESFJ는 충동적인 선택을 많이 하는 편인가?\n답: 아니다. 계획적으로 움직이는 편입니다."},{"role":"user","content":"ESTP는 학업에 어떤 자세로 임하는가?"},{"role":"assistant","content":"ESTP는 호기심이 많고 직접 경험하며 배우는 것을 선호하여 학업에 적극적으로 참여할 수 있습니다. 이들은 새로운 아이디어와 개념을 빠르게 습득하며, 다양한 문제를 해결하는 데 능숙합니다. 또한, 자신이 관심 있는 분야에서는 열정적으로 공부하며, 이를 통해 성취감을 느끼는 경향이 있습니다. 하지만, ESTP는 세부적인 내용이나 지루한 작업에는 집중력이 떨어질 수 있으므로, 체계적인 계획을 세우고 일정을 관리하는 것이 중요합니다."},{"role":"user","content":question}]

        request_data = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 512,
            'temperature': 0.5,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }
        # 질문을 보내는 부분
        result = completion_executor.execute(request_data)
        # 결과를 출력하는 부분
        r = json.loads(result[-4][5:])
        st.chat_message('user').write(question)
        st.chat_message('assistant').write(r['message']['content'])
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": r['message']['content']})
        