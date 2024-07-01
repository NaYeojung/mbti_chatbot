import json
import requests
import streamlit as st 
import pandas as pd

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
        api_key='NTA0MjU2MWZlZTcxNDJiY6o7O0mMGUuTEHU6yLaRpv/2IkicvAMe/Pab0BKS5gW8',
        api_key_primary_val='gIAB8vXgHEn5ZwAEgBHbnj6qZVa45KdMxz85pTjT',
        request_id='85faed7a-d6fc-413b-8858-513aeaebe9f1'
    )
    
    st.title('회의록 내용 요약')
  
    df=pd.read_csv('minutes_test.csv', encoding='utf-8')
    display=df.iloc[:,2]
    options=df.iloc[:,3]
    minutes=st.selectbox('조회할 회의록', options, format_func=lambda x: display[df.iloc[:, 3] == x].values[0])
    if minutes:
        preset_text = [{"role":"system","content":"- 데이터를 해독하고, 파싱하여 핵심 내용을 추출합니다."},{"role":"user","content":minutes}]

        request_data = {
            'messages': preset_text,
            'topP': 0.6,
            'topK': 0,
            'maxTokens': 4096,
            'temperature': 0.1,
            'repeatPenalty': 1.2,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }

        result = completion_executor.execute(request_data)
        r=json.loads(result[-4][5:])
        st.chat_message('assistant').write(r['message']['content'])
