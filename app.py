import os
import streamlit as st
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

# .envファイルから環境変数を読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="AI Chat Assistant",
    layout="wide"
)

# Azure AI の設定を初期化
@st.cache_resource
def init_azure_client():
    """Azure AI クライアントを初期化"""
    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    model_name = os.getenv("AZURE_AI_MODEL_NAME")
    api_key = os.getenv("AZURE_AI_API_KEY")
    
    if not all([endpoint, model_name, api_key]):
        st.error("必要な環境変数が設定されていません。.envファイルを確認してください。")
        st.stop()
    
    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        api_version="2024-05-01-preview"
    )
    
    return client, model_name

# LLMにメッセージを送信
def get_llm_response(client, model_name, system_prompt, user_message, chat_history):
    """LLMから応答を取得"""
    try:
        # メッセージリストを構築（Systemメッセージは最初に配置）
        messages = [SystemMessage(content=system_prompt)]
        
        # チャット履歴を追加（UserMessage と AssistantMessage の順番で）
        for chat in chat_history:
            messages.append(UserMessage(content=chat["user"]))
            messages.append(AssistantMessage(content=chat["assistant"]))
        
        # 現在のユーザーメッセージを追加
        messages.append(UserMessage(content=user_message))
        
        response = client.complete(
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            model=model_name
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# メイン関数
def main():
    # Azure クライアントを初期化
    client, model_name = init_azure_client()
    
    # タイトル
    st.title("AI Chat Assistant")
    
    # デフォルトのシステムプロンプト
    default_prompt = """あなたは親切で知識豊富なAIアシスタントです。
    ユーザーの質問に対して、正確で分かりやすい回答を提供してください。
    日本語で回答してください。 """ 

#    default_prompt = """あなたは親切なAIたぬきです。
#ユーザーの質問に対して、正確で分かりやすい回答を提供してください。
#日本語で回答してください。語尾に「ぽん」とつけます。"""
    
    # サイドバーでシステムプロンプトを設定
    with st.sidebar:
        st.header("設定の確認")
        
        # システムプロンプトの設定
        system_prompt = st.text_area(
            "設定されているシステムプロンプト",
            value=default_prompt,
            height=150,
        )
        
        # チャット履歴をクリアするボタン
        if st.button("チャット履歴をクリア", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # 設定情報を表示
        st.divider()
        st.caption(f"**Model:** {model_name}")
        st.caption(f"**Endpoint:** {os.getenv('AZURE_AI_ENDPOINT', 'Not set')}")
    
    # セッション状態を初期化
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # チャット履歴を表示
    chat_container = st.container()
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            # ユーザーメッセージ
            with st.chat_message("user"):
                st.write(chat["user"])
            
            # アシスタントメッセージ
            with st.chat_message("assistant"):
                st.write(chat["assistant"])
    
    # チャット入力
    if user_input := st.chat_input("メッセージを入力してください..."):
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.write(user_input)
        
        # アシスタントの応答を生成・表示
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                assistant_response = get_llm_response(
                    client, model_name, system_prompt, user_input, st.session_state.chat_history
                )
            st.write(assistant_response)
        
        # チャット履歴に追加
        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # ページを更新してチャット履歴を反映
        st.rerun()

if __name__ == "__main__":
    main()