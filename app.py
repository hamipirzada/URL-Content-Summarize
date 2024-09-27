import validators, streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
import asyncio

st.set_page_config(page_title="Langchain: Summarize text From YouTube or Website", page_icon="🦜️")
st.title("🦜️Summarize Text From YouTube or Website")
st.subheader = ("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL:", label_visibility="visible")

# Gemma model using Groq API Key
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 1000 words:
content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

async def fetch_documents(url):
    if "youtube.com" in url:
        try:
            # Attempt to fetch an English transcript
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language="en")
        except Exception as e:
            # Fall back to Hindi if English transcript is unavailable
            st.error(f"Could not retrieve an English transcript. Trying Hindi...")
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language="hi")
    else:
        loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
        })
    documents = loader.load()
    return documents

if st.button("Summarize Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please enter the information to get started.")
    elif not validators.url(generic_url):
        st.error("Invalid URL. Please enter a valid URL.")
    else:
        try:
            with st.spinner("Fetching content..."):
                documents = asyncio.run(fetch_documents(generic_url))

                if documents:
                    # Chain for summarization
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

                    # Run the summarization chain with the documents as input
                    summary = chain.run(input_documents=documents)

                    st.success(summary)
                else:
                    st.error("Failed to load content.")
        except Exception as e:
            st.exception(f"Exception: {e}")
