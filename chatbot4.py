#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv()


# In[2]:


from langchain_groq import ChatGroq


# In[3]:


model=ChatGroq(model="llama-3.3-70b-versatile")


# In[4]:


##chat memory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


# In[5]:


store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]


# In[6]:


## Rag
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# In[7]:


from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# In[8]:


##Example PDF
pdf_path="leave_policy.pdf"
loader=PyPDFLoader(pdf_path)
documents=loader.load()


# In[9]:


##Chunking
text_splitter=RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=80)
chunks=text_splitter.split_documents(documents=documents)


# In[10]:


#create Vector DB
vector_db=Chroma.from_documents(chunks,embedding_model,persist_directory="./new_leave_policy_db")


# In[11]:


retriever=vector_db.as_retriever(search_type="similarity",search_kwargs={"k":10})


# In[12]:


#prompt template with chat chat memory+context
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",
 "You are an HR policy assistant.\n"
 "You MUST answer strictly using the provided Leave Policy Context.\n"
 "If the context mentions weeks, convert them into days.\n"
 "If the answer is present, you MUST answer it.\n"
 "If not present, say 'Not specified in the policy document.'"),
        MessagesPlaceholder(variable_name="messages"),
                            ("human",
 "Leave Policy Context:\n{context}\n\n"
 "Question:\n{question}\n\n"
 "Instructions:\n"
 "- Answer ONLY from the context\n"
 "- Quote the relevant sentence\n"
 "- Perform unit conversion if needed")
    
    ]
)


# In[13]:


def retrieve_context(input_dict):
    docs = retriever.invoke(input_dict["question"])

    print("🔍 Retrieved docs count:", len(docs))
    for i, d in enumerate(docs):
        print(f"\n--- DOC {i} ---\n", d.page_content[:500])

    return "\n\n".join([d.page_content for d in docs])


# In[ ]:





# In[14]:


from langchain_core.runnables import RunnableParallel ,RunnablePassthrough

rag_chain=(
    RunnableParallel({
        "context":retrieve_context,
        "question":lambda x:x["question"],
        "messages": lambda x:x["messages"]
    })
    |prompt
    |model
)


# In[15]:


with_message_history=RunnableWithMessageHistory(
    rag_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="messages"
)


# In[16]:


config = {"configurable": {"session_id": "test_session_001"}}


# In[20]:


user_query="baby's birth?"


# In[21]:


response=with_message_history.invoke({"question":user_query},
                                     config=config)


# In[22]:


print(response.content)


# In[258]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




