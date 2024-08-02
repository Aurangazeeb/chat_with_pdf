    import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import pathlib

def load_document(file):
    filename_without_ext, extension = os.path.splitext(file)

    if extension == '.pdf':
    #     using a transform loader i.e native form to langchain doc form
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
        
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)

    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)

    elif extension == '.odt':
        from langchain_community.document_loaders.odt import UnstructuredODTLoader
        loader = UnstructuredODTLoader(file)

    else:
        print('Document format not supported')
        return None
    
    data = loader.load()
    return data


def chunk_data(data, chunk_size = 256, chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    doc_chunks = textsplitter.split_documents(data)
    return doc_chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents=chunks, embedding= embeddings)
    return vector_store


def ask_and_get_answer(vector_store, query, topk = 5):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model = 'gpt-3.5-turbo', temperature = 0.8)
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': topk})

    chain = RetrievalQA.from_chain_type(llm = llm, chain_type = 'stuff', retriever= retriever)
    
    # answer = chain.run(query)
    answer = chain.invoke(query)
    return answer['result']

def calculate_embedding_cost(chunks):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(doc_chunk.page_content)) for doc_chunk in chunks])
    return total_tokens, 84 * total_tokens/1e6 * 0.02

def ask_and_get_answer_from_memory(query, vector_store, topk):
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    system_template = r'''
    Use the following pieces of context to answer the user's question. 
    If you can't find the answer from the provided context, just reply "I don't know"
    -------------------------
    Context : ```{context}```
    '''
    user_template = '''
    Question : ```{question}```
    Chat History : ```{chat_history}```
    '''

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ]

    # create chat prompt template
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature = 0)
    # vector_store = load_embeddings_chroma()
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k': topk})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    crc = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        memory = memory,
        chain_type='stuff',
        combine_docs_chain_kwargs={'prompt' : qa_prompt}
    )
    answer = crc.invoke({'question' : query})
    return answer['answer']

def submit():
    st.session_state.user_query = st.session_state.widget
    st.session_state.widget = ''

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    
def update_extract_dir():
    destdir_prefix = os.path.abspath(os.getcwd())
    st.session_state.destdir = f'{destdir_prefix}/{st.session_state.extractdir}'

# def update_code_language():
#     st.session_state.codelanguage = st.session_state.codelang

# def update_file_suffix():
#     st.session_state.filesuffix = st.session_state.file_suffix

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(os.environ['ENV_PATH'], override= True)
    answer = None
    APPHOME = pathlib.Path(os.environ['CHATAPP']).parent
    SOURCEHOME = pathlib.Path(os.environ["SOURCE_HOME"])
    st.image(f'{APPHOME}/langchain.jpg')
    st.subheader("LLM Question Answering App 🤖")
    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file : ', type= ['pdf', 'docx', 'txt', 'odt'])
        
        chunk_size = st.number_input("Chunk size : ", min_value= 100, max_value= 2048, value=256,on_change=clear_history)
        topk = st.number_input('K (top chunks to use as context) : ', min_value= 1, max_value= 20, value=5, on_change= clear_history)
        st.text_input(label='Type you extract dir : ', value="./",key='extractdir', on_change=update_extract_dir)

        if "destdir" not in st.session_state:
            st.session_state.destdir= "./"
        st.write('Destdir : ', st.session_state.destdir)
        # st.text_input(label='Prefer code language : ', key='codelang', on_change=update_code_language)
        # st.text_input(label='Type your file suffix : ', key='file_suffix', on_change=update_file_suffix)
        
        
        add_data = st.button('Add Data', on_click=clear_history)

        # if 'chunksize' not in st.session_state:
        #     st.session_state.chunksize = chunk_size
        # if 'topk' not in st.session_state:
        #     st.session_state.topk = topk

        if uploaded_file and add_data:
            st.write(f'Abs path of uploaded file : {os.path.abspath(uploaded_file.name)}')
            st.session_state.filename = uploaded_file.name
            with st.spinner("Reading, chunking and embedding file ..."):
                # bytes_data = uploaded_file.read()
                # file_name = os.path.join('./uploaded_docs/', uploaded_file.name)
                # with open(file_name, 'wb') as f:
                #     f.write(bytes_data)
                uploadfile_prefix = os.path.abspath(os.getcwd())
                file_name = f"{uploadfile_prefix}/{SOURCEHOME}/{uploaded_file.name}"
                # if 'filename' not in st.session_state:
                #     st.session_state.filename = file_name


                # st.write(f'Current file : {st.session_state.filename}')
                # st.write(f'Newly uploaded file : {uploaded_file.name}')
                # st.session_state.new_filename = uploaded_file.name
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size= chunk_size)
                st.write(f'Chunk size : {chunk_size}, Chunks : {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost : INR {embedding_cost:.4f}')
                
                # uploaded_file_mgr = st.session_state.uploaded_file_mgr
                # if a new file is uploaded
                # if (st.session_state.filename != uploaded_file.name):
                vector_store = create_embeddings(chunks=chunks)
                st.session_state.vs = vector_store
                # st.session_state.filename = uploaded_file.name

                # if st.session_state.chunksize != chunk_size:

                # else:
                #     st.write('Vector store already exists. Not recreating embedddings')

                
                    # st.session_state.vs = ''
                    

                st.success('File uploaded, chunked and embedded successfully... 🔥')

    if 'user_query' not in st.session_state:
        st.session_state.user_query = ''


    st.text_input("Ask a question about the content of your file : ", key='widget', on_change=submit)
    query = st.session_state.user_query
    del st.session_state.user_query
    if query:
        # # if new file - update vector_store session key
        # if st.session_state.new_filename != st.session_state.filename:
        #     vector_store = create_embeddings(chunks=chunks)
        #     st.session_state.vs = vector_store
        # else:
        #     vector_store = st.session_state.vs

        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f'K : {topk}')
            # answer = ask_and_get_answer(vector_store=vector_store, query=query, topk=topk)
            # query = f'{query} in this document'
            answer = ask_and_get_answer_from_memory(query=query, vector_store=vector_store, topk=topk)
            # st.text_area('LLM Answer : ', value= answer)
            st.markdown('__Question__')
            st.markdown(query)
            st.markdown('__LLM Answer__')#, value= answer)
            st.markdown(answer)

    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''
    if answer:
        value = f'\nQuestion : {query} \n\nAnswer :\n{answer}\n'
        complete_name = pathlib.Path(st.session_state.filename).stem
        # complete_name = f'{complete_name}_{st.session_state.filesuffix}'
        # st.write(f'Writing to {st.session_state.destdir}/{complete_name}_{st.session_state.filesuffix}.txt')
        os.makedirs(st.session_state.destdir, exist_ok=True)
        with open(f'{st.session_state.destdir}/{complete_name}.txt', 'a') as summaryfile:
            summaryfile.write(value)
            summaryfile.write(f'\n{"-" * 100}\n')
        st.session_state.history = f'{st.session_state.history}\n\n{"-" * 100}\n{value}'
        h = st.session_state.history
        st.text_area(label= 'Chat History', value=h, key='history', height= 400)
        answer = None
