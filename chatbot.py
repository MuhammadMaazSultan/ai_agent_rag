import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import DATA_DIR, STORAGE_DIR, CHUNK_SIZE, RETRIEVAL_TOP_K, PROMPT_TEMPLATE

class HistoryChatBotRag:
    def __init__(self, model_name, embedding_model, prompt_template):

        print('---Connection Building---')
        self.llm = Ollama(model = model_name, temperature=0.7)
        self.embeddings = OllamaEmbeddings(model =  embedding_model)

        print('---2. Creating Vectors---')
        vectorstore = self.get_or_create_embedding()

        print('--3. RAG Setup---')
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_TOP_K}
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )
    def get_or_create_embedding(self):
        '''Create vectors if not
        fetch vectors from storage if available
        ;returns '''

        print('---Checking for existing embeddings---')
        faise_index_file = os.path.join(STORAGE_DIR, 'faiss_index')
        faise_index_meta = os.path.join(STORAGE_DIR, 'faiss.pkl')
        
        if os.path.exists(faise_index_file) and os.path.exists(faise_index_meta):
            return FAISS.load_local(
                STORAGE_DIR,
                self.embeddings,
                index_name = 'faiss_index',
            )
        if not any(os.listdir(DATA_DIR)):
            return FAISS.from_documents([Document(page_content="No data found. Please add documents to the data directory.")], self.embeddings)
        print('---Creating new embeddings from data directory---')
        loader = DirectoryLoader(DATA_DIR, 
                                 glob="**/*", 
                                 loader_cls=PyPDFLoader, 
                                 silent_errors=True)
        print('---Loading Documents---')
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = int(0.1 * CHUNK_SIZE)
        )
        print('---Splitting Documents---')
        texts = text_splitter.split_documents(document)
        vectorstore = FAISS.from_documents(
            texts,
            self.embeddings)
        vectorstore.save_local(STORAGE_DIR, index_name = 'faiss_index')
        return vectorstore

    def get_response(self, user_input):

        '''To take user input and and send to ai agent and give the answer back
        :param user_input:
        :return: '''
        result = self.chain.invoke({'query': user_input})
        sources = []
        for doc in result.get('source_documents', []):
            source_path = doc.metadata.get('source','Unknown Source')
            filename = os.path.basename(source_path)
            sources.append(f'{filename}, (page:{doc.metadata.get("page", "N/A")})')

        return { 
            'answer': result['result'],
            'sources': list(set(sources))
        }
    def clear_memory(self):

        '''
        Clear the conversation memory
        or reset the conversation
        '''
        self.memory.clear()
