from accelerate import Accelerator
import os
import torch
import textwrap
import glob
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TextSplitter
import matplotlib.pyplot as plt
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from pypdf import PdfWriter
from langchain_core.prompts import PromptTemplate

from transformers import pipeline

#from langchain_community.chains import ConversationalRetrievalChain

from nltk.corpus import stopwords

PATH_DOC = "./SentimentAnalysis-and-OpinionMining.pdf"
CHROMA_PATH = "chroma"
TEMP_PATH = "temp"
PROMPT_TEMPLATE = """{context}Provide solution to: {question}"""
def provide_answer(retriever, query):

    results = retriever.invoke(f"{query}")

    if len(results) == 0:
        return "I don't know that information, sorry..."
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map='auto', torch_dtype = torch.float16,)


    #for MACOS
    accelerate = Accelerator()
    model = accelerate.prepare(model)
    #model.to(device2)

    results = retriever.get_relevant_documents(query)

    context_text = ""
    print(context_text)


    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
    )
    prompt = prompt_template.format(context=context_text, question= query)
    
   
    
    hf_model = pipeline("text2text-generation", model= model, tokenizer = tokenizer, max_length=512, temperature=0, top_p = 0.95, repetition_penalty = 1.15,)  # device=-1 ensures CPU usage

    # Wrap the Hugging Face pipeline in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_model)
    retrievalQA = RetrievalQA.from_chain_type(llm=llm, 
    chain_type ="map_reduce", retriever = retriever, return_source_documents=True,) #chain_type_kwargs={"prompt": prompt_template})

    #print(context_text)
    response = retrievalQA(query)
    print(response)
    getResponse = response['result']
    delimiters = ["\n"]
 
    for delimiter in delimiters:
       getResponse = " ".join(getResponse.split(delimiter))
    lines = getResponse.split()

    wrapp = [textwrap.fill(line, width=110) for line in lines]
    wrapp = ' '.join(wrapp)
    return f"Response: {wrapp}"

def main():


    text_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    separators=["\n"]
    )

    documents = load_documents()
    chunks = text_split.split_documents(documents)
    
    if(os.path.exists(CHROMA_PATH)):
        shutil.rmtree(CHROMA_PATH)
    print(len(chunks))

    ##Optimization done for MacOS
    model_name = "hkunlp/instructor-large"
    model_kwargs = {'device': 'mps'}
    encode_kwargs = {'normalize_embeddings': True}


    hf = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
    )

    db = Chroma.from_documents(chunks, hf, persist_directory=CHROMA_PATH)

    #used the persist method to create the database
    db.persist()
    db_elements = db.as_retriever(search_kwargs={"k":3, "score_threshold": 0.72}, search_type="similarity_score_threshold")

    
    while(1):
        query = input("Enter your question : ")

        if query=="exit":
            break
        answer = provide_answer(db_elements, query)
        print(answer)

        
     

def merge_pdf():
    pdf_path = "./pdf_files"
    append_pdf = PdfWriter()
    our_files = glob.glob('*.{}'.format('pdf'), root_dir=pdf_path)

    for pdf in our_files:
        append_pdf.append(f"{pdf_path}/{pdf}")

    append_pdf.write(f"{TEMP_PATH}.pdf")
    


def load_documents():
    merge_pdf()
    document_loader = PyPDFLoader("temp.pdf")
    return document_loader.load()


if __name__=="__main__":
    main()
    os.remove("temp.pdf") 