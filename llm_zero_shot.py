import vllm
import datasets
from  huggingface_hub import login
import nltk
import vllm.sampling_params

# keeping my read token private
with open("token.txt", 'r') as f:
    hf_token = f.read()

# provides access to llama3.1 through my token
login(hf_token)

# 
#nltk.download('punkt',quiet=True)
#nltk.download('stopwords',quiet=True)
#nltk.download('wordnet',quiet=True)

# NLTK tools to help tokenize text
stop_words = set(nltk.stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()
tokenizer = nltk.tokenize.word_tokenize()
lemmatizer = nltk.WordNetLemmatizer()

# vllm llm function
def choose_model(name):
    # llama3.1 can take a max_model_len of 128k, max length of an item in quality is ~ 36k words
    # tensor_parallel_size = 1 means that only one gpu will be used - changing this based on gpu availability
    # gpu memory util: you want a high (close to 1) to make sure the model fits (not sure) but you may overload the GPU in doing so
    # this param depends on the GPU (and their number) used for training
    llm = vllm.LLM(model = name, gpu_memory_utilization = 0.9, max_model_len = 40000, tensor_parallel_size = 1) 
    return llm

def load_data(dataset_name):
    # trust_remote_code: target dataset may or may not include remote code to preprocess the dataset, not needed 
    # if your target dataset doesn't have it, but causes if no issues if left True. Only concern would be security when loading
    # datasets from untrusted sources (is my opinion)
    data = datasets.load_dataset(dataset_name, trust_remote_code = True)
    return data

def prompt_model(): #tbd later, using format 
    # you are an expert in _____,  (essentially description of the AI's role: e.g.: "You are a helpful assistant")
    system_prompt = "" 
    # what the model is actually being tasked with: different for each question in benchmark
    user_prompt = ""
    # here I usually specify format for the model
    instructions = ""
    return system_prompt, user_prompt, instructions

def generate(source_text, system_prompt, user_prompt, instructions, llm, nsampling = 1):
    t = 0.0 if nsampling == 1 else 0.8 # 0.8 should be probably sampled on validatoin set first, but it's a start
    #source where it will look for an answer, the prompts, nsampling is for self consistency (statistical decoding technique)
    request = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n**Source:** \"{source_text}\"\n\n **Instructions:** \n{instructions} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n["
    params = vllm.sampling_params(n = nsampling, temperature = t, seed = 42, max_tokens = 500) # 500 is max num of tokens in its answer
    # seed is random seed
    # may later include logprobs if of interest
    output = llm.generate(request, params, use_tqdm = True) # trying use_tqdm true for the first time, dunno how it will work
    return output[0].outputs[0].text #returns only the raw answer text




# processing functions - is this necessary actually? idk

def main():
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    dataset = "emozilla/quality"

    


    return 0
