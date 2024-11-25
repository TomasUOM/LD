from vllm import SamplingParams, LLM
import datasets
from  huggingface_hub import login
# import nltk
import re

# keeping my read token private
with open("token.txt", 'r') as f:
    hf_token = f.read() # request your own token, input here

# provides access to llama3.1 through my token
login(hf_token)

# 
#nltk.download('punkt',quiet=True)
#nltk.download('stopwords',quiet=True)
#nltk.download('wordnet',quiet=True)

# NLTK tools to help tokenize text
# stop_words = set(nltk.stopwords.words('english'))
# stemmer = nltk.stem.PorterStemmer()
# tokenizer = nltk.tokenize.word_tokenize()
# lemmatizer = nltk.WordNetLemmatizer()

# vllm llm function
def choose_model(name):
    # llama3.1 can take a max_model_len of 128k, max length of an item in quality is ~ 36k words
    # tensor_parallel_size = 1 means that only one gpu will be used - changing this based on gpu availability
    # gpu memory util: you want a high (close to 1) to make sure the model fits (not sure) but you may overload the GPU in doing so
    # this param depends on the GPU (and their number) used for training
    llm = LLM(model = name, gpu_memory_utilization = 0.95, max_model_len = 40000, tensor_parallel_size = 1) 
    return llm

def load_data(dataset_name):
    # trust_remote_code: target dataset may or may not include remote code to preprocess the dataset, not needed 
    # if your target dataset doesn't have it, but causes if no issues if left True. Only concern would be security when loading
    # datasets from untrusted sources (is my opinion)
    data = datasets.load_dataset(dataset_name, trust_remote_code = True)
    return data

def prompt_model(source, question, options): #tbd later, using format 
    # you are an expert in _____,  (essentially description of the AI's role: e.g.: "You are a helpful assistant")
    system_prompt = "You are a helpful assistant." 
    # what the model is actually being tasked with: different for each question in benchmark
    user_prompt = f"Please choose an appropriate answer from the four options, given the following Source material, Question, and Options."
    user_prompt = user_prompt + f"The Source material is: {source}/n/n" + f"The Question is: {question}\n\n" + f"The options are as follows:\n\n Option 1: {options[0]}\n\n Option 2 {options[1]}\n\n Option 3: {options[2]}\n\n Option 4: {options[3]}\n\n"
    # here I usually specify format for the model
    instructions = "Clarify your decision by returning the number, which represents your chosen option, at the end of your message in curly brackets, for example {1}."
    return system_prompt, user_prompt, instructions

def generate(system_prompt, user_prompt, instructions, llm, nsampling = 1):
    t = 0.0 if nsampling == 1 else 0.8 # 0.8 should be probably sampled on validatoin set first, but it's a start
    #source where it will look for an answer, the prompts, nsampling is for self consistency (statistical decoding technique)
    request = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\
        \n\n**Instructions:** \n{instructions} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n["
    params = SamplingParams(n = nsampling, top_k=-1,min_p=0.4,temperature = t, seed = 42, max_tokens = 500) # 500 is max num of tokens in its answer
    # seed is random seed
    # may later include logprobs if of interest
    output = llm.generate(request,params, use_tqdm = False) # trying use_tqdm true for the first time, dunno how it will work
    return output[0].outputs[0].text #returns only the raw answer text

def last_int(text):
    integers = re.findall(r'\d+', text)
    if integers:
        return int(integers[-1])
    else:
        return -1

# def predict(source, question, options):
def predict(source, question, options, llm):
    ans_full = []
    num_ans = []
    ext_failed = 0
    y = 1
    for a,b,c in zip(source, question, options):
        print(f"Question {y} in progress...")
        y += 1
        sp, up, i = prompt_model(a,b,c)
        text = generate(sp, up, i, llm)
        ans_full.append(text)
        num = last_int(text)
        num_ans.append(num)
        if num == -1:
            ext_failed += 1
    print(f"Extraction failed {ext_failed} times!")
    return ans_full, num_ans


def precision(num_ans, target):
    total = len(target)
    acc = 0
    for i in range(len(num_ans)):
        if num_ans[i]-1 == target[i]:
            acc += 1
    print(f"Accuracy: {acc} out of {total}")
    if total != 0:

        print(f"Accuracy: {acc/total}")
        return acc/total
    else:
        return 0



# processing functions - is this necessary actually? idk

def main():
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Manan, Kasyap
    dataset = "emozilla/quality" # Kasyap
    

    data = load_data(dataset)

    data_train = data['train']
    data_val = data['validation'][:1000] #anybody
    data_test = data['validation'][1000:]

    test = data_val
    source = test['article']
    question = test['question']
    options = test['options']
    target = test['answer']

    llm=choose_model(model)
    
    ans_full, num_ans = predict(source, question, options, llm)
    print(ans_full[:10])
    print(num_ans)
    print(target)
    precision(num_ans, target)
    


    # print(data_train[0])



    return 0

main()