from vllm import SamplingParams, LLM
import datasets
from  huggingface_hub import login
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize
from math import floor

# keeping my read token private
with open("token.txt", 'r') as f:
    hf_token = f.read() # request your own token, input here

# provides access to llama3.1 through my token
login(hf_token)
detokenizer = TreebankWordDetokenizer()

# vllm llm function
def choose_model(name):
    # llama3.1 can take a max_model_len of 128k, max length of an item in quality is ~ 36k words
    # tensor_parallel_size = 1 means that only one gpu will be used - changing this based on gpu availability
    # gpu memory util: you want a high (close to 1) to make sure the model fits (not sure) but you may overload the GPU in doing so
    # this param depends on the GPU (and their number) used for training
    llm = LLM(model = name, gpu_memory_utilization = 0.95, max_model_len = 45000, tensor_parallel_size = 1) 
    return llm

def load_data(dataset_name):
    config = "Character Archetype Classification"
    # trust_remote_code: target dataset may or may not include remote code to preprocess the dataset, not needed 
    # if your target dataset doesn't have it, but causes if no issues if left True. Only concern would be security when loading
    # datasets from untrusted sources (is my opinion)
    data = datasets.load_dataset(dataset_name, config, trust_remote_code = True)
    return data

def prompt_model(j, character): #target_otps could be experimented with, but not right now
    # you are an expert in _____,  (essentially description of the AI's role: e.g.: "You are a helpful assistant")
    system_prompt = "You are a helpful assistant." 
    # m = word_tokenize(j)
    # print(f"prompt_model: test1: This is length of input (in tokens): {len(m)}")
    # what the model is actually being tasked with: different for each question in benchmark
    
    user_prompt = f"I will present you a section of a book and a character. I want you to contemplate whether the character is a Hero, or a Villain, based on the section.\n\n"
    user_prompt = user_prompt + f"The Character is {character} and here is the section: {j}\n\n"
    
    # here I usually specify format for the model
    instructions = "Clarify your decision by returning the classification, a 'HERO' or 'VILLAIN', at the end of your message in curly brackets, for example {'HERO'}."
    return system_prompt, user_prompt, instructions

def generate(system_prompt, user_prompt, instructions, llm, nsampling = 1):
    t = 0.0 if nsampling == 1 else 0.8 # 0.8 should be probably sampled on validatoin set first, but it's a start
    #source where it will look for an answer, the prompts, nsampling is for self consistency (statistical decoding technique)
    request = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\
        \n\n**Instructions:** \n{instructions} <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n["
    # params = SamplingParams(n = nsampling, top_k=-1,min_p=0.4,temperature = t, seed = 42, max_tokens = 500) # 500 is max num of tokens in its answer
    params = SamplingParams(n = nsampling,temperature = t, seed = 42, max_tokens = 500) # 500 is max num of tokens in its answer
    # seed is random seed
    # may later include logprobs if of interest
    # m = word_tokenize(request)
    # print(f"generate: test2: len of request: {len(m)}")
    output = llm.generate(request,params, use_tqdm = False) 
    # print(f"generate: test1: {output}")
    if not output:
        return "NA"
    return output[0].outputs[0].text #returns only the raw answer text

def extractor(string):
    # Check for a word in curly brackets at the end of the string
    match_curly = re.search(r'\{(\w+)\}$', string)
    if match_curly:
        return match_curly.group(1)  # Return the word inside the curly brackets

    # If no curly bracket match, find the last occurrence of 'hero' or 'villain'
    match_role = re.findall(r'\b(hero|villain)\b', string, flags=re.IGNORECASE)
    if match_role:
        return match_role[-1].lower()  # Return the last occurrence, case-insensitive

    # If neither is found, return None
    return "NA"



def chonker(string, chonk_size):
    # I dedicate this function to my orange chonky cat
    string_tokenized = string.split()  # Split the string into words
    character = string_tokenized[0]
    l = len(string_tokenized)         # Total number of words
    lag = 0                           # Tracks the start of the current chunk
    down = floor(l / chonk_size)
    print(f"this is length of string_tokenized: {len(string_tokenized)}")
    print(f"this is the division: {l / chonk_size}")
    
    
    chonks = []                       # List to hold the resulting chunks

    # Create full chunks
    for i in range(down):
        a = lag * chonk_size
        b = (lag + 1) * chonk_size
        s = string_tokenized[a:b]
        lag += 1
        v = " ".join(s)
        chonks.append(v)

    # Handle the remainder (if any words are left after full chunks)
    if lag * chonk_size < l:
        s = string_tokenized[lag * chonk_size:]
        v = " ".join(s)
        chonks.append(v)

    return chonks, character

# An alternative that works with tokenizer instead of the s.split()
# might represent a more accurate number of tokens, but llama's tokenizer still might be different
# it might also struggle to accurately reconstruct the exact same string
def chonker2(string, chonk_size):
    # I dedicate this function to my orange chonky cat
    character = string.split()[0]  # Split the string into words
    string_tokenized = word_tokenize(string)
    
    l = len(string_tokenized)         # Total number of words
    lag = 0                           # Tracks the start of the current chunk
    down = floor(l / chonk_size)
    print(f"this is length of string_tokenized: {len(string_tokenized)}")
    print(f"this is the division: {l / chonk_size}")
    print(f"This is the floor of div: {down}")

    
    chonks = []                       # List to hold the resulting chunks

    # Create full chunks
    for i in range(down):
        a = lag * chonk_size
        b = (lag + 1) * chonk_size
        s = string_tokenized[a:b]
        lag += 1
        v = detokenizer.detokenize(s)
        chonks.append(v)

    # Handle the remainder (if any words are left after full chunks)
    if lag * chonk_size < l:
        s = string_tokenized[lag * chonk_size:]
        v = detokenizer.detokenize(s)
        chonks.append(v)

    return chonks, character


def predict(inputs, llm, chonk_size, targ_class):
    ans_full = [] # for debugging purposes
    predicts_full = []
    ext_failed = 0 # deal with later
    predicts = []
    print(len(inputs))
    for i in range(len(inputs)):
        hero = 0
        villain = 0
        print(f"Question {i+1} in progress...")
        chonks, character = chonker(inputs[i], chonk_size)
        otps = []
        target_otps = []

        for k in range(len(chonks)):
            j = chonks[k] #this changes just fine
            # print(f"This is the crux right now: {chonks[k]}") #this is a real string output
            sp, up, inst = prompt_model(j, character) #if empty bla bla, if not, ...
            text = generate(sp, up, inst, llm) #this part is the problem
            otps.append(j)
            ans = extractor(text)
            target_otps.append(ans)

            if ans.lower() == "hero":
                hero += 1
            elif ans.lower() == "villain":
                villain += 1
            else:
                ext_failed += 1
            
        if hero > villain:
            predicts.append("hero")
        else:
            predicts.append("villain")
        ans_full.append(otps)
        print(f"prediction {predicts[-1]}")
        print(f"target: {targ_class[i]}")
        predicts_full.append(target_otps)
            
    print(f"Extraction failed {ext_failed} times!")
    return ans_full, predicts_full, predicts
    
def precision(predict_class, target_class):
    l = len(predict_class)
    acc = 0
    for a,b in zip(predict_class, target_class):
        if a.lower() in b[0].lower():
            acc += 1

    if l != 0:
        return acc/l
    else:
        return 0


def main():
    model = "meta-llama/Llama-3.2-1B-Instruct"
    # model = "meta-llama/Llama-3.2-3B-Instruct"
    # model = "meta-llama/Llama-3.1-8B-Instruct"

    dataset = "ghomasHudson/muld" # selects the full MULD 
    data = load_data(dataset)

    data_train = data['train']
    data_val = data['validation']
    data_test = data['test']
    
    # chunk up 
    test_source = data_test['input']
    test_class = data_test['output']

    llm=choose_model(model)
    ans_full, predicts_full, predicts = predict(test_source, llm, 10000, test_class)
    for i in range(len(test_class)):
        print(test_class[i])
        print(predicts[i])
        print(predicts_full[i])
        print("\n\n")
    
    print(precision(predicts, test_class))


    return 0

main()