from transformers import AutoTokenizer, AutoModelForCausalLM

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words

nltk.download('punkt')
nltk.download('words')

def is_fluent_english(text):
    english_vocab = set(w.lower() for w in words.words())
    tokens = word_tokenize(text)
    num_words = len(tokens)
    num_real_words = sum(token.lower() in english_vocab for token in tokens)

    if num_words == 0:
        return 0.
    real_word_ratio = num_real_words / num_words

    return float(real_word_ratio > 0.5)

DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

topics = [
    "Celebrity culture seems to be a topic of discussion these days. What's your take?",
    "There seems to be a debate between physical books and e-books. Do you have a preference?",
    "Some people enjoy watching movies in a theater, while others prefer home settings. What about you?",
    "Smartphone usage has certainly increased over the years. What are your thoughts on this?",
    "Self-driving cars are being talked about as a part of the future. How do you feel about them?",
    "Space exploration is a topic that divides opinions. What's your perspective?",
    "There are varying opinions on the current state of education. What are your thoughts on this?",
    "Social media has its pros and cons. Do you have an opinion on this?",
    "Working from home versus commuting to an office is a hot topic nowadays. What do you think?",
    "Artificial intelligence is seen as a potential game-changer. What are your thoughts?",
    "Modern art has been a topic of debate recently. What's your viewpoint?",
    "Traditional education has its critics and supporters. Do you see it in a particular way?",
    "Virtual reality is being touted as the next big thing. What's your stance?",
    "Reality TV shows have a dedicated following. Can you relate?",
    "Cryptocurrency is being discussed as a potential future of finance. What's your opinion?",
    "The topic of gun control laws is quite polarizing. What are your thoughts?",
    "Video games are being recognized as an art form by some. Do you agree?",
    "Fast food has changed our relationship with food in various ways. What's your take?",
    "Renewable energy sources are being discussed as a way forward. What are your thoughts?",
    "The concept of privacy in the digital age is evolving. What do you think?",
    "Urban living versus rural living is a topic of discussion. What's your perspective?",
    "Fitness culture has been growing in popularity. Do you have any thoughts on this?",
    "The state of print media is a topic of discussion nowadays. What are your views?",
    "Traveling is considered by some as a great form of education. What about you?",
    "The traditional 9-5 work schedule is being questioned by some. Do you see it the same way?",
    "Traditional classrooms are being discussed as potentially outdated. What's your stance?",
    "Gourmet coffee has become quite popular. Do you have any thoughts on this?",
    "Virtual reality is being seen as a new frontier in entertainment. What do you think?",
    "Reality TV shows seem to be a trend these days. What's your opinion?",
    "Cryptocurrency is being discussed as a potential future of finance. What's your take?",
    "The popularity of organic food seems to be on the rise. Do you have an opinion on this?",
    "Renewable energy sources are being discussed as a potential hope for the future. What are your thoughts?",
    "Fast fashion has been a topic of discussion recently. What's your perspective?",
    "Celebrity gossip seems to be prevalent in the media. What are your thoughts on this?",
    "Urban living versus rural living is a topic of discussion. What do you think?",
    "Video games are being discussed as a potential art form. What's your viewpoint?",
    "The traditional 9-5 workday is being discussed as potentially outdated. What are your thoughts?",
    "Social influencers have become quite popular. Do you have any thoughts on this?",
    "Streaming services have changed the way we consume cinema. What's your take?",
    "The culture of instant gratification is a topic of discussion nowadays. What about you?",
    "The focus of the education system is being debated. What are your thoughts?",
    "Podcasts have been rising in popularity. What's your perspective?",
    "Travel is considered by some as a great form of education. Do you agree?",
    "The modern approach to fitness is a topic of discussion. What's your take?",
    "The internet has changed human interaction in various ways. What's your opinion?"
]

ENGINE_MAP = {
    'llama2_7B': 'meta-llama/Llama-2-7b-hf', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

def llama_v2_prompt(messages: list[dict]):
    if len(messages) == 0 or messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    if len(messages) == 1:
        sys_prompt = messages[0]["content"]
        return f"{BOS}{B_INST} {B_SYS}{sys_prompt}{E_SYS}"
    
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])  # zip's first arguesment is one too long for the second
    ]
    if len(messages) % 2 == 1:  # UAUAUAU
        messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
    else:
        messages_list.append(f"{BOS}{B_INST}")
    return "".join(messages_list)

def process_answer(s):
    s = s.strip()
    return s.replace("\n", " ")

def pkl2script(pkl):
    topic = pkl["topic"]
    history = pkl["history"]
    prompt = ""
    for i in range(len(history)):
        if i % 2 == 0:
            prompt += f"Turn {i+1}: User: {history[i]}\n\n"  # Bot A is the user, who started the conversation
        else:
            prompt += f"Turn {i+1}: Assistant: {history[i]}\n\n"
    prompt = prompt.strip()
    return prompt

def qa2qa_prompt(q, a):
    return f"In the last round, you asked me: \"{q}\" and I answered: \"{a}\""

def pkl2dict(pkl):
    """
    pkl = {
        "topic": topic, 
        "history": [topic], 
        "seed": args.seed, 
        "persona": persona, 
        "user": user,
    }
    """
    res = []
    his = pkl["history"]
    if len(pkl["history"]) % 2 == 0:  # prompt user to talk: Q, A, Q, A into QA, Q, A
        res.append({"role": "system", "content": pkl["user"]})
        res.append({"role": "user", "content": qa2qa_prompt(his[0], his[1])})
        for i, msg in enumerate(his[2:]):
            if i % 2 == 0:
                res.append({"role": "assistant", "content": msg})
            else:
                res.append({"role": "user", "content": msg})
    else:  # prompt assistant to talk: Q, A, Q, A, Q into Q, A, Q, A, Q
        res.append({"role": "system", "content": pkl["persona"]})
        for i, msg in enumerate(his):
            if i % 2 == 0:
                res.append({"role": "user", "content": msg})
            else:
                res.append({"role": "assistant", "content": msg})
    return res


def load_model(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", load_in_8bit=True)
    return tokenizer, model