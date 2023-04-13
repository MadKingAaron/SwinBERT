from src.tasks.run_caption_inference_batch import get_captions

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, FlaxLongT5ForConditionalGeneration, T5ForConditionalGeneration

import torch
#from torchtext.models import T5_BASE_GENERATION
#from torchtext.prototype.generate import GenerationUtils

class Summarizer():
    def __init__(self) -> None:
        pass

    def format_caption_list(self, captions:list)->str:
        caption_string = ". ".join(captions)
        caption_string = "summarize: " + caption_string
        return caption_string

    def summarize(self, captions:list) -> str:
        pass

# class T5SummarizerTorchText(Summarizer):
#     def __init__(self) -> None:
#         t5_base = T5_BASE_GENERATION
#         transform = T5_BASE_GENERATION.transform()




#         super().__init__()

class T5Summarizer(Summarizer):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model =T5ForConditionalGeneration.from_pretrained("t5-small", )
        
        super().__init__()
    
    def summarize(self, captions: list) -> str:
        to_summarize = self.format_caption_list(captions)
        input_ids = self.tokenizer(to_summarize, return_tensors="pt", max_length='model_max_length').input_ids

        outputs = self.model.generate(input_ids)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class FlaxLongT5Summarizer(Summarizer):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")
        
        super().__init__()

    def summarize(self, captions:list):
        to_summarize = self.format_caption_list(captions)
        inputs = self.tokenizer([to_summarize], return_tensors="np", max_length=512)
        summary_ids = self.model.generate(inputs["input_ids"]).sequences
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)


class Sentence_Compare():
    def __init__(self, model_type = "sentence-transformers/all-MiniLM-L6-v2", device = "cpu", threshold = 0.75) -> None:
        self.model = SentenceTransformer(model_type, device = device)
        self.threshold = threshold
    
    def compare(self, sentence1, sentence2):
        embedding_1 = self.model.encode(sentence1, convert_to_tensor=True, show_progress_bar=False)
        embedding_2 = self.model.encode(sentence2, convert_to_tensor=True, show_progress_bar=False)

        sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return (sim > self.threshold)

        

def compare_sentences(sentence1, sentence2, threshold = 0.8, device = "cpu"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    embedding_1= model.encode(sentence1, convert_to_tensor=True, show_progress_bar=False)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True, show_progress_bar=False)

    sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    
    return (sim > threshold)


def is_in(value, value_list, comparer_function):
    """ checkes whether "value" already in "value_list" """
    for vi in value_list:
        if comparer_function(vi, value):
            return True
    return False

def make_unique_set(in_list, comparer_function=lambda a, b: a == b):
    """ retusn unique set of "in_list" """
    new_list = []
    for i in in_list:
        if not is_in(i, new_list, comparer_function):
            new_list.append(i)
    return new_list

def get_caption_list(caption_outputs:list, comparor:Sentence_Compare):
    captions = []
    for i in caption_outputs:
        for j in i:
            captions.append(j[1])
    
    return make_unique_set(captions, comparor.compare)

def get_summary(captions:list, summarizer):
    caption_string = ". ".join(captions)
    caption_string = "summarize: " + caption_string


if __name__ == "__main__":
    device = "cuda:0"
    print(torch.version.cuda)
    # summarizer = FlaxLongT5Summarizer()
    # summarizer = T5Summarizer()
    comparitor = Sentence_Compare(device=device)
    


    outputs = get_captions(device=device, video_batch_folder='./input_videos')
    
    print(len(outputs))
    print(outputs)
    parsed_output = get_caption_list(outputs, comparor=comparitor)
    print(parsed_output)
    print(len(parsed_output))

    # print(summarizer.summarize(parsed_output))