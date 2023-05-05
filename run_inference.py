from src.tasks.run_caption_inference_batch import get_captions, get_captions_from_folder, get_args, get_model_tokenizer_tensorizer

import pandas as pd

#from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, FlaxLongT5ForConditionalGeneration, T5ForConditionalGeneration

import torch

import os

from tqdm import tqdm

import logging
#from torchtext.models import T5_BASE_GENERATION
#from torchtext.prototype.generate import GenerationUtils


VIDEO_FOLDER = os.path.join('./VideoFolder', 'youcook2', 'Test')

class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)

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

"""
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
    return new_list """

def get_caption_list(caption_outputs:list, comparor:any):
    captions = []
    for i in caption_outputs:
        for j in i:
            captions.append(j[1])
    
    return captions#make_unique_set(captions, comparor.compare)

def get_summary(captions:list, summarizer):
    caption_string = ". ".join(captions)
    caption_string = "summarize: " + caption_string

def infer_row(row:pd.Series, args, components, video_folder:str = VIDEO_FOLDER):
    file_names = row['video_files'].split(';;;')
    parent_folder = os.path.join(video_folder, row['video_id'])
    file_names = [os.path.join(parent_folder, x) for x in file_names]
    parsed_output = []
    if os.path.exists(parent_folder):
        with DisableLogger():
            comparitor = None#Sentence_Compare(device=device)
            outputs = get_captions(video_batch_files=file_names, args=args, **components)

            parsed_output = get_caption_list(outputs, comparor=comparitor)

    return parsed_output


def format_captions(captions:list) -> str:
    output = '. '.join(captions) + '.'
    return output

def replace_df_column(df:pd.DataFrame, column_name:str, new_values:list) -> pd.DataFrame:
    df[column_name + "_old"] = df[column_name]
    df[column_name] = new_values

    return df

def run_inference_on_ds(df:pd.DataFrame, device, video_folder:str=VIDEO_FOLDER) -> pd.DataFrame:
    inferences = []
    args = get_args(device=device)
    components = get_model_tokenizer_tensorizer(args=args)
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        captions = infer_row(row, args, components, video_folder)
        inferences.append(format_captions(captions))
    
    
    return replace_df_column(df, 'input', inferences)




if __name__ == "__main__":
    device = "cuda:0"
    print(torch.version.cuda)
    # summarizer = FlaxLongT5Summarizer()
    # summarizer = T5Summarizer()
    comparitor = None#Sentence_Compare(device=device)
    args = get_args(device=device, video_batch_folder='./input_videos')

    with DisableLogger():
        outputs = get_captions_from_folder(args=args, video_batch_folder='./input_videos')
    
    print(len(outputs))
    print(outputs)
    parsed_output = get_caption_list(outputs, comparor=comparitor)
    print(parsed_output)
    print(len(parsed_output))

    # print(summarizer.summarize(parsed_output))