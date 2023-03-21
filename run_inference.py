from src.tasks.run_caption_inference_batch import get_captions
from sentence_transformers import SentenceTransformer, util

def compare_sentences(sentence1, sentence2, threshold = 0.5, device = "cpu"):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    embedding_1= model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)

    sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    
    return (sim > threshold)


def is_in(value, value_list, comparer_function, device="cpu"):
    """ checkes whether "value" already in "value_list" """
    for vi in value_list:
        if comparer_function(vi, value, device=device):
            return True
    return False

def make_unique_set(in_list, comparer_function=lambda a, b: a == b, device="cpu"):
    """ retusn unique set of "in_list" """
    new_list = []
    for i in in_list:
        if not is_in(i, new_list, comparer_function, device):
            new_list.append(i)
    return new_list

def get_caption_list(caption_outputs:list, device="cpu"):
    captions = []
    for i in caption_outputs:
        for j in i:
            captions.append(j[1])
    
    return make_unique_set(captions, compare_sentences, device)

if __name__ == "__main__":
    outputs = get_captions(device="cuda:0", video_batch_folder='./input_videos')
    
    print(len(outputs))
    print(outputs)
    for output in outputs:
        print(output)
    print(get_caption_list(outputs, "cuda:0"))