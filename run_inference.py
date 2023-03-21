from src.tasks.run_caption_inference_batch import get_captions


if __name__ == "__main__":
    outputs = get_captions(device="cuda:0", video_batch_folder='./input_videos')
    print(outputs)