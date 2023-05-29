import pandas as pd 
import run_inference
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="7"


def filter(df:pd.DataFrame, column:str = 'input', filter_out:str = '.'):
    df = df[df[column] != filter_out]
    return df

def get_infered_models(test_dataset:str = './datasets/CSV_DS/YouCook2/new_test.csv', device:str = 'cpu', video_folder:str = './Videos'):
    df = pd.read_csv(test_dataset)
    new_df = run_inference.run_inference_on_ds(df, device, video_folder)
    new_df = filter(new_df)

    return new_df

if __name__ == '__main__':
    df = pd.read_csv('./vid_split.csv')
    new_df = run_inference.run_inference_on_ds(df, 'cpu')
    new_df = filter(new_df)
    new_df.to_csv('./vid_inf_new.csv', index=False)
