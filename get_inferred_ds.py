import pandas as pd 
import run_inference

def filter(df:pd.DataFrame, column:str = 'input', filter_out:str = '.'):
    df = df[df[column] != filter_out]
    return df

def get_infered_models(test_dataset:str = './datasets/CSV_DS/YouCook2/new_test.csv', device:str = 'cpu'):
    df = pd.read_csv(test_dataset)
    new_df = run_inference.run_inference_on_ds(df, device)
    new_df = filter(new_df)

    return new_df

if __name__ == '__main__':
    df = pd.read_csv('./vid_split.csv')
    new_df = run_inference.run_inference_on_ds(df, 'cuda:0')
    new_df = filter(new_df)
    new_df.to_csv('./vid_inf.csv', index=False)
