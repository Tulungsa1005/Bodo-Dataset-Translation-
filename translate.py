import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
import os
from tqdm import tqdm
import logging
from datetime import datetime

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available! Running on GPU:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")

    return device

# Function to set up logging
def setup_logging(log_file_path):
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Function to read text from file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to write text to file
def write_text_file(output_chunks, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for chunk in output_chunks:
            sentences = chunk.split('. ')
            text = '\n'.join(sentences)
            file.write(text + '\n')

# Function to translate large file
def translate_large_file(input_text, output_file_path, model, tokenizer, ip, device, chunk_size=1000):
    chunks = [input_text[i:i+chunk_size] for i in range(0, len(input_text), chunk_size)]
    output_chunks = []

    for chunk in chunks:
        batch = ip.preprocess_batch([chunk], src_lang="eng_Latn", tgt_lang="brx_Deva")
        batch = tokenizer(batch, src=True, return_tensors="pt").to(device)

        with torch.no_grad():  
            outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

        outputs = tokenizer.batch_decode(outputs, src=False)
        outputs = ip.postprocess_batch(outputs, lang="brx_Deva")
        
        output_chunks.extend(outputs) 
    
    write_text_file(output_chunks, output_file_path)
    print("Translation completed. Translated text saved to", output_file_path)

# Function to translate multiple files
def translate_multiple_files(input_directory, output_directory, model, tokenizer, ip, device):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_files = os.listdir(input_directory)    

    for filename in tqdm(input_files, desc="Translating files"):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename.split('.')[0] + ".txt")
        input_text = read_text_file(input_file_path)

        logging.info(f"Started translating file: {filename}")
        translate_large_file(input_text, output_file_path, model, tokenizer, ip, device)
        logging.info(f"Finished translating file: {filename}")

    print("Translation of all files completed.")

def main():
    log_file_path = 'translation_log.log'  
    setup_logging(log_file_path)

    device = set_device()
    
    # Model and tokenizer initialization
    tokenizer = IndicTransTokenizer(direction="indic-en")
    ip = IndicProcessor(inference=True)
    path = '/home/mn/repos/bodo-dataset-creation/indictrans2-indic-en-1B'
    model = AutoModelForSeq2SeqLM.from_pretrained(path, trust_remote_code=True, local_files_only=True).to(device)
    model.tie_weights()  

    # Directories and file lists
    train_data_directory = "/home/mn/repos/bodo-dataset-creation/tempo"
    test_data_directory = "/home/mn/repos/bodo-dataset-creation/dataset/IN-Abs/test-data"
    data_flag = 'train'  # it can be either test or train
    dataset_directory = ''

    if data_flag == 'train':
        dataset_directory = train_data_directory
    elif data_flag == 'test':
        dataset_directory = test_data_directory
    else:
        raise Exception('dataset flag not properly set')

    input_directory = dataset_directory
    output_directory = '/home/mn/repos/bodo-dataset-creation/zy'
    
    # Translation
    translate_multiple_files(input_directory, output_directory, model, tokenizer, ip, device)

if __name__ == "__main__":
    main()
