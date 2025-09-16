import json
import os
import datasets
from tqdm import tqdm


def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def get_all_data_under_dir(dir):
    all_files = os.listdir(dir)
    data = []
    for f in all_files:
        file_path = os.path.join(dir, f)
        data.extend(load_json(file_path))
    # print(f'Number of data {len(data)}')
    return data


# reformat photochat
def reformat_photochat_dp(dp):
    dialogue_utterances = []
    img_idx = 0
    for turn in dp['dialogue']:
        turn_info = {'utterance': turn['message'],
                    'speaker': turn['user_id'],
                    'speaker_id': turn['user_id'],
                    'shared_images': None}
        if turn['share_photo']:
            turn_info['shared_images'] = img_idx
            img_idx += 1
        dialogue_utterances.append(turn_info)
    return dialogue_utterances


# reformat dialogCC
def reformat_dialogCC_dp(dp):
    dialogue_utterances = []
    img_idx = 0
    speaker_map = {}
    speaker_count = 0
    for turn in dp['dialogue']:
        if turn['speaker'] not in speaker_map:
            speaker_map[turn['speaker']] = speaker_count
            speaker_count += 1
        turn_info = {'utterance': turn['utterance'],
                    'speaker': turn['speaker'],
                    'speaker_id': speaker_map[turn['speaker']], # first speaker is 0, second speaker is 1
                    'shared_images': None}
        if turn['shared_image'] != []:
            for i, img in enumerate(turn['shared_image']):                 
                turn_info['shared_images'] = img_idx
                img_idx += 1       
                break

        dialogue_utterances.append(turn_info)
    return dialogue_utterances


# function to find the original dialogue
def find_original_dialogue(dp):
    dialogue_id = dp['original_dialogue_id']
    dialogue_split = dp['original_data_split']
    dialogue_dataset = dp['original_dataset']

    if dialogue_dataset == 'PhotoChat':
        search_base = photochat_data[dialogue_split]
        for dp in search_base:
            if dp['dialogue_id'] == dialogue_id:
                return 'PhotoChat', dp
    elif dialogue_dataset == 'DialogCC':
        if dialogue_split == 'train':
            search_base = dialogCC_dataset['train']
        elif dialogue_split == 'valid':
            search_base = dialogCC_dataset['validation']
        elif dialogue_split == 'test':
            search_base = dialogCC_dataset['test']
        
        for dp in search_base:
            if dp['dialogue_id'] == dialogue_id:
                return 'DialogCC', dp
        
if __name__ == '__main__':
    # Load dialogCC dataset
    dialogCC_dataset = datasets.load_dataset("passing2961/dialogcc")

    # Load photochat dataset
    splits = ['train', 'dev', 'test']  
    photochat_data = {}
    for split in splits:
        data_dir = f'google-research/multimodalchat/photochat/{split}'
        photochat_data[split] = get_all_data_under_dir(data_dir)

    # Load mdsEval annotations
    data_path = 'MDSEval_annotations.json'
    mdsEval_annotations = load_json(data_path)

    # matched_original_data = []
    matched_dialogCC_data = []
    matched_photochat_data = []
    for dp in tqdm(mdsEval_annotations):
        dialogue_dataset, matched_dp = find_original_dialogue(dp)
        if dialogue_dataset == 'PhotoChat':
            reformated_dp = reformat_photochat_dp(matched_dp)
            dp['dialogue'] = reformated_dp
        elif dialogue_dataset == 'DialogCC':
            reformated_dp = reformat_dialogCC_dp(matched_dp)
            dp['dialogue'] = reformated_dp


    # save merged data
    save_path = 'MDSEval_data.json'
    save_json(mdsEval_annotations, save_path)