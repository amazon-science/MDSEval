import torch
from PIL import Image
import open_clip
import requests
from io import BytesIO
from tqdm import tqdm
import numpy as np
import json


def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data


def get_image_from_path(image_path):
    img = Image.open(image_path)
    return img


def projection_A_to_B(A, B):
    B_magnitudes = np.sqrt(np.einsum('ij, ij -> i', B, B))
    B_unit = B / B_magnitudes[:, np.newaxis]
    proj_A_to_B_magnitudes = np.einsum('ij, ij -> i', A, B_unit)
    proj_A_to_B = proj_A_to_B_magnitudes[:, np.newaxis] * B_unit
    return proj_A_to_B, proj_A_to_B_magnitudes


def calculate_meki(A, B, C):
    proj_A_B, _ = projection_A_to_B(A, B)
    meki, meki_magnitude = projection_A_to_B(A - proj_A_B, C)
    return meki, meki_magnitude


def encode_images_from_path(image_paths, model, preprocess_func, device, image_batch_size=32):
    with torch.no_grad(), torch.amp.autocast(device.type):
        image_features_buf = []
        print('Start encoding images ...')
        for i in tqdm(range(0, len(image_paths), image_batch_size)):
            batch_data = image_paths[i:i+image_batch_size]
            batch_image = []
            for image_path in batch_data:
                img = get_image_from_path(image_path)
                if img is not None:
                    batch_image.append(img)

            batch_image = torch.stack([preprocess_func(img) for img in batch_image]).to(device)  # [batch_size, channels, height, width]
            batch_image_features = model.encode_image(batch_image)  # Process all images in batch.
            image_features_buf.append(batch_image_features)
        image_features = torch.cat(image_features_buf, dim=0)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    image_features = image_features.detach().cpu().numpy().tolist()
    return image_features


def encode_text(texts, tokenizer, model, device, text_batch_size=10):
    with torch.no_grad(), torch.amp.autocast(device.type):
        text_features_buf = []
        print('Start encoding texts ...')
        for i in tqdm(range(0, len(texts), text_batch_size)):
            batch_data = texts[i:i+text_batch_size]
            batch_text = tokenizer(batch_data)
            batch_text = batch_text.to(device)
            batch_text_features = model.encode_text(batch_text)  # Process all texts in batch.
            text_features_buf.append(batch_text_features)
        text_features = torch.cat(text_features_buf, dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.detach().cpu().numpy().tolist()
    return text_features


def aggregate_dialogue_string(dialogue):
    dialog_string = ''
    for turn in dialogue:
        if turn['shared_images'] == None:
            dialog_string += f'Speaker {turn["speaker"]}: {turn["utterance"]}\n'
        else:
            dialog_string += f'Speaker {turn["speaker"]}: [SHARED PHOTO, PLEASE REFER TO THE PHOTO DESCRIPTIONS]\n'
    return dialog_string.strip()


def extract_dialog_text(data):
    for dp in data:
        dialog_string = aggregate_dialogue_string(dp['dialogue'])
        dp['dialogue_text'] = dialog_string
    return data


def load_clip_model(model_name, pretrained):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, preprocess, device




if __name__ == '__main__':
    # Load data
    data_path = 'MDSEval/MDSEval_data.json'
    data = load_json(data_path)


    # Load CLIP model
    model_name = 'ViT-H-14-378-quickgelu'
    pretrained = 'dfn5b'
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()  
    tokenizer = open_clip.get_tokenizer(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Encode all modalities into CLIP space - image, dialogue text, pseudo summary
    data = encode_images_from_path(data, model, preprocess, device, image_batch_size=32)
    data = extract_dialog_text(data)
    data = encode_text(data, tokenizer, model, device, modality='dialogue_text', text_batch_size=10)    
    data = encode_text(data, tokenizer, model, device, modality='pseudo_summary', text_batch_size=10)

    # Get the encoding features for each modality, I: image, T: dialogue text, S: pseudo summary
    I = np.array([dp['image_features'] for dp in data])
    T = np.array([dp['dialogue_text_features'] for dp in data])
    S = np.array([dp['pseudo_summary_features'] for dp in data])

    # Calculate MEKI scores
    _, meki_I = calculate_meki(I, T, S)
    _, meki_T = calculate_meki(T, I, S)

    # Save MEKI scores to data
    for idx, dp in enumerate(data):
        dp['meki_I'] = meki_I[idx]
        dp['meki_T'] = meki_T[idx]

    # Save data with MEKI scores
    save_path = 'MDSEval_data_with_meki.json'
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        