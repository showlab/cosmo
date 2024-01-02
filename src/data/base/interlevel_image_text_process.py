import os
import json
import braceexpand
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

try:
    from azfuse import File
except Exception as e:
    print("azfuse not supported on this cluster, use local file system instead")

def load_sizes(sizes_filename, use_azfuse):
    """
    Helper function to load sizes from the num_samples.json file.
    """
    sizes = {}
    try:
        open_func = File.open if use_azfuse else open
        with open_func(sizes_filename, "r") as fp:
            sizes = json.load(fp)
    except Exception as e:
        print(e)
    return sizes

def calculate_total_size(sizes, shards_list):
    """
    Helper function to calculate the total size from a list of shards.
    """
    return sum(
        [int(sizes[os.path.basename(shard)]) if os.path.basename(shard) in sizes else 0 for shard in shards_list]
    )

def get_dataset_size(shards, use_azfuse=False, estimated_sample_per_shard=5000):
    """
    Read the total training samples from num_samples.json;
    the problem is we need to compute the train sample size for each shard with utils/count_webdataset_sample.py at first.
    """
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "num_samples.json")
    num_shards = len(shards_list)
    
    sizes = load_sizes(sizes_filename, use_azfuse)
    total_size = calculate_total_size(sizes, shards_list)
    print(f"Sizes Filename: {sizes_filename}, Total size: {total_size}, num_shards: {num_shards}")

    if total_size == 0:
        print(f"!!!Could not find num_samples.json or __len__ in {dir_path}")
        print(f"!!!Since don't know the dataset size, estimating dataset size as {num_shards * estimated_sample_per_shard}")
        print(f"!!!It's best to generate num_samples.json with utils/count_webdataset_sample.py at first")
        total_size = num_shards * estimated_sample_per_shard

    return total_size, num_shards




def select_mmc4_subsampled_text(text_lists, img_len, matched_sentence_idxs, adjcent_sampling_rate=0.5, num_selected_tokens=64, ideal_num_images=3):
    """
    We want the model to sample ideal_num_images at best.
    The logic here is quite simple. Since we match N images with N sentences,
    4/5 is a magic number, it is the average mapping from tokens to characters.
    But in such way, some text maybe too short and a lot of tensor in label may be -100.
    In addition, the generated text may be unnatural.
    """
    new_sentences = []
    if img_len >= ideal_num_images:
        num_selected_characters = int(num_selected_tokens * 4) // ideal_num_images
    else:
        num_selected_characters = int(num_selected_tokens * 5) // img_len
    for idx, sentence in enumerate(text_lists):
        if idx not in matched_sentence_idxs:
            preserved_characters = int(num_selected_characters * 0.4)
        else:
            preserved_characters = num_selected_characters
        if len(sentence) > preserved_characters:
            if random.random() <= adjcent_sampling_rate:
                begin_index = 0
            else:
                begin_index = random.randint(0, len(sentence) - preserved_characters)
            sampled_sentence = sentence[begin_index:begin_index + preserved_characters]
            new_sentences.append(sampled_sentence)
        else:
            new_sentences.append(sentence)
    return new_sentences


def select_cc3m_subsampled_text(text_lists, img_len, matched_sentence_idxs, adjcent_sampling_rate=0.9, num_selected_tokens=64, ideal_num_images=3):
    """
    We want the model to sample ideal_num_images at best.
    The logic here is quite simple. Since we match N images with N sentences,
    4/5 is a magic number, it is the average mapping from tokens to characters.
    But in such way, some text maybe too short and a lot of tensor in label may be -100.
    In addition, the generated text may be unnatural.
    """
    new_sentences = []
    if img_len >= ideal_num_images:
        num_selected_characters = int(num_selected_tokens * 4.5) // ideal_num_images
    else:
        num_selected_characters = int(num_selected_tokens * 5) // img_len
    for idx, sentence in enumerate(text_lists):
        if idx not in matched_sentence_idxs:
            preserved_characters = int(num_selected_characters * 0.4)
        else:
            preserved_characters = num_selected_characters
        if len(sentence) > preserved_characters:
            if random.random() <= adjcent_sampling_rate:
                begin_index = 0
            else:
                begin_index = random.randint(0, len(sentence) - preserved_characters)
            sampled_sentence = sentence[begin_index:begin_index + preserved_characters]
            new_sentences.append(sampled_sentence)
        else:
            new_sentences.append(sentence)
    return new_sentences


def select_obelics_subsampled_text(sentences, img_len, num_selected_tokens=64, adjcent_sampling_rate=0.5, ideal_num_images=3):
    """
    preserve adjcent text with higher probability
    """
    new_sentences = []
    non_empty_sentences = [sentence for sentence in sentences if sentence is not None]
    real_img_num = min(img_len, ideal_num_images)
    num_selected_characters = int(num_selected_tokens * 3.7) // real_img_num  # control the length of the text
    previous_is_img = False
    for sentence in sentences:
        if sentence is None:
            previous_is_img = True 
            new_sentences.append(None)
            continue
        else:
            # Remove newline characters
            sentence = sentence.replace("\n", "")
        if len(sentence) > num_selected_characters:
            # if only one sentence, we need to sample it multiple times
            if previous_is_img:
                if random.random() <= adjcent_sampling_rate:
                    begin_index = 0
                else:
                    begin_index = random.randint(0, len(sentence) - num_selected_characters)
                sampled_sentence = sentence[begin_index:begin_index + num_selected_characters]
            else: 
                if random.random() <= adjcent_sampling_rate:
                    sampled_sentence = sentence[:num_selected_characters//min(ideal_num_images,len(non_empty_sentences))]
                else:
                    begin_index = random.randint(0, len(sentence) - num_selected_characters//min(ideal_num_images,len(non_empty_sentences)))
                    sampled_sentence = sentence[begin_index:begin_index + num_selected_characters//min(ideal_num_images,len(non_empty_sentences))]
            new_sentences.append(sampled_sentence)
        else:
            new_sentences.append(sentence)
        previous_is_img = False
    # original_num_characters_array = np.array([len(sentence) for sentence in non_empty_sentences])
    # sampled_num_characters_array = np.array([len(sentence) for sentence in new_sentences if sentence is not None])
    # print(f"Original num characters: {original_num_characters_array}, sampled num characters: {sampled_num_characters_array}")
    return new_sentences

def flip_scores(scores, text_len, img_len):
    """
    if we flip the texts, the scores for text should be flipped as well
    """
    flipped_scores = []
    for index in range(len(scores)):
        row = int(scores[index][0].split('_')[0][1:])
        col = int(scores[index][0].split('_')[1][1:])
        flipped_scores.append([f"t{text_len-1-row}_i{img_len-1-col}", scores[index][1]])
    return flipped_scores




def select_image_index_from_score(scores, text_len, image_len, disturb=True):
    """
    select the image that maximum thresh outperform image_sim_thresh;
    strategy: image_wise, document_wise;
    To prevent overfitting, we distub the scores a little bit.
    For sample with low score, we may replace it with noisy generated text.
    """
    # Compute the average score for each image
    score_matrix = np.zeros((text_len, image_len))
    # extra row and column from 't0_i1'
    for index in range(len(scores)):
        row = int(scores[index][0].split('_')[0][1:])
        col = int(scores[index][0].split('_')[1][1:])
        score_matrix[row][col] = scores[index][1]
    # if strategy == "document_wise":
    #     # return all images
    #     if np.max(score_matrix) > image_sim_thresh:
    #         return np.arange(image_len)
    #     else:
    #         raise ValueError("The max score of this document is lower than image_sim_thresh in obelics!")
    # elif strategy == "image_wise":
    #     # return the images that max score > image_sim_thresh
    #     selected_image_ixs = []
    #     for j in range(image_len):
    #         if np.max(score_matrix[:, j]) >= image_sim_thresh:
    #             selected_image_ixs.append(j)
    #     if len(selected_image_ixs) == 0:
    #         # print("Max score: ", np.max(score_matrix))
    #         raise ValueError("No images in obelics sample due to low simlarity score!")
    # else:
    #     raise ValueError("strategy not defined!")
    # disturb the score a little bit to prevent all images use same texts
    
    selected_image_ixs = []
    for j in range(image_len):
        if np.max(score_matrix[:, j]) > 0:
            selected_image_ixs.append(j)
            
    if disturb:
        disturb_matrix = np.random.normal(0, 0.02, score_matrix.shape)
        disturb_matrix = np.clip(disturb_matrix, -0.04, 0.04)
        score_matrix = score_matrix + disturb_matrix

    # for each selectd image, find the sentence that have the highest matched score
    matched_sentence_ixs = []
    matched_sentence_scores = []
    for i in selected_image_ixs:
        matched_sentence_ixs.append(np.argmax(score_matrix[:, i]))
        matched_sentence_scores.append(np.max(score_matrix[:, i]))
    return selected_image_ixs, matched_sentence_ixs, matched_sentence_scores



def obelics_optim_assignments(scores, text_len, image_len, disturb=True):
    """
    select the image that maximum thresh outperform image_sim_thresh;
    strategy: image_wise, document_wise;
    To prevent overfitting, we distub the scores a little bit.
    For sample with low score, we may replace it with noisy generated text.
    """
    # as some images are None
    # Compute the average score for each image
    score_matrix = np.zeros((text_len, image_len))
    # extra row and column from 't0_i1'
    for index in range(len(scores)):
        col = int(scores[index][0].split('_')[0][1:]) # text
        row = int(scores[index][0].split('_')[1][1:]) # image
        score_matrix[row][col] = scores[index][1]
    
    selected_image_ixs = []
    for j in range(image_len):
        if np.max(score_matrix[j, :]) > 0:
            selected_image_ixs.append(j)
            
    if disturb:
        disturb_matrix = np.random.normal(0, 0.02, score_matrix.shape)
        disturb_matrix = np.clip(disturb_matrix, -0.04, 0.04)
        score_matrix = score_matrix + disturb_matrix

    # for each selectd image, find the sentence that have the highest matched score
    sim_matrix = score_matrix[selected_image_ixs]
    cost_matrix = -sim_matrix
    image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    matched_sentence_ixs = sentence_indices
    matched_sentence_scores = sim_matrix[image_indices, sentence_indices]

    return image_indices, matched_sentence_ixs, matched_sentence_scores
    # return selected_image_ixs, matched_sentence_ixs, matched_sentence_scores