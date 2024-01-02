import random


def gen_mixed_caption(original_caption, gen_captions, clean_data_use_strategy):
    """
    Generate mixed caption from original caption and generated caption
    """
    if clean_data_use_strategy == "clean_only":
        modifed_caption = random.choice(gen_captions)
    elif clean_data_use_strategy == "mixed":
        if random.random() <= 0.5:
            modifed_caption = random.choice(gen_captions)
        else:
            modifed_caption = original_caption
    elif clean_data_use_strategy == "clean_first":
        modifed_caption = random.choice(gen_captions) + ', ' + original_caption
    elif clean_data_use_strategy == "clean_last":
        modifed_caption = original_caption + ', ' + random.choice(gen_captions)
    elif clean_data_use_strategy == "clean_random":
        if random.random() <= 0.5:
            modifed_caption = random.choice(gen_captions) + ', ' + original_caption
        else:
            modifed_caption = original_caption + ', ' + random.choice(gen_captions)
    elif clean_data_use_strategy == "noisy_only":
        modifed_caption = original_caption
    else:
        raise NotImplementedError
    return modifed_caption