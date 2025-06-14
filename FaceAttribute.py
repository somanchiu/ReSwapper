import os
import numpy as np

import swap

def create_linear_direction_dataset(dataset_dir, output_path = None):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            image_paths.append(os.path.join(root, file))

    embeds_list = []
    for path in image_paths:
        embed = swap.create_source(path)
        if embed is not None:
            embeds_list.append(embed)

    embeds = np.stack(embeds_list)

    if output_path is not None:
        np.save(output_path, embeds)

    return embeds

def get_direction(dataset_a, dataset_b, output_path = None):
    direction = np.mean(dataset_a, axis=0) - np.mean(dataset_b, axis=0)
    if output_path is not None:
        np.save(output_path, direction)

    return direction
