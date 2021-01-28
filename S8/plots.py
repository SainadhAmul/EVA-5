# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:10:50 2021

@author: saina
"""
import matplotlib.pyplot as plt



def save_and_show_result(data, metric):
    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {label}\nPrediction: {prediction}')
        axs[row_count][idx % 5].imshow(result['image'][0], cmap='gray_r')

        # Save each image individually in labelled format
        extent = axs[row_count][idx % 5].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{metric}/labelled/{metric}_{idx + 1}.png', bbox_inches=extent.expanded(1.1, 1.5))
    
    # Save image
    fig.savefig(f'{metric}/{metric}_incorrect_predictions.png', bbox_inches='tight')