import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
#import seaborn as sns
import os,sys
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve, roc_auc_score

#sys.path('~/groups/icecube/simon/GNN/workspace')
#from External_Functions.ExternalFunctions import nice_string_output

# Set custom color cycle
custom_colors = [    
    '#0081C8',  # Olynmpic Blue
    '#FCB131', # Olympic Yellow
    '#000000', # Olympic Black
    '#00A651',  # Olympic Green
    '#EE334E',  # Olynmpic Red
]


mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colors)

Save = True
Nbins = 50
file_path = '/groups/icecube/simon/GNN/workspace/storage/Training/stopped_through_classification/train_model_without_configs/osc_next_level3_v2/dynedge_stopped_muon_example/results.csv'
run_name = 'Run1'
output_dir = '/groups/icecube/simon/GNN/workspace/plots'
pdf_filename =  os.path.join(output_dir, 'Classification_Plots.pdf')
pdf = PdfPages(pdf_filename)


try:
    # Load data from CSV
    df = pd.read_csv(file_path)

    # Separate true and false cases based on 'stopped_muon'
    true_cases = df[df['stopped_muon'] == 1]['stopped_muon_pred']
    false_cases = df[df['stopped_muon'] == 0]['stopped_muon_pred']

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(true_cases, bins=Nbins, alpha=0.5, label='True Stopped Muon', color='blue')
    plt.hist(false_cases, bins=Nbins, alpha=0.5, label='False Stopped Muon', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities for Stopped Muon Classification')
    plt.legend()

    # Save the histogram plot to PDF
    if Save:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(df['stopped_muon'], df['stopped_muon_pred'])
    roc_auc = roc_auc_score(df['stopped_muon'], df['stopped_muon_pred'])
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Save the ROC curve plot to PDF
    if Save:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

finally:
    # Ensure the PDF is properly closed
    pdf.close()