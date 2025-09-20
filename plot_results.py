import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import segmentation


def process_seg(seg):
    seg[seg > 0] = 1
    return seg.astype(np.intp, copy=False)

def binarize_labels(dxa_labels):
    labels = {}
    # Scoliosis
    if dxa_labels['scoliosis'] == 'Scoliosis':
        labels['scoliosis'] = 1
    elif dxa_labels['scoliosis'] == 'No Scoliosis':
        labels['scoliosis'] = 0
    else:
        labels['scoliosis'] = -1 
    # Number of curves
    if dxa_labels['n_curves'] == 2:
        labels['n_curves'] = 1
    elif dxa_labels['n_curves'] == 1:
        labels['n_curves'] = 0
    else:
        labels['n_curves'] = -1
    # Largest angle direction
    if dxa_labels['largest_angle_direction'] == 'Left':
        labels['largest_angle_direction'] = 0
    elif dxa_labels['largest_angle_direction'] == 'Right':
        labels['largest_angle_direction'] = 1
    else:
        labels['largest_angle_direction'] = -1
    # Largest angle location
    if dxa_labels['largest_angle_location'] == 'Thoracic':
        labels['largest_angle_location'] = 0
    elif dxa_labels['largest_angle_location'] == 'Lumbar':
        labels['largest_angle_location'] = 1
    else:
        labels['largest_angle_location'] = -1

    return labels

class ToText:
    def __init__(self):
        super().__init__()
        
        self.scoliosis = ["No", "Yes"] + ["Ignored"]
        self.n_curves = ["1", "2+"] + ["Ignored"]
        self.direction = ["Left", "Right"] + ["Not\nLabelled"]
        self.location = ["Thoracic", "Lumbar"] + ["Ignored"]

    def decode(self, labels, preds, xray=False):
        labels_str = [self.scoliosis[labels[0]], self.n_curves[labels[1]], self.direction[labels[2]], self.location[labels[3]]]
        if xray:
            preds_str = [self.scoliosis[preds[0]], self.n_curves[preds[1]], self.direction[-1], self.location[preds[3]]]
        else:
            preds_str = [self.scoliosis[preds[0]], self.n_curves[preds[1]], self.direction[preds[2]], self.location[preds[3]]]
        return labels_str, preds_str
    

def plot_results(img, seg, gt_labels, pred_labels, tasks, save_path=None, modality=''):
    h, w = img.shape[:2]
    aspect = w / h 
    # Fig size height and width
    fig_height = 3
    fig_width  = fig_height * (aspect + 1)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(1, 2, width_ratios=[aspect, 1], wspace=0.05)

    # Left side: image and segmentation overlay
    ax_img = fig.add_subplot(gs[0])
    if modality == 'xray':
        
        ax_img.imshow(img, cmap='gray')  
        ax_img.scatter(seg[:, 0], seg[:, 1], c='yellow', s=1)
    else:
        ax_img.imshow(img, cmap="gray")
        ax_img.contour(segmentation.find_boundaries(seg, mode="outer"), levels=[0.95], colors="yellow", linewidths=0.3)
    ax_img.axis("off")
    ax_img.set_title("Input Image", fontsize=9, pad=3)
    
    # Right side
    ax_tab = fig.add_subplot(gs[1])
    ax_tab.axis("off")

    # Define vertical positions in axis coordinates
    y_top = 0.90          # header text
    y_bottom = 0.10       # bottom line lower margin
    table_height = y_top - y_bottom
    n = len(tasks)
    header_height = 0.10
    body_height = table_height - header_height
    row_height = body_height / n

    # X positions for cols
    x_task = 0.05
    x_gt   = 0.50
    x_pred = 0.80
    x_sep  = 0.45  # vertical line between Task and GT/Pred

    # Draw header text
    ax_tab.text(x_task, y_top, "Task", va="center", ha="left", fontsize=8, weight="bold")
    ax_tab.text(x_gt,   y_top, "GT",   va="center", ha="left", fontsize=8, weight="bold")
    ax_tab.text(x_pred, y_top, "Pred", va="center", ha="left", fontsize=8, weight="bold")

    # Draw horizontal line under header
    y_line_header = y_top - (header_height / 2)
    ax_tab.plot([0, 1], [y_line_header, y_line_header], color="black", linewidth=0.8)

    # Draw each row of text
    for i in range(n):
        y_row = y_line_header - (row_height * (i + 0.5))
        ax_tab.text(x_task,  y_row, tasks[i],       va="center", ha="left", fontsize=8)
        ax_tab.text(x_gt,    y_row, gt_labels[i],    va="center", ha="left", fontsize=8)
        color = "red" if pred_labels[i] != gt_labels[i] else "black"
        weight = "bold" if pred_labels[i] != gt_labels[i] else "normal"
        ax_tab.text(x_pred,  y_row, pred_labels[i],  va="center", ha="left",
                    fontsize=8, color=color, weight=weight)

    # 4) Draw bottom horizontal line
    y_line_bottom = y_bottom + (header_height / 2)
    ax_tab.plot([0, 1], [y_line_bottom, y_line_bottom], color="black", linewidth=0.8)

    # Draw single vertical line separating Task from GT/Pred
    ax_tab.plot([x_sep, x_sep], [y_line_bottom, y_line_header], color="black", linewidth=0.8)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
