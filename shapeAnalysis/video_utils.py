import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import vigra
import numpy as np

import matplotlib.animation as manimation

import segmfriends.vis as segm_vis

segm_plot_kwargs = segm_vis.segm_plot_kwargs
rand_cm = segm_vis.rand_cm

def save_segmentation_video(stack_images, stack_segmentations,
                   out_path,
                   nb_frames=100):
    # Prepare video:
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Cell Shape Detection', artist='alberto.bailoni@iwr.uni-heidelberg.de')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    length_video = stack_images.shape[0]
    assert length_video == stack_segmentations.shape[0]

    timestampt_at_frame = np.linspace(0, length_video, nb_frames)

    fig, ax = plt.subplots(ncols=1, nrows=1,
                         figsize=(30, 7))
    for a in fig.get_axes():
        a.axis('off')
    img = ax.matshow(stack_images[0], cmap="gray", vmin=0, vmax=255,
                         interpolation='none')
    segm_plt = ax.matshow(segm_vis.mask_the_mask(stack_segmentations[0]), cmap='autumn', alpha=0.4, interpolation='none', vmin=0, vmax=1)
    plt.subplots_adjust(wspace=0, hspace=0)

    with writer.saving(fig, out_path, nb_frames):
        for i in range(nb_frames):
            timestamp = int(timestampt_at_frame[i])
            timestamp = timestamp if timestamp < length_video else timestamp - 1
            img.set_data(stack_images[timestamp])
            segm_plt.set_data(segm_vis.mask_the_mask(stack_segmentations[timestamp]))
            writer.grab_frame()
