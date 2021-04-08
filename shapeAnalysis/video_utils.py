import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as manimation


def mask_array(mask, value_to_mask=0., interval=None):
    if interval is not None:
        return np.ma.masked_where(np.logical_and(mask < interval[1], mask > interval[0]), mask)
    else:
        return np.ma.masked_where(np.logical_and(mask < value_to_mask+1e-3, mask > value_to_mask-1e-3), mask)


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
    segm_plt = ax.matshow(mask_array(stack_segmentations[0]), cmap='autumn', alpha=0.4, interpolation='none', vmin=0, vmax=1)
    plt.subplots_adjust(wspace=0, hspace=0)

    with writer.saving(fig, out_path, nb_frames):
        for i in range(nb_frames):
            timestamp = int(timestampt_at_frame[i])
            timestamp = timestamp if timestamp < length_video else timestamp - 1
            img.set_data(stack_images[timestamp])
            segm_plt.set_data(mask_array(stack_segmentations[timestamp]))
            writer.grab_frame()
