import numpy as np

def gaussian_heatmap(shape, keypoint_coordinates, std = 1.5):
    """
        Computes a square gaussian kernel

        :param shape: Shape of the output heatmap
        :param keypoint_coordinates: Location of the keypoint
        :param std: Standard deviation

        :return: Heatmap of shape (1,shape,shape)
    """
    
    # Get the coordinates
    x = keypoint_coordinates[0]
    y = keypoint_coordinates[1]
    
    a = np.arange(0, shape, 1, float)
    b = a[:,np.newaxis]

    # Generate the heatmap
    heatmap_raw = np.exp(-(((a-x)**2)/(2*std**2) + ((b-y)**2)/(2*std**2)))
    
    # Normalize
    heatmap_max = np.amax(heatmap_raw)
    heatmap_normalized = heatmap_raw/heatmap_max
    
    # Get it in the accurate format
    heatmap = np.expand_dims(heatmap_raw, axis=0)
    return heatmap

def gaussian_heatmaps(xs, ys, vs, image_width, image_height, shape=32, std=1.):
    """
        Computes heatmaps from the keypoints
        :param xs: Array of x coordinates for the keypoints
        :param ys: Array of y coordinates for the keypoints
        :param shape: shape of the heatmaps
        :param image_height: Height of the images the keypoints are for
        :param image_width: Width of the images the keypoints are for
        :param std: Standard deviation of the gaussion function used
        
        :return: Heatmaps as numpy arrays of shape (shape, shape, n_keypoints)
    """
    
    # Rescale keypoints coordinates to the heatmaps scale
    # ys
    height_scale = shape/image_height
    ys = ys*height_scale
    # xs
    width_scale = shape/image_width
    xs = xs*width_scale
    
    
    # Render a heatmap for each joint
    if vs[0]!=0:
        heatmaps = gaussian_heatmap(shape, (xs[0],ys[0]))
    else:
        heatmaps = np.zeros((1, shape, shape))
    for i, v in enumerate(vs):
        if i!=0:
            # If the joint is visible, generate a heatmaps
            if v!=0:
                new_heatmap = gaussian_heatmap(shape, (xs[i],ys[i]))
            # Otherwise the heatmaps is composed of zeros
            else:
                new_heatmap = np.zeros((1, shape, shape))
            heatmaps = np.append(heatmaps, new_heatmap, axis=0)

    return heatmaps

def keypoints_from_heatmap(heatmap):
    """Get the coordinates of the max value heatmap - it is the keypoint"""
    max_heatmap = np.amax(heatmap)
    keypoints = np.where(heatmap == max_heatmap)
    if len(keypoints) == 2:
        return keypoints[1][0], keypoints[0][0], max_heatmap
        
    elif len(keypoints) == 3:
        return keypoints[2][0], keypoints[1][0], max_heatmap

def keypoints_from_heatmaps(heatmaps, shape=32, image_height=256, image_width=256):
    """Get the coordinates of the keypoints from the 17 heatmaps"""
    keypoints = []
    for i, heatmap in enumerate(heatmaps):
        x, y, max_heatmap = keypoints_from_heatmap(heatmap)
        if max_heatmap == 0:
            keypoints += [0,0,0]
        else:
            x = x*image_width/shape
            y = y*image_height/shape
            keypoints += [x,y,2]
    return keypoints

def get_xs_ys_vs(keypoints):
    """ Splits MSCOCO keypoints notations from [x0, y0, v0, ...] to [x0, ...], [y0, ...] and [v0, ...] """
    keypoints_array = np.asarray(keypoints)
    xs = np.take(keypoints_array, [3*i for i in range(17)])
    ys = np.take(keypoints_array, [3*i+1 for i in range(17)])
    vs = np.take(keypoints_array, [3*i+2 for i in range(17)])
    return xs, ys, vs

def heatmaps_from_keypoints(keypoints, width, height):
    xs, ys, vs = get_xs_ys_vs(keypoints)
    heatmaps = gaussian_heatmaps(xs, ys, vs, width, height)
    return heatmaps
