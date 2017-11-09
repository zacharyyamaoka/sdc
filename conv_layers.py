import numpy as np

def conv_forward(x, filter, bias, padding, stride):

    N, C, W, H = x.shape
    F, C, WW, HH = filter.shape

    out_width = ((W - WW + 2 * padding) / stride) + 1
    out_height = ((H - HH + 2 * padding) / stride) + 1

    out = np.zeros((N, F, out_height, out_width), dtype=x.dtype)

    cols = im2col(x, WW, HH, padding, stride)


    # turn 10, 3 x 3 x 3 filters into 10 by 27 for easy dot product
    filter = filter.reshape(F, -1)
    # for 10 filters on cifar 10 this is a (10,27) x (27,1032), each 27 filter being multiple
    # by the 27 corresponding cells on input volume 1032 times.For multiple images this is just (10, 27)  x (27, 103200)
    activation_map = filter.dot(cols) + b.reshape(-1, 1)

    # turn (10, 103200) activation map into (10, 32, 32, 1000)
    activation_map = activation_map.reshape(F, out[2], out[3], N)

    # turn (10, 32, 32, 1000) into (1000, 10, 32, 32)
    out = activation_map.transpose(3, 0, 1, 2)

    cache = (x, filter, bias, padding, stride, cols)
    return out, cache

def conv_backward(dout, cache):

    x, filter, bias, padding, stride, cols = cache
    N, C, W, H = x.shape
    F, C, WW, HH = filter.shape
    out_width = ((W - WW + 2 * padding) / stride) + 1
    out_height = ((H - HH + 2 * padding) / stride) + 1

    # derivate of bias is just one * dout. Sum over volume channels to account for effect on
    # all examples and all cells per channel
    db = np.sum(dout, axis=(0, 2, 3))

    dout = dout.transpose(1, 2, 3 0).reshape(F, -1)
    # use summing property of dot product to account for all the x's the filter w
    # contributed too, multiplied by the contribution of each of thoose x's to the output
    dw = dout.dot(cols.T).reshape(filter.shape)


    return dx, dw


def im2col_indices(x, filter_height, filter_width, padding=1, stride=1):

    N, C, W, H = x.shape

    output_W = ((W - filter_width + 2 * padding) / stride) + 1
    output_H = ((H - filter_height + 2 * padding) / stride) + 1

    # Create indices for rows
    # [1, 2, 3]
    # [1, 2, 3]
    # [1, 2, 3]
    i0 = np.repeat(np.arange(filter_height), filter_width)

    # Tile into depth of input volume
    # [1, 2, 3] [1, 2, 3] [1, 2, 3]
    # [1, 2, 3] [1, 2, 3] [1, 2, 3]
    # [1, 2, 3] [1, 2, 3] [1, 2, 3]
    i0 = np.tile(i0, C)

    # Determine offset for each filter index. Right now the filter is just for the
    # the base case in the top left corner. Use the offset to move it around
    i1 = stride * np.repeat(np.arange(H), W)

    # same as for rows but now for colums
    j0 = np.repeat(np.arange(filter_width), filter_height)
    j0 = np.tile(j0, C)
    j1 = stride * np.repeat(np.arange(W), H)

    # Use broadcasting to apply the offset to the indicies. Zero will be added to the base case.
    # so it remains in the same spot, while np.arange(output_dim) so, 1, 2, 3 will be added to the rest.
    # When stride is not 1, then the offset is mroe like, 0, 2, 4, 6, ect.. for an stride of 2
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # create indices for depth, for depth of 3, you need 1, 2, 3 repeated for the size of the filter so that each
    k = np.repeat(np.arange(C), filter_width * filter_height).reshape(-1, 1)
    print(k.shape, i.shape, j.shape)
    return (k, i, j)

def im2col(x, field_height, field_width, padding=1, stride=1):

    # pad the width and height
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = im2col_indices(x, field_height, field_width, padding, stride)

    #index matrix by slicing over all examples
    cols = x_padded[:, k, i, j]

    C = x.shape[1]
    # stack all examples togther
    # for a 3 by 3 filter on 32 by 32 image it was previously (100, 27, 1032) now it is (27, 1032 by 1032 by 1032 ect..)
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols

def col2im(cols, x_shape, field_height, field_width, padding=1):

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

     k, i, j = im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
