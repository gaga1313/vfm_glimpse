import numpy as np
import torch
import torch.nn.functional as F


def transformer(input_fmap, theta, H, W, scale=1.0, grids=False, reverse=False):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localization_net: takes the original image as input and outputs
        the parameters of the affine transformation that should be applied
        to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that
        correspond to a set of points where the input should be sampled
        to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
        and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
        transformer layer is at the beginning of architecture. Should be
        a tensor of shape (B, C, H, W) for PyTorch.
    - theta: affine transform tensor of shape (B, 6). Permits cropping,
        translation and isotropic scaling. Initialize to identity matrix.
        It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, C, H, W).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
            (https://arxiv.org/abs/1506.02025)
    """
    # generate grids of same size or upsample/downsample if specified
    batch_tgrids, batch_grids, mats = affine_grid_generator(H, W, theta, reverse, scale)

    x_s = batch_tgrids[:, 0, :, :]
    y_s = batch_tgrids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)
    
    if grids:
        return out_fmap, batch_tgrids
    else:
        return out_fmap


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a 4D tensor image.
    Input
    -----
    - img: tensor of shape (B, C, H, W) for PyTorch
    - x: tensor of shape (B, H, W)
    - y: tensor of shape (B, H, W)
    Returns
    -------
    - output: tensor of shape (B, C, H, W)
    """
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = torch.arange(0, batch_size, device=x.device)
    batch_idx = batch_idx.view(batch_size, 1, 1)
    b = batch_idx.expand(-1, height, width)

    # Convert to long for indexing
    x = x.long()
    y = y.long()
    b = b.long()

    # PyTorch indexing: [B, C, H, W]
    return img[b, :, y, x]


def generate_grid(height, width, scale):
    x_H = np.linspace(-scale, scale, height)
    x_W = np.linspace(-scale, scale, width)

    axes = [x_H, x_W]
    packed_grid = np.meshgrid(*axes)

    np_H = packed_grid[0].reshape([-1, 1])
    np_W = packed_grid[1].reshape([-1, 1])

    np_grid = np.concatenate([np_H, np_W], axis=1)

    grid = torch.tensor(np.float32(np_grid))

    return grid


def affine_grid_generator(height, width, theta, reverse, scale):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = theta.shape[0]

    theta = theta.float()

    base_grid = generate_grid(height, width, scale)
    grid = base_grid.unsqueeze(0).unsqueeze(-1)
    
    # Move to same device as theta
    if theta.is_cuda:
        grid = grid.cuda()

    b = theta[:, :2]
    theta_rot = theta[:, 2:3]
    theta_s = theta[:, 3:5]
    theta_z = theta[:, 5:6]

    cos_rot = torch.cos(theta_rot).view(-1, 1, 1)
    sin_rot = torch.sin(theta_rot).view(-1, 1, 1)

    A_rot1 = torch.cat([cos_rot, -sin_rot], dim=-1)
    A_rot2 = torch.cat([sin_rot, cos_rot], dim=-1)
    A_rot = torch.cat([A_rot1, A_rot2], dim=1)

    # Note the +1.0 in scale. This makes x=0 represent the identity
    # transformation
    A_s = torch.diag_embed(theta_s + 1.0)

    theta_z = theta_z.view(-1, 1, 1)
    A_z1 = torch.cat([torch.ones_like(theta_z), theta_z], dim=-1)
    A_z2 = torch.cat([torch.zeros_like(theta_z), torch.ones_like(theta_z)], dim=-1)
    A_z = torch.cat([A_z1, A_z2], dim=1)

    A = torch.matmul(A_z, A_rot)
    A = torch.matmul(A_s, A)

    A = A.view(-1, 1, 2, 2)
    b = b.view(-1, 1, 2, 1)

    if not reverse:
        grid = torch.matmul(A, grid) + b
    else:
        grid = torch.matmul(A, grid - b)
    
    grid = grid.squeeze(-1)
    grid = grid.transpose(1, 2)

    # reshape to (num_batch, 2, H, W)
    batch_grids = grid.view(num_batch, 2, height, width)

    return batch_grids, base_grid, [A_rot, A_s, A_z, b]


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, C, H, W) layout for PyTorch.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = img.shape[2]
    W = img.shape[3]
    max_y = H - 1
    max_x = W - 1
    zero = 0

    # rescale x and y to [0, W-1/H-1]
    x = x.float()
    y = y.float()
    x = 0.5 * ((x + 1.0) * (max_x - 1))
    y = 0.5 * ((y + 1.0) * (max_y - 1))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = torch.clamp(x0, zero, max_x)
    x1 = torch.clamp(x1, zero, max_x)
    y0 = torch.clamp(y0, zero, max_y)
    y1 = torch.clamp(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)

    # compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out


def transformer_pytorch_native(input_fmap, theta, H, W, scale=1.0, grids=False, reverse=False):
    """
    Alternative PyTorch implementation using native grid_sample function.
    This is more efficient but may have slightly different behavior.
    """
    num_batch = theta.shape[0]
    
    # Generate affine transformation matrix
    theta = theta.float()
    
    # Extract transformation parameters
    b = theta[:, :2]  # translation
    theta_rot = theta[:, 2:3]  # rotation
    theta_s = theta[:, 3:5]  # scale
    theta_z = theta[:, 5:6]  # shear
    
    # Build transformation matrix
    cos_rot = torch.cos(theta_rot)
    sin_rot = torch.sin(theta_rot)
    
    # Rotation matrix
    rot_matrix = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=1),
        torch.stack([sin_rot, cos_rot], dim=1)
    ], dim=1)
    
    # Scale matrix
    scale_matrix = torch.diag_embed(theta_s + 1.0)
    
    # Shear matrix
    shear_matrix = torch.stack([
        torch.stack([torch.ones_like(theta_z), theta_z], dim=1),
        torch.stack([torch.zeros_like(theta_z), torch.ones_like(theta_z)], dim=1)
    ], dim=1)
    
    # Combine transformations
    A = torch.matmul(shear_matrix, rot_matrix)
    A = torch.matmul(scale_matrix, A)
    
    # Add translation
    if not reverse:
        A = torch.cat([A, b.unsqueeze(-1)], dim=2)
    else:
        A = torch.cat([A, -b.unsqueeze(-1)], dim=2)
    
    # Pad to 3x3 matrix for grid_sample
    batch_size = A.shape[0]
    A_padded = torch.zeros(batch_size, 3, 3, device=A.device)
    A_padded[:, :2, :2] = A[:, :2, :2]
    A_padded[:, :2, 2] = A[:, :2, 2]
    A_padded[:, 2, 2] = 1.0
    
    # Generate sampling grid
    grid = F.affine_grid(A_padded[:, :2, :], (batch_size, input_fmap.shape[1], H, W), align_corners=False)
    
    # Apply transformation
    output = F.grid_sample(input_fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    if grids:
        return output, grid.permute(0, 3, 1, 2)  # Convert to (B, 2, H, W) format
    else:
        return output
