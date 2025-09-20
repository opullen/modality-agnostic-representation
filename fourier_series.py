import torch

def _linear_interpolate_torch(old_x, old_y, new_x):
    idx = torch.searchsorted(old_x, new_x) - 1
    idx = idx.clamp(min=0, max=old_x.size(0)-2)
    x0 = old_x[idx]
    x1 = old_x[idx+1]
    y0 = old_y[idx]
    y1 = old_y[idx+1]
    t = (new_x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def dst_type_1(x, norm=None):
    x = x.reshape(-1, 1)
    N = x.shape[0]

    if norm=='ortho':
        scaling_factor = torch.sqrt(2 / (N + 1))
    else: scaling_factor = 2.0
    
    k = torch.arange(N).reshape(1, N)
    n = torch.arange(N).reshape(N, 1)
    sine_matrix = torch.sin(((torch.pi * (n + 1) * (k + 1)) / (N + 1)))
    X = scaling_factor * torch.matmul(sine_matrix.float(), x.float())
    return X

def normalize_resize_rotate_torch(x, y, norm_length=209):
    # Scale and shift
    y_height = y.max() - y.min()
    scaling_factor = (norm_length - 1) / y_height
    x = x * scaling_factor
    y = y * scaling_factor
    x = x - x[0]
    y = y - y[0]

    # Create old/new indices just like NumPy code
    original_length = y.numel()
    old_x_indices = torch.linspace(0, original_length - 1, original_length, dtype=torch.float32)
    new_indices = torch.linspace(0, original_length - 1, norm_length, dtype=torch.float32)

    # Interpolate to new length
    x_interpolated = _linear_interpolate_torch(old_x_indices, x, new_indices)
    y_interpolated = _linear_interpolate_torch(old_x_indices, y, new_indices)

    # Rotate
    y_rotated, t_rotated = rotate_curve_torch(x_interpolated, y_interpolated)
    return y_rotated, t_rotated

def rotate_curve_torch(x, y):
    x_translated = x - x[0]
    y_translated = y - y[0]
    angle = torch.arctan2(y_translated[-1], x_translated[-1]) - torch.deg2rad(torch.tensor(90.0, dtype=torch.float32))

    cos_angle = torch.cos(-angle)
    sin_angle = torch.sin(-angle)
    rotation_matrix = torch.tensor([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]], dtype=torch.float32)
    coordinates = torch.vstack((x_translated, y_translated))
    rotated_coordinates = torch.matmul(rotation_matrix, coordinates)

    x_rotated = rotated_coordinates[0] + x[0]
    y_rotated = rotated_coordinates[1] + y[0]

    return x_rotated, y_rotated

def normalize_x_f(x_f, normalize):
        reshaped_coefficient_array = x_f.squeeze(1)
        coefficent_array = (reshaped_coefficient_array - normalize['coefficient_means']) / normalize['coefficient_sds']
        return coefficent_array