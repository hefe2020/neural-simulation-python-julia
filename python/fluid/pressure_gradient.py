def pressure_gradient_staggered(p, h=1.0):
    p_field_x = np.column_stack([p, p[:, 0]])
    
    p_field_y = np.vstack([p, p[0, :]])

    p_grid_x = (p_field_x[:, 1:] - p_field_x[:, :-1]) / h
    p_grid_y = (p_field_y[1:, :] - p_field_y[:-1, :]) / h

    return p_grid_x, p_grid_y

