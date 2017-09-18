def _check_convolution_layer(filter_size,filter_num,zero_padding,stride):
    filter_height,filter_width=filter_size
    if not isinstance(filter_height,int):
        raise ValueError('filter_height must be int')
    if not isinstance(filter_width,int):
        raise ValueError('filter_width must be int')
    if not isinstance(filter_num,int):
        raise ValueError('filter_num must be int')
    if not isinstance(zero_padding,tuple) and not isinstance(zero_padding,list):
        raise ValueError('zero_padding must be tuple() or int')
