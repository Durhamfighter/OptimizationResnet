import torch
import torch.nn as nn

def prune_layer(layer, filters_to_keep):

    ### 대체할 레이어 생성
    new_layer=nn.Conv2d(
            in_channels = layer.in_channels,
            out_channels = len(filters_to_keep),  
            kernel_size = layer.kernel_size,
            stride = layer.stride,
            padding = layer.padding,
            bias = (layer.bias is not None))
    ### keep 할 필터들 웨이트 옮겨주기
    new_layer.weight.data=layer.weight.data[filters_to_keep,:,:,:]
    if layer.bias is not None:
        new_layer.bias.data=layer.bias.data[filters_to_keep]
    return new_layer


def adjust_next_layer(layer, filters_to_keep):
    
    ### 조정할 다음 레이어 생성
    adjust_layer=nn.Conv2d(
                in_channels=len(filters_to_keep),
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding = layer.padding,
                bias= (layer.bias is not None)
    )
    
    adjust_layer.weight.data=layer.weight.data[:,filters_to_keep,:,:]
    if layer.bias is not None:
        adjust_layer.bias.data=layer.bias.data[filters_to_keep]
    return adjust_layer

def adjust_batch_layer(batch_layer,filters_to_keep):

    ### 조정할 배치 레이어 만들기
    new_batch=nn.BatchNorm2d(
                num_features=len(filters_to_keep),
                eps = batch_layer.eps,
                momentum = batch_layer.momentum,
                affine = batch_layer.affine,
                track_running_stats = batch_layer.track_running_stats
                )
    with torch.no_grad():
        new_batch.weight.data=batch_layer.weight.data[filters_to_keep]
        new_batch.bias.data = batch_layer.bias.data[filters_to_keep]
        new_batch.running_mean.data=batch_layer.running_mean.data[filters_to_keep]
        new_batch.running_var.data=batch_layer.running_var[filters_to_keep]
    return new_batch


def adjust_new_linear(layer,filters_to_keep):

    ### 조정할 linear 레이어 만들기
    new_linear = nn.Linear(
                in_features = len(filters_to_keep),
                out_features = layer.out_features,
                bias = (layer.bias is not None)
    )
    ### weight 옮기기
    new_linear.weight.data = layer.weight.data[:, filters_to_keep]
    if layer.bias is not None:
        new_linear.bias.data = layer.bias.data[:]

    return new_linear

def sort_filter(layer,steps):
    L1_weight = torch.sum(abs(layer.weight), dim=(1, 2, 3))
    sorted_indices = torch.argsort(L1_weight, descending=True)

    filter_to_keep = sorted_indices[:steps]

    return filter_to_keep



