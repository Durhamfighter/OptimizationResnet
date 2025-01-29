from prune_function import *
import torch
import torch.nn as nn

def prune_step(network,name,num_channel,idx2name_module,index):
    # 현재 이름에 따라 filters_to_keep 정함
    if 'layer' in name:
        name_lst=name.split('.')
        current_block=getattr(network,name_lst[0])[int(name_lst[1])]
        current_layer=getattr(current_block,name_lst[2])
        filters_to_keep=sort_filter(current_layer,num_channel)    
        # 그다음 레이어
        next_name,next_model = idx2name_module[index+2] # prune enxt

        if isinstance(next_model,nn.Conv2d):
            next_lst=next_name.split('.')
            next_current_block=getattr(network,next_lst[0])[int(next_lst[1])]
            next_current_layer=getattr(next_current_block,next_lst[2])

            #현재레이어 프룬
            current_block.prune_now(current_layer,filters_to_keep)
            current_layer=getattr(current_block,name_lst[2])
            #next layer프룬
            next_current_block.prune_next(next_current_layer,filters_to_keep)
            next_current_layer=getattr(next_current_block,next_lst[2])

            if current_block is not next_current_block:
                # 객체 memory 가바뀌었으므로 다시해야함.
                ## 중요함함
                current_block.adjust_downsample(current_layer,filters_to_keep)
                next_current_block.adjust_downsample(next_current_layer,filters_to_keep)                                    
        else:
            # Linear인 경우
            current_block.prune_now(current_layer,filters_to_keep)
            current_block.adjust_downsample(current_layer,filters_to_keep)
            next_current_layer=getattr(network,next_name)
            next_new_layer=adjust_new_linear(next_current_layer,filters_to_keep)
            setattr(network,next_name,next_new_layer)
    else:
        current_layer=getattr(network,name)
        filters_to_keep=sort_filter(current_layer,num_channel)

        # Pruned 된 레이어로 교체
        new_layer= prune_layer(current_layer,filters_to_keep)
        setattr(network, name, new_layer)

        #batch layer 교체
        batch_name,batch_modoule=idx2name_module[index+1]
        current_batch=getattr(network,batch_name)
        new_batch_layer = adjust_batch_layer(current_batch,filters_to_keep)
        setattr(network,batch_name,new_batch_layer)

        #다음 cnn교체
        next_name,next_model = idx2name_module[index+2]
        next_lst=next_name.split('.')
        next_current_block=getattr(network,next_lst[0])[int(next_lst[1])]

        next_current_layer=getattr(next_current_block,next_lst[2])
        next_current_block.prune_next(next_current_layer,filters_to_keep)
        next_current_layer=getattr(next_current_block,next_lst[2])

        next_current_block.adjust_downsample(next_current_layer,filters_to_keep)
    return network