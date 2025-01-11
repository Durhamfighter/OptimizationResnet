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
        len(filters_to_keep)
        # 현재 레이어 프룬하자
        new_layer=  prune_layer(current_layer,filters_to_keep)
        setattr(current_block, name_lst[2], new_layer)

        # 현재 레이어 해당 block에 downsample 이없고 residual connection의 차원이 안맞을시 추가.
        if current_block.downsample is None and current_block.conv1.in_channels!=current_block.conv2.out_channels:
            conv=nn.Conv2d(
                    in_channels=current_block.conv1.in_channels,
                    out_channels=current_block.conv2.out_channels,
                    kernel_size=1,
                    bias=False
                    )
            setattr(current_block,'downsample',conv)
        ## sequential 부분도 생각해야함
        elif isinstance(current_block.downsample,nn.Sequential):
            if name_lst[2]=='conv2':
                current_block.downsample[0] = prune_layer(current_block.downsample[0],filters_to_keep)
                current_block.downsample[1]= adjust_batch_layer(current_block.downsample[1],filters_to_keep)
                
        # batch 레이어 바꿔야지
        batch_name,_ = idx2name_module[index+1]
        batch_name_lst=batch_name.split('.')
        batch_current_block=getattr(network,batch_name_lst[0])[int(batch_name_lst[1])]
        batch_current_layer=getattr(batch_current_block,batch_name_lst[2])
        new_batch_layer = adjust_batch_layer(batch_current_layer,filters_to_keep)
        setattr(batch_current_block,batch_name_lst[2],new_batch_layer)

        # 그다음 레이어
        next_name,next_model = idx2name_module[index+2]
        if isinstance(next_model,nn.Conv2d):
            next_lst=next_name.split('.')
            next_current_block=getattr(network,next_lst[0])[int(next_lst[1])]
            next_current_layer=getattr(next_current_block,next_lst[2])                    
            next_new_layer=adjust_next_layer(next_current_layer,filters_to_keep)
            setattr(next_current_block, next_lst[2], next_new_layer)
            if next_current_block.downsample is None and next_current_block.conv1.in_channels!=next_current_block.conv2.out_channels:
                conv=nn.Conv2d(
                        in_channels=next_current_block.conv1.in_channels,
                        out_channels=next_current_block.conv2.out_channels,
                        kernel_size=1,
                        bias=False
                        )
                setattr(next_current_block,'downsample',conv)

            elif isinstance(next_current_block.downsample,nn.Sequential):
                print(name_lst[0],next_name[0])
                if name_lst[0]!=next_lst[0]:  # 같은 레이어면 바꿀 필요없음 아무런 이상이없음..
                    downsample_cnn=next_current_block.downsample[0]
                    downsample_newcnn=adjust_next_layer(downsample_cnn,filters_to_keep)
                    next_current_block.downsample[0] = downsample_newcnn
        else:
            # Linear인 경우
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
        next_new_layer=adjust_next_layer(next_current_layer,filters_to_keep)
        setattr(next_current_block, next_lst[2], next_new_layer)
        conv=nn.Conv2d(
                    in_channels=next_current_block.conv1.in_channels,
                    out_channels=next_current_block.conv2.out_channels,
                    kernel_size=1,
                    bias=False
                    )
        setattr(next_current_block,'downsample',conv)
    return network