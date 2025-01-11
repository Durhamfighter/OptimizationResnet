from prune_function import *
from prune import*
from train import *
from utils import *
from evaluate import *
from optimizer import*
import ast
import argparse
import os
import joblib

def main(args):
    print('prune?')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not args.pretrained:
        model = train(args,num_epochs=10)

        os.makedirs(args.model_path, exist_ok=True)
        model_name=os.path.join(args.model_path,'resnet18.joblib')
        joblib.dump(model,model_name)
    else:
        #모델 불르기
        model_name=os.path.join(args.model_path,'resnet18.joblib')
        model = joblib.load(model_name)

        #test loader
        _,test_dataset=get_train_data(args)
        test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
        max_ratio=0.9
        step_ratio=8
        idx2name_module= extract_target_modules(model)
        ## prune 시작
        print('prune?')
        idx=0
        for name,module in model.named_modules():
            if isinstance(module,nn.Conv2d) and 'downsample' not in name:
                step=np.linspace(0,int(module.out_channels*max_ratio),step_ratio,dtype=int)
                steps=step[1:]-step[:-1]
                # steps는 얼마만큼의 filter를 제거할꺼인지 정함.
                for i in range(len(steps)//2): 
                    # 매번 필터를 제거하는양이 달라서 network부름
                    network=joblib.load(model_name).to('cpu')
                    num_channel=module.out_channels- sum(steps[:i+1])
                    print(name,sum(steps[:i+1]))
                    network=prune_step(network,name,num_channel,idx2name_module,index=idx).to(device)
                    print("-*-"*10 + "\n\tPrune network\n" + "-*-"*10)
                    print(network)
                    
                    network_name_v='resenet'+'_'+ name +'_'+str(sum(steps[:i+1]))+'.joblib'
                    network_name=os.path.join(args.model_path,network_name_v)

                    joblib.dump(network,network_name)
                    test(args,network,test_loader,sum(steps[:i+1]))
                idx+=2
                
        






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument


    arg('--pretrained', '--pr', '-pr', type=ast.literal_eval)
    arg('--model_path', '-mp', '--mp', type=str, 
        help='model_path를 지정하세요')
    arg('--seed','-s','--s',type=int,
        help= '시드설정')
    arg('--device', '-d', '--d', type=str, 
         help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--lr, -l, --l', type=float,
        help='learning rate')
    arg('--momentum', '-mo', '--mo', type=float,
        help='momentum')
    arg('--weight_decay','-wd','--wd',type=float,
        help='weight_decay')
    arg('--data_path',type=str,
        help='data_path 지정')


    #args.data_path= ./data
    #args.model_path=./saved
    args = parser.parse_args()
    args.pretrained = True
    args.data_path ='./data'
    args.model_path ='./saved'
    args.seed=42
    args.lr=0.01
    args.momentum=0.9
    args.weight_decay=5e-4
    args.device='cuda'
    #lr=0.01, momentum=0.9, weight_decay=5e-4
    
    main(args)