# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/11 下午4:32


import warnings
from lxj_utils_sys import ExperimentLogger, ModelCheckpoint, same_seed
from lxj_utils_sys import print_colored, print_config_info
from utils_dataset_metric import get_model_and_dataloader
from utils_porcess import *

warnings.filterwarnings("ignore")
same_seed(2026)



def load_model(args, config, device, valid_X, valid_y, test_X, test_y, num_classes):
    """加载模型"""
    ckp_path = os.path.join("../../checkpoints", args['dataset'], config['model'], args['note']).rstrip('/')
    mode = 'max'
    metric_name = "f1" if args['num_tabs'] == 1 else "map"
    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode=mode, metric_name=metric_name)

    # 获取模型结构
    model, _, _ = get_model_and_dataloader(valid_X, valid_y, test_X, test_y, num_classes, config, args)
    model = modelsaver.load(model, device)[0]
    return model


def main():
    parser = get_parser("test")
    args, args_help = parse_args(parser, is_print_help=False)
    config = load_config_file(args['config'])
    args, config = adjust_system_args(args, config)
    print_config_info({**config, **args}, {**param_description, **args_help})
    # 设备设置
    device = torch.device(args['device'])

    # 路径设置
    ckp_path = os.path.join(args['checkpoint_path'], args['dataset'], config['model'], args['note']).rstrip('/')
    test_path = os.path.join(str(ckp_path), "test")
    print_colored(f"保存位置：{test_path}", 'red')

    # 日志初始化
    logger = ExperimentLogger(json_path=os.path.join(test_path, "result.json"),
                              log_path=os.path.join(test_path, "log.txt"))
    if args['first_check_ratio']:
        pass
    else:
        logger.load()

    # 数据加载
    dataname = ['valid', 'test']
    valid_X, valid_y, test_X, test_y = load_dataset_data(args, dataname)
    
    # 类别数确定
    num_classes = dataset_lib[args['dataset']]['num_classes']

    # 模型和数据加载器
    model, val_loader, test_loader = get_model_and_dataloader(valid_X, valid_y,
                                                              test_X, test_y,
                                                              num_classes, config, args)

    # 加载模型
    mode = 'max'
    metric_name = "f1" if args['num_tabs'] == 1 else "map"
    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode=mode, metric_name=metric_name)
    model = modelsaver.load(model, device)[0]

    # 运行测试
    result, main_metric = evaluate_one_epoch(model, test_loader, device, config, args, num_classes)

    # PR AUC计算
    if args['is_pr_auc']:
        from lxj_utils_sys import compute_pr_result
        pr_auc_result = compute_pr_result(model=model, dataloader=test_loader, task_type="multilabel", average="micro",
                                          fcn_move_to_device=lambda X,y,device:move_to_device((X,y),device,config,args),
                                          fcn_model_forward=lambda model, X:model_forward(model, X, config, args)
                                          )
        logger.record('test.pr_auc', pr_auc_result)
        result['**AUC'] = pr_auc_result['auc']*100

    # 保存结果
    logger.record("load_ratio", args['load_ratio'])
    logger.record('test', result, unpack_dict=True)
    logger.print(f"Test metrics: {result}")

    logger.print("\n\n" + "=" * 20 + " end " + "=" * 20 + "\n")

if __name__ == "__main__":
    main()
    print_colored("test 全部运行结束", "green")