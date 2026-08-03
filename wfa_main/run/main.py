import time
import warnings


from lxj_utils_sys import ExperimentLogger, ModelCheckpoint, same_seed, LearningRateScheduler, print_dict
from lxj_utils_sys import IncrementalMeanCalculator
from lxj_utils_sys import print_colored, print_config_info
from utils_dataset_metric import get_model_and_dataloader
from utils_porcess import *
from tqdm import tqdm
warnings.filterwarnings("ignore")
same_seed(2026)


def get_criterion(config, args, num_classes):
    """根据模型和任务类型获取损失函数"""
    if args['num_tabs'] > 1:
        return torch.nn.MultiLabelSoftMarginLoss()
    else:
        if args['optim']:
            from timm.loss import LabelSmoothingCrossEntropy
            return LabelSmoothingCrossEntropy(smoothing=0.1)
        else:
            return torch.nn.CrossEntropyLoss()


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, args, epoch, scheduler=None):
    """执行单个训练轮次"""
    model.train()
    loss_cal = IncrementalMeanCalculator()
    for index, cur_data in enumerate(tqdm(train_loader, dynamic_ncols=True)):
        if args['optim'] and scheduler:
            scheduler.step(epoch + index / len(train_loader))

        optimizer.zero_grad()

        # 数据准备
        cur_X, cur_y = move_to_device(cur_data, device, config, args)
        # 前向传播
        outs = model_forward(model, cur_X, config, args)
        # 损失计算
        loss = criterion(outs, cur_y)

        if torch.isnan(loss):
            print_colored("loss is nan", "red")

        # 反向传播
        loss.backward()
        optimizer.step()
        loss_cal.add(loss.detach().cpu().numpy().item())
    return loss_cal.get(round_num=4)





def main():
    parser = get_parser("train")
    args, args_help = parse_args(parser, is_print_help=False)
    config = load_config_file(args['config'])
    args, config = adjust_system_args(args, config)

    print_config_info({**config, **args}, {**param_description, **args_help})
    # 设备设置
    device = torch.device(args['device'])
    # 路径和日志初始化
    ckp_path = os.path.join(args['checkpoint_path'], args['dataset'], config['model'], args['note']).rstrip('/')
    os.makedirs(ckp_path, exist_ok=True)

    logger = ExperimentLogger(json_path=os.path.join(ckp_path, "result.json"),
                              log_path=os.path.join(ckp_path, "log.txt"))

    # 数据加载
    dataname = ['train', args['valid_name']] if not args['test_flag'] else ['valid', 'test']
    train_X, train_y, valid_X, valid_y = load_dataset_data(args, dataname)

    # 类别数确定
    num_classes = dataset_lib[args['dataset']]['num_classes']

    # 模型和数据加载器
    model, train_loader, val_loader = get_model_and_dataloader(train_X, train_y,
                                                               valid_X, valid_y,
                                                               num_classes, config, args)

    # 记录配置
    logger.record('config.config', config, True)
    logger.record('config.args', args, True)

    # 模型和优化器
    model.to(device)
    optimizer = eval(f"torch.optim.{config['optimizer']}")(model.parameters(), lr=float(config['learning_rate']))
    logger.print(str(model.__class__))
    logger.print(str(train_loader.dataset.__class__))
    logger.record("config.model", str(model.__class__))
    logger.record("config.dataset", str(train_loader.dataset.__class__))

    # 损失函数
    criterion = get_criterion(config, args, num_classes)

    # 学习率调度器
    scheduler = None
    if args['optim']:
        scheduler = LearningRateScheduler(optimizer, lr=float(config['learning_rate']),
                                          min_lr=args['min_lr'], warmup_epochs=args['warmup_epochs'],
                                          total_epochs=args['train_epochs'])

    # 模型检查点
    mode = 'max'
    metric_name = "f1" if args['num_tabs'] == 1 else "map"
    modelsaver = ModelCheckpoint(filename=os.path.join(ckp_path, f"model.pth"),
                                 mode=mode, metric_name=metric_name,
                                 max_stagnation_epochs=args['stag_epochs'] if args['optim'] else None)

    # 时间统计
    train_timmer = IncrementalMeanCalculator()
    valid_timmer = IncrementalMeanCalculator()
    metric_best_value = 0

    # 训练日志
    logger.print("\n\n" + "-" * 20 + " start " + "-" * 20 + "\n")

    # 训练循环
    for epoch in range(args['train_epochs']):
        print_colored(f"{epoch+1}/{args['train_epochs']}", "blue")

        # 训练
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config, args, epoch, scheduler)
        logger.record("train.loss", train_loss)
        logger.print(f"epoch {epoch + 1}: train_loss = {train_loss}")
        train_timmer.add(time.time() - start_time)

        # 验证
        start_time = time.time()
        valid_result, main_metric = evaluate_one_epoch(model, val_loader, device, config, args, num_classes)

        logger.record("valid", valid_result, unpack_dict=True)
        logger.print(f"epoch {epoch + 1}: {valid_result}")
        should_stop = modelsaver.save(main_metric, model, epoch + 1, final=(epoch + 1) == args['train_epochs'])
        if main_metric > metric_best_value:
            metric_best_value = main_metric

        # 时间记录
        valid_timmer.add(time.time() - start_time)
        logger.record("time.train", train_timmer.get())
        logger.record("time.valid", valid_timmer.get())
        logger.print(f"epoch {epoch + 1}: time.train = {train_timmer.get():.2f}, time.valid = {valid_timmer.get():.2f}, {metric_name} = {main_metric}")

        if should_stop:
            print("训练早停：达到最大停滞epoch次数")
            break

    logger.print("\n\n" + "=" * 20 + " end " + "=" * 20 + "\n")

if __name__ == "__main__":
    main()
    print_colored("train 全部运行结束", "green")