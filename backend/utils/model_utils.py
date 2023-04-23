import importlib
import os
import shutil
from utils.util import get_datetime, check_increase, send_tt_msg


def backup_model(path: str) -> None:
    """
    备份tensorflow模型
    :param path:
    :return:
    """
    dayno = get_datetime(fmt="%Y%m%d")
    source_dir = os.path.dirname(path)

    root_dir = os.path.dirname(source_dir)
    bk_path = os.path.join(root_dir, f"backup_model")
    os.makedirs(bk_path, exist_ok=True)

    target_dir = os.path.join(bk_path, f"{dayno}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print("该备份模型已存在，将覆盖该模型")
    print("备份历史最佳模型成功, 目录:", shutil.copytree(source_dir, target_dir))


def model_online_monitor(core_metrics_constraint, new_core_metrics, history_core_metrics):
    """
    模型安全上线，只有满足指标中的所有要求后才可上线模型，否则不上线模型并且发送告警
    :param core_metrics_constraint: Dict[str, str]核心指标和 约束条件，约束条件为该核心指标的范围如: [-0.01, +inf)，代表该核心指标变化范围不能下降超过1%，无上限
    :param new_core_metrics: Dict[str, float] 当前模型的指标字典
    :param history_core_metrics: Dict[str, float] 该模型的历史指标字典
    :return: 满足条件可以上线则为True 否则告警
    """
    for metric, score_scale_str in core_metrics_constraint.items():
        tmp_score = new_core_metrics[metric]

        if metric not in history_core_metrics:
            history_core_metrics[metric] = {"best_score": tmp_score}

        best_score = history_core_metrics[metric]["best_score"]
        history_core_metrics[metric]["last_core"] = tmp_score
        flag, msg = check_increase(tmp_score, best_score, score_scale_str)
        print(msg)
        if not flag:
            msg = f"{metric}指标异常: {msg}"
            send_tt_msg(msg, [80302421])
            raise Exception(msg)
        else:
            if tmp_score >= best_score:
                history_core_metrics[metric]["best_score"] = tmp_score
    return history_core_metrics


def build_network(network_name, params, model_checkpoint=None):
    """
    构建模型
    :param network_name: 网络具体包名：包路径.脚本名.模型名（类名）
    :param params: 网络参数
    :param model_checkpoint: 如果指定了，表示加载此checkpoint数据
    :return: 构建好的网络模型
    """
    print(f"Build Network: {network_name}")
    package, script, model = ".".join(network_name.split(".")[:-2]), network_name.split(".")[-2], network_name.split(".")[-1]
    package = package or "models"
    net_package = importlib.import_module(f"{package}.{script}")
    net_builder = getattr(net_package, model)
    assert net_builder is not None, f"{network_name} class not exist"
    net = net_builder(**params)
    if model_checkpoint is not None:
        print("load model: {}".format(model_checkpoint))
        net.load_weights(model_checkpoint)
    return net
