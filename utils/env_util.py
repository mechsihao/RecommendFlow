import subprocess


def activate_hadoop_env(script: str = "/home/notebook/data/group/sihao_work/programs/scripts/hadoop_conf.sh"):
    import os
    import tensorflow_io as tfio
    # 对于一些动态链接库找不到的，可以用软连接工具将其软连接到/usr/lib/x86_64-linux-gnu/ 中
    if os.path.exists(script):
        pipe = subprocess.Popen(". %s; env" % script, stdout=subprocess.PIPE, shell=True)
    else:
        raise FileNotFoundError(f"script: {script} not found")
    output = pipe.communicate()[0]
    env_new = {line.split("=")[0]: line.split("=")[1] for line in output.decode("utf-8").splitlines() if '=' in line}
    os.environ.update(env_new)
    os.environ.update({"TFIO_VERSION": tfio.version.VERSION})
