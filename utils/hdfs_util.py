import logging
import os
import shutil
import subprocess

hadoop_cmd = 'hadoop'


def get_latest_partitions(path, num=1):
    hdfs_files = ls_hdfs_paths(path)
    sorted_partition = sorted([path.strip("/").split("/")[-1] for path in hdfs_files], reverse=True)
    return sorted_partition[:num]


def ls_hdfs_paths(path_pattern):
    logging.info(f"list hdfs paths for {path_pattern}")
    cmd_output = subprocess.check_output([hadoop_cmd, 'fs', '-ls', path_pattern])
    raw_dirs = cmd_output.decode().split('\n')
    raw_dirs = filter(lambda x: not (x.strip().startswith("Found") or x.strip() == "" or "_SUCCESS" in x or x.lower() == "done"), raw_dirs)
    hdfs_dirs = [raw_dir.strip().split(" ")[-1] for raw_dir in raw_dirs]
    logging.info("sub paths:")
    logging.info("\n".join(hdfs_dirs))
    return hdfs_dirs


def get_hdfs_to_local(hdfs_dir, local_dir, max_try=3):
    logging.info(f"get hdfs to local: from {hdfs_dir} to {local_dir}")
    success = False
    for i in range(1, max_try+1):
        try:
            subprocess.check_output(
                [hadoop_cmd, 'fs', '-get', hdfs_dir, local_dir], stderr=subprocess.STDOUT)
            success = True
            break
        except subprocess.CalledProcessError as e:
            logging.info(f'error_info={e.output},error_code={e.returncode}')
            logging.info("{} time failed, try again".format(i))
            rmdir(local_dir)
    if not success:
        raise SystemExit(f"fail get hdfs to local: from {hdfs_dir} to {local_dir}")


def put_local_to_hdfs(local_dir, hdfs_dir, max_try=3):
    logging.info(f"put local to hdfs: from {local_dir} to {hdfs_dir}")
    success = False
    for i in range(1, max_try+1):
        try:
            subprocess.check_output(
                [hadoop_cmd, 'fs', '-put', local_dir, hdfs_dir], stderr=subprocess.STDOUT)
            touch_hdfs_file(hdfs_dir)
            success = True
            break
        except subprocess.CalledProcessError as e:
            logging.error(f'error_info={e.output},error_code={e.returncode}')
            logging.info("{} time failed, try again".format(i))
            rm_hdfs_dir(hdfs_dir)
    if not success:
        raise SystemExit(f"fail put local to hdfs: from {local_dir} to {hdfs_dir}")


def mk_hdfs_dir(hdfs_path):
    if exist_hdfs_path(hdfs_path):
        return
    logging.info(f"mk hdfs dir for {hdfs_path}")
    try:
        subprocess.check_output(
            [hadoop_cmd, 'fs', '-mkdir', '-p', hdfs_path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f'error_info={e.output},error_code={e.returncode}')
        raise SystemExit(e)


def rm_hdfs_dir(hdfs_path):
    if not exist_hdfs_path(hdfs_path):
        return
    logging.info(f"rm hdfs dir for {hdfs_path}")
    try:
        subprocess.check_output(
            [hadoop_cmd, 'fs', '-rm', '-r', hdfs_path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f'error_info={e.output},error_code={e.returncode}')
        msg = e.output
        raise SystemExit(msg)


def exist_hdfs_path(hdfs_path):
    try:
        subprocess.check_output(
            [hadoop_cmd, 'fs', '-test', '-e', hdfs_path], stderr=subprocess.STDOUT)
        logging.info(f"path exists: {hdfs_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f'error_info={e.output},error_code={e.returncode}')
        return False


def get_hdfs_parent_dir(hdfs_path):
    return "/".join(hdfs_path.strip("/").split("/")[:-1]) + "/"


def touch_hdfs_file(hdfs_dir, file_name="_SUCCESS"):
    file_path = f"{hdfs_dir}/{file_name}"
    logging.info(f"touch hdfs file: {file_path}")
    try:
        subprocess.check_output(
            [hadoop_cmd, 'fs', '-touchz', file_path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f'error_info={e.output},error_code={e.returncode}')
        raise SystemExit(e)


def copy_hdfs_file(source_dir, dest_dir, max_try=3):
    logging.info(f"copy hdfs file: from {source_dir} to {dest_dir}")
    success = False
    for i in range(1, max_try+1):
        try:
            subprocess.check_output(
                [hadoop_cmd, 'fs', '-cp', source_dir, dest_dir], stderr=subprocess.STDOUT)
            touch_hdfs_file(dest_dir)
            success = True
            break
        except subprocess.CalledProcessError as e:
            logging.error(f'error_info={e.output},error_code={e.returncode}')
            logging.info("{} time failed, try again".format(i))
            rm_hdfs_dir(dest_dir)
    if not success:
        raise SystemExit(f"fail copy hdfs file: from {source_dir} to {dest_dir}")


def cat_hdfs_path(hdfs_path):
    if not exist_hdfs_path(hdfs_path):
        logging.info(f"{hdfs_path} not exists")
        return None
    logging.info(f"cat path: {hdfs_path}")
    cmd_output = subprocess.check_output([hadoop_cmd, 'fs', '-cat', hdfs_path])
    lines = cmd_output.decode().split('\n')
    lines = list(filter(lambda x: x != "", lines))
    logging.info("content:")
    logging.info("\n".join(lines))
    return lines


def rmdir(path):
    if os.path.exists(path):
        print("remove path: {}".format(path))
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    else:
        print("{} not exist, no need rm".format(path))
