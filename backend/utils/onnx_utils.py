import os
import tensorflow as tf
from functools import lru_cache
from tensorflow.python.framework import importer
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def freeze_keras_model2pb(keras_model, pb_filepath, input_variable_name_list=None, output_variable_name_list=None):
    """
    karas 模型转pb
    :param keras_model: 待转换模型
    :param pb_filepath: 模型pb文件保存路径
    :param input_variable_name_list: 输入变量名称列表
    :param output_variable_name_list: 输出变量名称列表
    :return:
    """
    assert hasattr(keras_model, 'inputs'), "the keras model must be built with functional api or sequential"
    # save pb
    if input_variable_name_list is None:
        input_variable_name_list = list()
    if output_variable_name_list is None:
        output_variable_name_list = list()

    if len(input_variable_name_list) == len(keras_model.inputs):
        input_variable_list = input_variable_name_list
    else:
        input_variable_list = ['x%d' % i for i in range(len(keras_model.inputs))]

    input_func_signature_list = [
        tf.TensorSpec(item.shape, dtype=item.dtype, name=name) for name, item in zip(input_variable_list, keras_model.inputs)]
    full_model = tf.function(lambda *x: keras_model(x, training=False))
    # To obtain an individual graph, use the get_concrete_function method of the callable created by tf.function.
    # It can be called with the same arguments as func and returns a special tf.Graph object
    concrete_func = full_model.get_concrete_function(input_func_signature_list)
    # Get frozen ConcreteFunction
    frozen_graph = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_graph.graph.as_graph_def()
    out_idx = 0
    for node in graph_def.node:
        node.device = ""
        if node.name.startswith('Identity'):
            out_idx += 1
    if len(output_variable_name_list) == out_idx:
        output_variable_list = output_variable_name_list
    else:
        output_variable_list = ['y%d' % i for i in range(out_idx)]
    out_idx = 0
    for node in graph_def.node:
        node.device = ""
        if node.name.startswith('Identity'):
            node.name = output_variable_list[out_idx]
            out_idx += 1
    new_graph = tf.Graph()
    with new_graph.as_default():
        importer.import_graph_def(graph_def, name="")

    return tf.io.write_graph(graph_or_graph_def=new_graph,
                             logdir=os.path.dirname(pb_filepath),
                             name=os.path.basename(pb_filepath),
                             as_text=False), input_variable_list, output_variable_list


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    graph = tf.Graph()

    def _imports_graph_def():
        tf.graph_util.import_graph_def(graph_def, name="")

    with graph.as_default():
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)
    return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                tf.nest.map_structure(import_graph.as_graph_element, outputs))


def pb_file_to_concrete_function(pb_file, inputs, outputs, print_graph=False):
    """
    pb_file 转 concrete function
    :param pb_file:
    :param inputs:
    :param outputs:
    :param print_graph:
    :return:
    """
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=inputs,
                                        outputs=outputs,
                                        print_graph=print_graph)
        return graph_def, frozen_func


class OnnxEncoder(object):
    """
    为了加速线上推理，离线生产的tf/keras/pytorch版本的bert模型，均可以通过此方法加载
    需要提前约定好输入list、输出list、词典、maxlen
    """

    def __init__(self, pb_model_file, preprocessor=None, input_variable_names=None, output_variable_names=None, max_cache_len=5000):
        """
        加载onnx存储的pb模型文件，并且封装encoder方法
        :param pb_model_file: tvm优化后的bert模型so库
        :param preprocessor: 数据预处理器，比如文本场景下的tokenizer
        :return: cls
        """
        input_variable_list = input_variable_names or [f'x{i}:0' for i in range(2)]
        output_variable_list = output_variable_names or [f'y{i}:0' for i in range(1)]
        graph, onnx_model = pb_file_to_concrete_function(pb_model_file, input_variable_list, output_variable_list)
        self.preprocessor = preprocessor
        self.onnx_model = onnx_model
        self.graph = graph
        self.lru_encode = self.__inti_lru_encoder(max_cache_len)

    def __inti_lru_encoder(self, max_cache_len):
        """用一条样本激活lru_encoder，因为lru_encoder第一条encode非常慢，构建好缓存后速度恢复正常
        """
        @lru_cache(max_cache_len, typed=False)
        def lru_encode_(text):
            return self.encode(text)
        lru_encode_("欢迎使用ONNX Encoder")
        return lru_encode_

    def __token_encode__(self, text):
        res = self.preprocessor(text)
        return [tf.cast([i], tf.float32) for i in res]

    def encode(self, text):
        return self.onnx_model(self.__token_encode__(text))
