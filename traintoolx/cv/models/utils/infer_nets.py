

import paddle


class PostProcessor(paddle.nn.Layer):
    def __init__(self, model_type):
        super(PostProcessor, self).__init__()
        self.model_type = model_type

    def forward(self, net_outputs):
        if self.model_type == 'classifier':
            outputs = paddle.nn.functional.softmax(net_outputs, axis=1)
        else:
            # label_map [NHW], score_map [NHWC]
            logit = net_outputs[0]
            outputs = paddle.argmax(logit, axis=1, keepdim=False, dtype='int32'), \
                      paddle.transpose(paddle.nn.functional.softmax(logit, axis=1), perm=[0, 2, 3, 1])

        return outputs


class InferNet(paddle.nn.Layer):
    def __init__(self, net, model_type):
        super(InferNet, self).__init__()
        self.net = net
        self.postprocessor = PostProcessor(model_type)

    def forward(self, x):
        net_outputs = self.net(x)
        outputs = self.postprocessor(net_outputs)

        return outputs
