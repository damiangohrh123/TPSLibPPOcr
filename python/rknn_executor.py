from rknnlite.api import RKNNLite
import sys


class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None):
        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            print('Load model failed:', model_path)
            sys.exit(ret)

        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            sys.exit(ret)
        print('done')

        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return self.rknn.inference(inputs=inputs)

    def release(self):
        self.rknn.release()
        self.rknn = None
