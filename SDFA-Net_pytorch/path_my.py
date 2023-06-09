class Path(object):
    @staticmethod
    def get_path_of(name):
        if name == 'uasol':
            return 'dataset/uasol'
        else:
            raise NotImplementedError