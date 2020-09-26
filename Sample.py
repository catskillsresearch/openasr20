class Sample:

    def __init__(self, _root, _key, _source, _target):
        self.root = _root
        self.key = _key
        self.source=_source
        self.target=_target

    def display(self):
        print('SOURCE')
        self.source.display()
        print('TARGET')
        self.target.display()
