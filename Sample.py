class Sample:

    def __init__(self, _key, _source, _target):
        self.key = _key
        self.source=_source
        self.target=_target

    def display(self):
        print('SOURCE')
        self.source.display()
        print('TARGET')
        self.target.display()
