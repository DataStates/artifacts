from ctypes import cdll

def load(filename):
    if filename not in load.libraries:
        load.libraries[filename] = cdll.LoadLibrary(filename)
    return load.libraries[filename]

load.libraries = dict()
