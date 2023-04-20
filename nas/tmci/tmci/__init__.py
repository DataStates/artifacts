
def get_library_dir():
    import os
    return os.path.join(os.path.dirname(__file__), '..')

def get_library():
    import glob
    lib = glob.glob(get_library_dir() + '/_tmci_ops.*.so')
    return lib[0].split('/')[-1]

