import os


class LibrarySpec(object):
    def __init__(self, d):
        for key in d:
            self.__setattr__(key, d[key])

    def __repr__(self):
        return "{0}".format(self.name)


# get top directory
filepath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))

PROJECT_DIR = filepath

LIBRARIES = [
    {
        "name": "metal",
        "conda_env": "metal",
        "parallel": True,
        "path": "{0}/projects/metal/".format(PROJECT_DIR),
        "enabled": True,
        "deps": ["numpy", "torch"]
    },
]

LIBRARIES = [LibrarySpec(lib) for lib in LIBRARIES]

print(LIBRARIES)
