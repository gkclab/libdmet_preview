
# name space 'libdmet_solid' will be deprecated in future, use name space 'libdmet' instead.
import libdmet

for name in dir(libdmet):
    globals()[name] = getattr(libdmet, name)

