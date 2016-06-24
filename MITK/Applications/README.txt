IMPORTANT:
----------

Executables in this folder must only depend on our NifTK Modules
and things that are hence included by transitivity.

DO NOT: Make a dependency on a library, e.g. libopencv_nonfree
that is not included via any of the GUI modules. This WILL break
the packaging procedure, as the CMake fixup_bundle process
works recursively for the GUI, and hence will not pick up said library.

