Manuals
-------

Both the User Manual and Technical Manual are Doxygen generated.
Files ending in .dox.in need to be converted by CMake to a dox file.
The CMakeLists.txt in this directory will take any .dox.in file 
and generate it into ${CMAKE_BINARY_DIR}/Doxygen and due to the
fact that Doc/Doxygen/doxygen.config.in contains ${CMAKE_BINARY_DIR}/Doxygen
as an input directory, the generated file will be picked up when
running doxygen within the build folder.

So the conventions to follow are:

1. In this folder keep generated files, ending in .dox.in
2. Technical and User manual go in the obviously named subfolder.
3. Each time you update a file in this folder, you must run make before doxygen. 