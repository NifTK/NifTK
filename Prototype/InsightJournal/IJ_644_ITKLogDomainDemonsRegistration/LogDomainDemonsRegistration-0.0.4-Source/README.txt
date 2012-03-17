A few notes without special order.

This project is meant to be compiled with cmake.

Unless you know what you are doing, it is recommended to use ITK version 3.14 or greater with the following compilation options:
  - ITK_USE_REVIEW = ON
  - BUILD_SHARED_LIBS = ON
  - CMAKE_BUILD_TYPE = Release


Then the project itself can be compiled as follows:
  cd symdemons
  cmake -DITK_DIR=_path_to_itk_ -DMATLAB_ROOT=_path_to_matlab_ -DCMAKE_BUILD_TYPE=Release
  make -j2

MATLAB_ROOT is optionnal for the ITK code to compile. You may use it if you have matlab on your machine. If matlab is found, it will turn on some unit tests to check some c++ results against matlab ones.

Of course the parts between underscores should be replaced by your own paths. In my case for example I have:
  _path_to_matlab_ = /usr/local/matlab75/
  _path_to_itk_ = /usr/local/mkt-dev/install/itk-3.8.0/RelWithDebInfo/lib/InsightToolkit/

You can also build the project out of the sources. For example:
  cd <symdemons-srcdir>/..
  mkdir symdemons-build
  cd symdemons-build
  cmake -DITK_DIR=/usr/local/mkt-dev/build/itk-cvswrite/RelWithDebInfo/ -DMATLAB_ROOT=/usr/local/matlab75/ -DCMAKE_BUILD_TYPE=RelWithDebInfo ../symdemons
  make -j2


The project has been tested on a linux 32 bits machine, a linux 64 bits machine and an intel mac machine. Compilation was tested on windows XP with visual studio express.
