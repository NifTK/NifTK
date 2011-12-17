
set(module_dirs
  niftkMitkExt
  niftkQmitkExt
)

foreach(module_dir ${module_dirs})
  add_subdirectory(${module_dir})
endforeach()
