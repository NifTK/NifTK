
MITK_CREATE_MODULE(XnatRest
  EXPORT_DEFINE XnatRest_EXPORT
  PACKAGE_DEPENDS curl zlib
)

if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(XnatRest PROPERTIES COMPILE_FLAGS "-fvisibility=default")
endif()

# add testing dir
add_subdirectory(Testing)
