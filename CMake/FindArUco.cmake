include_directories()
set(aruco_INCLUDE_DIRS )

link_directories("@ArUco_DIR@/lib")
set(aruco_LIB_DIR "")

set(ArUco_LIBS @ArUco_LIBS@)

set(ArUco_FOUND YES)
set(ArUco_VERSION @ArUco_VERSION@)
