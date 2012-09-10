## MASSIVE HACK: Adding a patch to libcurl as a workaround for cmake bug 0011240
## http://public.kitware.com/Bug/view.php?id=11240  
  
  #set( CMAKE_SIZEOF_VOID_P @CMAKE_SIZEOF_VOID_P@ )
  #set( MSVC @MSVC@ )
  set( _SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  
  set( _TARGET "${_SOURCE_DIR}/CMakeLists.txt" )
    
  if( NOT (EXISTS "${_TARGET}") )
	message( FATAL_ERROR "Cannot find ${_TARGET}" )
  endif()
  
  file( READ "${_TARGET}" _DATA )
  list( APPEND _DATA "#=== Patch - append BEGIN ===\n" )
  
  message("\n ********* CMAKE_SIZEOF_VOID_P: ${CMAKE_SIZEOF_VOID_P} ********* \n" )
  message("\n ********* MSVC: ${MSVC} ********* \n" )
  
  #if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND MSVC )
	# Error detected with generator Visual Studio 10 Win64
	# > LINK : warning LNK4068: /MACHINE not specified; defaulting to X86
	# > fatal error LNK1112: module machine type 'x64' conflicts with target machine type 'X86'
	#
	# There is no default value for static libraries and cmake isn't setting it either.
	# We fix this by adding the flag manually.
	
	message("\n ********* Specifying machine type: /machine:x64 ********* \n" )
	list( APPEND _DATA "set_target_properties(libcurl PROPERTIES STATIC_LIBRARY_FLAGS \"/MACHINE:x64\")\n" )
	#list( APPEND _DATA "set(CURL_SIZEOF_CURL_SOCKLEN_T 4)\n" )
	#list( APPEND _DATA "set(HAVE_CURL_SIZEOF_CURL_SOCKLEN_T TRUE)\n" )
	
	
   #endif()
   list( APPEND _DATA "#=== Patch - append END ===\n" )
   file( WRITE "${_TARGET}" ${_DATA} )
   
## END OF MASSIVE HACK: Adding a patch to libcurl as a workaround for cmake bug 0011240 