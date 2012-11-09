SET(SRC_CPP_FILES
  QmitkBaseAppWorkbenchAdvisor.cpp
  QmitkBaseWorkbenchWindowAdvisor.cpp
)

SET(INTERNAL_CPP_FILES
  QmitkCommonAppsApplicationPlugin.cpp
  QmitkCommonAppsApplicationPreferencePage.cpp  
)

SET(MOC_H_FILES
  src/QmitkBaseWorkbenchWindowAdvisor.h
  src/internal/QmitkCommonAppsApplicationPlugin.h
  src/internal/QmitkCommonAppsApplicationPreferencePage.h
)

SET(CACHED_RESOURCE_FILES
# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
  plugin.xml
  resources/icon_cmic.xpm
  resources/icon_ion.xpm
  resources/icon_ucl.xpm
)

SET(QRC_FILES
# uncomment the following line if you want to use Qt resources
  resources/CommonAppsResources.qrc
)

SET(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
