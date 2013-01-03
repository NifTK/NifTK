SET(SRC_CPP_FILES
  QmitkBaseView.cpp
)

SET(INTERNAL_CPP_FILES
  CommonActivator.cpp
  VisibilityChangedCommand.cpp
  VisibilityChangeObserver.cpp
)

SET(UI_FILES
)

SET(MOC_H_FILES
  src/internal/CommonActivator.h
  src/QmitkBaseView.h
)

SET(CACHED_RESOURCE_FILES
  plugin.xml
)

# todo: add some qt style sheet resources
SET(QRC_FILES
  resources/CommonResources.qrc
)

SET(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  SET(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
