#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(CPP_FILES
  XnatCategoryNode.cpp
  XnatConnection.cpp
  XnatConnectionFactory.cpp
  XnatDownloadDialog.cpp
  XnatDownloadManager.cpp
  XnatEmptyNode.cpp
  XnatException.cpp
  XnatExperimentNode.cpp
  XnatLoginDialog.cpp
  XnatLoginProfile.cpp
  XnatModel.cpp
  XnatNameDialog.cpp
  XnatNode.cpp
  XnatProjectNode.cpp
  XnatReconstructionNode.cpp
  XnatReconstructionResourceFileNode.cpp
  XnatReconstructionResourceNode.cpp
  XnatRootNode.cpp
  XnatScanNode.cpp
  XnatScanResourceFileNode.cpp
  XnatScanResourceNode.cpp
  XnatSettings.cpp
  XnatSubjectNode.cpp
  XnatTreeView.cpp
  XnatUploadDialog.cpp
  XnatUploadManager.cpp
)

set(UI_FILES
  XnatLoginDialog.ui
)

set(MOC_H_FILES
  XnatDownloadDialog.h
  XnatDownloadManager.h
  XnatLoginDialog.h
  XnatModel.h
  XnatNameDialog.h
  XnatTreeView.h
  XnatUploadDialog.h
  XnatUploadManager.h
)

set(QRC_FILES
)
