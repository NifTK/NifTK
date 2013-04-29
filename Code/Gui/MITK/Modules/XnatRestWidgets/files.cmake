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
  XnatCategoryNode.cxx
  XnatConnection.cxx
  XnatConnectionFactory.cxx
  XnatDownloadDialog.cxx
  XnatDownloadManager.cxx
  XnatEmptyNode.cxx
  XnatException.cxx
  XnatExperimentNode.cxx
  XnatLoginDialog.cxx
  XnatLoginProfile.cxx
  XnatModel.cxx
  XnatNameDialog.cxx
  XnatNode.cxx
  XnatProjectNode.cxx
  XnatReconstructionNode.cxx
  XnatReconstructionResourceFileNode.cxx
  XnatReconstructionResourceNode.cxx
  XnatRootNode.cxx
  XnatScanNode.cxx
  XnatScanResourceFileNode.cxx
  XnatScanResourceNode.cxx
  XnatSettings.cxx
  XnatSubjectNode.cxx
  XnatTreeView.cxx
  XnatUploadDialog.cxx
  XnatUploadManager.cxx
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
