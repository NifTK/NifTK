/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMorphologicalSegmentorView.h"

#include <QMessageBox>

#include <berryIWorkbenchPage.h>

#include <mitkColorProperty.h>
#include <mitkDataStorageUtils.h>
#include <mitkImage.h>
#include <mitkImageAccessByItk.h>
#include <mitkImageCast.h>
#include <mitkITKImageImport.h>

#include <niftkMIDASImageUtils.h>
#include <niftkMIDASOrientationUtils.h>

#include <itkConversionUtils.h>
#include <mitkITKRegionParametersDataNodeProperty.h>
#include <niftkMIDASTool.h>
#include <niftkMIDASPaintbrushTool.h>

#include <niftkMIDASOrientationUtils.h>

#include "niftkMorphologicalSegmentorController.h"
#include <niftkMorphologicalSegmentorGUI.h>


const std::string niftkMorphologicalSegmentorView::VIEW_ID = "uk.ac.ucl.cmic.midasmorphologicalsegmentor";

//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::niftkMorphologicalSegmentorView()
: niftkBaseSegmentorView()
{
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::niftkMorphologicalSegmentorView(
    const niftkMorphologicalSegmentorView& other)
{
  Q_UNUSED(other)
  throw std::runtime_error("Copy constructor not implemented");
}


//-----------------------------------------------------------------------------
niftkMorphologicalSegmentorView::~niftkMorphologicalSegmentorView()
{
}


//-----------------------------------------------------------------------------
std::string niftkMorphologicalSegmentorView::GetViewID() const
{
  return VIEW_ID;
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorController* niftkMorphologicalSegmentorView::CreateSegmentorController()
{
  m_MorphologicalSegmentorController = new niftkMorphologicalSegmentorController(this);
  return m_MorphologicalSegmentorController;
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::SetFocus()
{
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::NodeRemoved(const mitk::DataNode* removedNode)
{
  assert(m_MorphologicalSegmentorController);
  m_MorphologicalSegmentorController->OnNodeRemoved(removedNode);
}


//-----------------------------------------------------------------------------
QString niftkMorphologicalSegmentorView::GetPreferencesNodeName()
{
  return niftkMorphologicalSegmentorPreferencePage::PREFERENCES_NODE_NAME;
}


//-----------------------------------------------------------------------------
void niftkMorphologicalSegmentorView::onVisibilityChanged(const mitk::DataNode* node)
{
  assert(m_MorphologicalSegmentorController);
  m_MorphologicalSegmentorController->OnNodeVisibilityChanged(node);
}
