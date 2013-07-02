/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "BreastSegmentationView.h"

// Qt
#include <QMessageBox>

#include <mitkDataStorageUtils.h>
#include <mitkNodePredicateDataType.h>

#include <_seg_EM.h>


const std::string BreastSegmentationView::VIEW_ID = "uk.ac.ucl.cmic.views.breastsegmentation";

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

BreastSegmentationView::BreastSegmentationView()
{
  m_Modified = false;

  m_NumberOfGaussianComponents = 3;
  m_NumberOfMultiSpectralComponents = 1;
  m_NumberOfTimePoints = 1;
  m_MaximumNumberOfIterations = 30;
  m_BiasFieldCorrectionOrder = 4;
  m_BiasFieldRatioThreshold = 0;
  m_AdiposeGlandularAdjacencyCost = 0.15;
  m_BackgroundGlandularAdjacencyCost = 6.;
  
}


// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

BreastSegmentationView::~BreastSegmentationView()
{

}


// ---------------------------------------------------------------------------
// SetFocus()
// ---------------------------------------------------------------------------

void BreastSegmentationView::SetFocus()
{
  m_Controls.m_InputImageComboBox->setFocus();
}


// ---------------------------------------------------------------------------
// GetNodes()
// ---------------------------------------------------------------------------

mitk::DataStorage::SetOfObjects::ConstPointer BreastSegmentationView::GetNodes()
{
  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage;
  isImage = mitk::TNodePredicateDataType<mitk::Image>::New();

  return dataStorage->GetSubset(isImage);
}


// ---------------------------------------------------------------------------
// CreateQtPartControl()
// ---------------------------------------------------------------------------

void BreastSegmentationView::CreateQtPartControl( QWidget *parent )
{
  std::string name;

  // create GUI widgets from the Qt Designer's .ui file
  m_Controls.setupUi( parent );


  // Initialise the input image combo box

  mitk::DataStorage::SetOfObjects::ConstPointer nodes = GetNodes();

  if ( nodes ) 
  {
    if (nodes->size() > 0) 
    {
      for (unsigned int i=0; i<nodes->size(); i++) 
      {
	(*nodes)[i]->GetStringProperty("name", name);
	m_Controls.m_InputImageComboBox->insertItem(i, QString(name.c_str()));
      }
    }
  }


  this->CreateConnections();
}


// ---------------------------------------------------------------------------
// CreateConnections()
// ---------------------------------------------------------------------------

void BreastSegmentationView::CreateConnections()
{

  connect(m_Controls.m_CancelButton, SIGNAL(pressed()), this,
	  SLOT(OnCancelButtonPressed()) );

  connect(m_Controls.m_ExecuteButton, SIGNAL(pressed()), this,
	  SLOT(OnExecuteButtonPressed()) );

  // Register data storage listeners

  this->GetDataStorage()->AddNodeEvent
    .AddListener( mitk::MessageDelegate1<BreastSegmentationView, const mitk::DataNode*>
		  ( this, &BreastSegmentationView::OnNodeAdded ) );

  this->GetDataStorage()->ChangedNodeEvent
    .AddListener( mitk::MessageDelegate1<BreastSegmentationView, const mitk::DataNode*>
		  ( this, &BreastSegmentationView::OnNodeChanged ) );

  this->GetDataStorage()->RemoveNodeEvent
    .AddListener( mitk::MessageDelegate1<BreastSegmentationView, const mitk::DataNode*>
		  ( this, &BreastSegmentationView::OnNodeRemoved ) );

}


// ---------------------------------------------------------------------------
// OnCancelButtonPressed()
// ---------------------------------------------------------------------------

void BreastSegmentationView::OnCancelButtonPressed()
{
  cout << "CancelButtonPressed" << endl;
}


// ---------------------------------------------------------------------------
// OnExecuteButtonPressed()
// ---------------------------------------------------------------------------

void BreastSegmentationView::OnExecuteButtonPressed()
{
  cout << "ExecuteButtonPressed" << endl;

  if ( ! m_Modified )
  {
    return;
  }

  seg_EM segmentation( m_NumberOfGaussianComponents,
		       m_NumberOfMultiSpectralComponents,
		       m_NumberOfTimePoints );

  segmentation.SetMaximalIterationNumber( m_MaximumNumberOfIterations );
 
  segmentation.Turn_BiasField_ON( m_BiasFieldCorrectionOrder, m_BiasFieldRatioThreshold);

}


// ---------------------------------------------------------------------------
// OnNodeAdded()
// ---------------------------------------------------------------------------

void BreastSegmentationView::OnNodeAdded(const mitk::DataNode* node)
{
  int index;
  std::string name;

  node->GetStringProperty("name", name);

  index = m_Controls.m_InputImageComboBox->findText( QString(name.c_str()) );
  m_Controls.m_InputImageComboBox->addItem( QString(name.c_str()) );
}


// ---------------------------------------------------------------------------
// OnNodeRemoved()
// ---------------------------------------------------------------------------

void BreastSegmentationView::OnNodeRemoved(const mitk::DataNode* node)
{
  int index;
  std::string name;

  node->GetStringProperty("name", name);

  index = m_Controls.m_InputImageComboBox->findText( QString(name.c_str()) );
  
  // If found, remove item
  if ( index >= 0 )
  {
    m_Controls.m_InputImageComboBox->removeItem( index );
  } 
}


// ---------------------------------------------------------------------------
// OnNodeChanged()
// ---------------------------------------------------------------------------

void BreastSegmentationView::OnNodeChanged(const mitk::DataNode* node)
{
  int index;
  std::string name;

  node->GetStringProperty("name", name);

  index = m_Controls.m_InputImageComboBox->findText( QString(name.c_str()) );
  
  // If this is the current item then the segmentation is out of date
  if ( (index != -1) && (index == m_Controls.m_InputImageComboBox->currentIndex()) )
  {
    m_Modified = true;
  }
}
