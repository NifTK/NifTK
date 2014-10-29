/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkPointSetCropper.h"

#include <iostream>

#include <QAction>
#include <QInputDialog>
#include <QMessageBox>
#include <QCheckBox>
#include <QSpinBox>
#include <QSlider>

#include <vtkRenderWindow.h>

#include <mitkRenderingManager.h>
#include <mitkEllipsoid.h>
#include <mitkCylinder.h>
#include <mitkCone.h>
#include <mitkProperties.h>
#include <mitkGlobalInteraction.h>
#include <mitkUndoController.h>
#include <mitkBoundingObjectCutter.h>
#include <mitkIDataStorageService.h>
#include <mitkNodePredicateDataType.h>

#include <itkCommand.h>

// To be moved to mitkInteractionConst.h by StateMachineEditor
const mitk::OperationType QmitkPointSetCropper::OP_EXCHANGE = 718;

//-----------------------------------------------------------------------------
QmitkPointSetCropper::opExchangeNodes::opExchangeNodes(
    mitk::OperationType type,
    mitk::DataNode* node,
    mitk::BaseData* oldData,
    mitk::BaseData* newData )
: mitk::Operation(type)
, m_Node(node)
, m_OldData(oldData)
, m_NewData(newData)
, m_NodeDeletedObserverTag(0)
, m_OldDataDeletedObserverTag(0)
, m_NewDataDeletedObserverTag(0)
{
  itk::MemberCommand<opExchangeNodes>::Pointer nodeDeletedCommand = itk::MemberCommand<opExchangeNodes>::New();
  nodeDeletedCommand->SetCallbackFunction(this, &opExchangeNodes::NodeDeleted);

  m_NodeDeletedObserverTag = m_Node->AddObserver(itk::DeleteEvent(), nodeDeletedCommand);
  m_OldDataDeletedObserverTag = m_OldData->AddObserver(itk::DeleteEvent(), nodeDeletedCommand);
  m_NewDataDeletedObserverTag = m_NewData->AddObserver(itk::DeleteEvent(), nodeDeletedCommand);
}


//-----------------------------------------------------------------------------
QmitkPointSetCropper::opExchangeNodes::~opExchangeNodes()
{
  if (m_Node != NULL)
  {
    m_Node->RemoveObserver(m_NodeDeletedObserverTag);
    m_Node=NULL;
  }

  if (m_OldData.IsNotNull())
  {
    m_OldData->RemoveObserver(m_OldDataDeletedObserverTag);
    m_OldData=NULL;
  }

  if (m_NewData.IsNotNull())
  {
    m_NewData->RemoveObserver(m_NewDataDeletedObserverTag);
    m_NewData=NULL;
  }
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::opExchangeNodes::NodeDeleted(const itk::Object * /*caller*/, const itk::EventObject &/*event*/)
{
  m_Node = NULL;
  m_OldData = NULL;
  m_NewData = NULL;
}


//-----------------------------------------------------------------------------
QmitkPointSetCropper::QmitkPointSetCropper(QObject *parent)
: m_Controls(NULL)
, m_ParentWidget(0)
{
   m_Interface = new mitk::PointSetCropperEventInterface;
   m_Interface->SetPointSetCropper( this );
}


//-----------------------------------------------------------------------------
QmitkPointSetCropper::~QmitkPointSetCropper()
{
  m_CroppingObjectNode = NULL;
  m_CroppingObject = NULL;
  m_Interface->Delete();
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::SetFocus()
{

}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::CreateQtPartControl(QWidget* parent)
{
  if (!m_Controls)
  {
    m_ParentWidget = parent;

    // Build ui elements
    m_Controls = new Ui::QmitkPointSetCropperControls;
    m_Controls->setupUi(parent);

    // Setup ui elements
    m_Controls->groupInfo->hide();
    m_Controls->m_BoxButton->setEnabled(true);

    // create ui element connections
    this->CreateConnections();
  }
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::CreateConnections()
{
  if ( m_Controls )
  {
    connect( m_Controls->m_CropButton, SIGNAL(clicked()), this, SLOT(CropPointSet()));   // click on the crop button
    connect( m_Controls->m_BoxButton, SIGNAL(clicked()), this, SLOT(CreateNewBoundingObject()) );
    connect( m_Controls->chkInformation, SIGNAL(toggled(bool)), this, SLOT(ChkInformationToggled(bool)) );
  }
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::ExecuteOperation (mitk::Operation *operation)
{
  if (!operation) return;

  switch (operation->GetOperationType())
  {
  case OP_EXCHANGE:
    {
      // RemoveBoundingObjectFromNode();
      opExchangeNodes* op = static_cast<opExchangeNodes*>(operation);
      op->GetNode()->SetData(op->GetNewData());
      mitk::RenderingManager::GetInstance()->RequestUpdateAll();
      break;
    }
  default:;
  }

}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::CreateNewBoundingObject()
{
  if (this->IsVisible())
  {
    if (m_PointSetNode.IsNotNull())
    {
      m_PointSetToCrop = dynamic_cast<mitk::PointSet*>(m_PointSetNode->GetData());

      if(m_PointSetToCrop.IsNotNull())
      {
        if (this->GetDataStorage()->GetNamedDerivedNode("CroppingObject", m_PointSetNode))
        {
          // Remove m_Cropping
          this->RemoveBoundingObjectFromNode();
        }

        bool fitCroppingObject = false;
        if(m_CroppingObject.IsNull())
        {
          CreateBoundingObject();
          fitCroppingObject = true;
        }

        if (m_CroppingObject.IsNull())
        {
          return;
        }

        // Add m_Cropping
        AddBoundingObjectToNode( m_PointSetNode, fitCroppingObject );

        m_PointSetNode->SetVisibility(true);
        mitk::RenderingManager::GetInstance()->RequestUpdateAll();
        m_Controls->m_BoxButton->setText("Reset bounding box!");
        m_Controls->m_CropButton->setEnabled(true);
      }
    }
    else
    {
      QMessageBox::information(NULL, "PointSet cropping functionality", "Load a PointSet first!");
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::CropPointSet()
{
  if (m_PointSetToCrop.IsNull())
  {
    QMessageBox::information(NULL, "PointSet cropping plugin", "Select a PointSet first!");
    return;
  }

  if (m_CroppingObjectNode.IsNull())
  {
    QMessageBox::information(NULL, "PointSet cropping plugin", "Generate a new bounding object first!");
    return;
  }

  mitk::PointSet::Pointer resultPointSet = mitk::PointSet::New();

  mitk::PointSet::DataType* uncroppedPointSet = m_PointSetToCrop->GetPointSet(0);
  mitk::PointSet::PointsContainer* uncroppedPoints = uncroppedPointSet->GetPoints();
  mitk::PointSet::PointsIterator uncroppedPointsIt;
  mitk::PointSet::PointIdentifier pointID;
  mitk::PointSet::PointType uncroppedPoint;

  for (uncroppedPointsIt = uncroppedPoints->Begin(); uncroppedPointsIt != uncroppedPoints->End(); ++uncroppedPointsIt)
  {
    pointID = uncroppedPointsIt->Index();
    uncroppedPoint = uncroppedPointsIt->Value();
    if (m_CroppingObject->IsInside(uncroppedPoint))
    {
      resultPointSet->InsertPoint(pointID, uncroppedPoint);
    }
  }

  RemoveBoundingObjectFromNode();

  {
    opExchangeNodes*  doOp   = new opExchangeNodes(OP_EXCHANGE, m_PointSetNode.GetPointer(), m_PointSetToCrop, resultPointSet);
    opExchangeNodes* undoOp  = new opExchangeNodes(OP_EXCHANGE, m_PointSetNode.GetPointer(), resultPointSet, m_PointSetToCrop);

    mitk::OperationEvent* operationEvent = new mitk::OperationEvent( m_Interface, doOp, undoOp, "Crop PointSet");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent ); // tell the undo controller about the action
    ExecuteOperation(doOp); // execute action
  }

  m_Controls->m_BoxButton->setEnabled(true);
  m_Controls->m_CropButton->setEnabled(false);
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::CreateBoundingObject()
{
  QStringList items;
  items << tr("Cuboid") << tr("Ellipsoid") << tr("Cylinder") << tr("Cone");

  bool ok;
  QString item = QInputDialog::getItem(m_ParentWidget, tr("Select Bounding Object"), tr("Type of Bounding Object:"), items, 0, false, &ok);

  if (!ok)
    return;

  if (item == "Ellipsoid")
    m_CroppingObject = mitk::Ellipsoid::New();
  else if(item == "Cylinder")
    m_CroppingObject = mitk::Cylinder::New();
  else if (item == "Cone")
    m_CroppingObject = mitk::Cone::New();
  else if (item == "Cuboid")
    m_CroppingObject = mitk::Cuboid::New();
  else
    return;

  m_CroppingObjectNode = mitk::DataNode::New();
  m_CroppingObjectNode->SetData( m_CroppingObject );
  m_CroppingObjectNode->SetProperty( "name", mitk::StringProperty::New( "CroppingObject" ) );
  m_CroppingObjectNode->SetProperty( "color", mitk::ColorProperty::New(1.0, 1.0, 0.0) );
  m_CroppingObjectNode->SetProperty( "opacity", mitk::FloatProperty::New(0.4) );
  m_CroppingObjectNode->SetProperty( "layer", mitk::IntProperty::New(99) ); // arbitrary, copied from segmentation functionality
  m_CroppingObjectNode->SetProperty( "helper object", mitk::BoolProperty::New(true) );

  m_AffineInteractor = mitk::AffineInteractor::New("AffineInteractions ctrl-drag", m_CroppingObjectNode);
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::OnSelectionChanged(std::vector<mitk::DataNode*> nodes)
{
  this->RemoveBoundingObjectFromNode();

  if (nodes.size() != 1 || dynamic_cast<mitk::PointSet*>(nodes[0]->GetData()) == 0)
  {
    m_ParentWidget->setEnabled(false);
    return;
  }

  m_PointSetNode = nodes[0];
  m_ParentWidget->setEnabled(true);
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::AddBoundingObjectToNode(mitk::DataNode* node, bool fit)
{
  m_PointSetToCrop = dynamic_cast<mitk::PointSet*>(node->GetData());
  if(!this->GetDataStorage()->Exists(m_CroppingObjectNode))
  {
    this->GetDataStorage()->Add(m_CroppingObjectNode, node);
    if (fit)
    {
      m_CroppingObject->FitGeometry(m_PointSetToCrop->GetGeometry());
    }

    mitk::GlobalInteraction::GetInstance()->AddInteractor( m_AffineInteractor );
  }
  m_CroppingObjectNode->SetVisibility(true);
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::RemoveBoundingObjectFromNode()
{
  if (m_CroppingObjectNode.IsNotNull())
  {
    if(this->GetDataStorage()->Exists(m_CroppingObjectNode))
    {
      this->GetDataStorage()->Remove(m_CroppingObjectNode);
      mitk::GlobalInteraction::GetInstance()->RemoveInteractor(m_AffineInteractor);
      m_CroppingObject = NULL;
    }
    m_Controls->m_BoxButton->setText("New bounding box!");
  }
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::ChkInformationToggled( bool on )
{
  if (on)
  {
    m_Controls->groupInfo->show();
  }
  else
  {
    m_Controls->groupInfo->hide();
  }
}


//-----------------------------------------------------------------------------
void QmitkPointSetCropper::NodeRemoved(const mitk::DataNode *node)
{
  std::string name = node->GetName();

  if (strcmp(name.c_str(), "CroppingObject")==0)
  {
    m_Controls->m_CropButton->setEnabled(false);
    m_Controls->m_BoxButton->setEnabled(true);
  }
}
