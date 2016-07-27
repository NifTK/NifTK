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
#include "PointSetConverterView.h"

// Qt
#include <QMessageBox>
#include <qinputdialog.h>

//mitk image
#include <mitkImage.h>

const std::string PointSetConverterView::VIEW_ID = "org.mitk.views.pointsetconverter";


PointSetConverterView::PointSetConverterView()
: m_Controls(NULL)
, m_Parent(NULL)
, m_ReferenceImage(NULL)
{
}

PointSetConverterView::~PointSetConverterView()
{
}

void PointSetConverterView::SetFocus()
{
}

void PointSetConverterView::CreateQtPartControl( QWidget *parent )
{
  // create GUI widgets from the Qt Designer's .ui file
  m_Parent = parent;

  if (!m_Controls)
  {
    // create GUI widgets from the Qt Designer's .ui file
    m_Controls = new Ui::PointSetConverterViewControls();
    m_Controls->setupUi( parent );

    // Connect Qt signals and slots programmatically.
    connect( m_Controls->m_PolygonsToPointSetButton, SIGNAL(clicked()), this, SLOT(OnConvertPolygonsToPointSetButtonClicked()) );
    connect( m_Controls->m_AddPointSetButton, SIGNAL(clicked()), this, SLOT(OnCreateNewPointSetButtonClicked()) );

    m_Controls->m_ReferenceImageNameLabel->setText("Not selected");
    m_Controls->m_ReferenceImageNameLabel->setStyleSheet("QLabel { color: red } ");
    
    m_Controls->m_ActivePointSetNameLabel->setText("Not selected");
    m_Controls->m_ActivePointSetNameLabel->setStyleSheet("QLabel { color: red } ");
  }   
}

void PointSetConverterView::OnSelectionChanged( berry::IWorkbenchPart::Pointer /*source*/,
                                             const QList<mitk::DataNode::Pointer>& nodes )
{
  if (nodes.isEmpty())
    return;  

  // Get the first selected
  mitk::DataNode::Pointer currentDataNode;
  currentDataNode = nodes[0];

  mitk::PointSet::Pointer pointSet = dynamic_cast<mitk::PointSet *> ( currentDataNode->GetData() );
  
  if( pointSet.IsNotNull() )
  {
    m_Controls->m_PointSetWidget->SetPointSetNode(currentDataNode);
    
    m_Controls->m_ActivePointSetNameLabel->setText( currentDataNode->GetName().c_str() );
    m_Controls->m_ActivePointSetNameLabel->setStyleSheet("QLabel { color: black } ");
  }

  mitk::Image::Pointer reference = dynamic_cast< mitk::Image * > ( currentDataNode->GetData() );
  if( reference.IsNotNull() )
  {
    m_ReferenceImage = currentDataNode;
    m_Controls->m_ReferenceImageNameLabel->setText( currentDataNode->GetName().c_str() );
    m_Controls->m_ReferenceImageNameLabel->setStyleSheet("QLabel { color: black } ");
  }
}

void PointSetConverterView::OnCreateNewPointSetButtonClicked()
{
  //Ask for the name of the point set
  bool ok = false;
  QString name = QInputDialog::getText( QApplication::activeWindow()
    , "Add point set...", "Enter name for the new target points", QLineEdit::Normal, "Point Set", &ok );

  if ( ! ok || name.isEmpty() )
    return;

  //Create a new empty pointset
  mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
  
  // Create a new data tree node
  mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
  pointSetNode->SetData( pointSet );
  pointSetNode->SetProperty( "name", mitk::StringProperty::New( name.toStdString() ) );
  pointSetNode->SetProperty( "opacity", mitk::FloatProperty::New( 1 ) );

  // add to datastorage
  this->GetDataStorage()->Add(pointSetNode, m_ReferenceImage);
    

  m_Controls->m_PointSetWidget->SetPointSetNode(pointSetNode);
  m_Controls->m_ActivePointSetNameLabel->setText( pointSetNode->GetName().c_str() );
  m_Controls->m_ActivePointSetNameLabel->setStyleSheet("QLabel { color: black } ");

}

void PointSetConverterView::OnConvertPolygonsToPointSetButtonClicked()
{
 
  if( m_ReferenceImage.IsNull() )
    return;

  //get all the children nodes of the reference image
  typedef itk::VectorContainer<unsigned int, mitk::DataNode::Pointer > DataNodeContainerType;
  DataNodeContainerType::ConstPointer nodes = this->GetDataStorage()->GetDerivations( m_ReferenceImage );

  for( unsigned int i=0;i<nodes->Size();i++ )
  {
    mitk::DataNode::Pointer currentDataNode = nodes->at(i);
    mitk::PlanarCircle::Pointer object = dynamic_cast< mitk::PlanarCircle * > ( currentDataNode->GetData() );

    if( object.IsNotNull() )
    {
      mitk::Point3D point = PlanarCircleToPoint(object);
      mitk::PointSet::Pointer pointSet = this->FindPointSetNode( currentDataNode->GetName() );

      if( pointSet.IsNull() )
      {
        pointSet = mitk::PointSet::New();

        mitk::DataNode::Pointer pointSetNode = mitk::DataNode::New();
        pointSetNode->SetName( currentDataNode->GetName() );
        pointSetNode->SetData( pointSet );

        this->GetDataStorage()->Add(pointSetNode, m_ReferenceImage);

      }

      pointSet->SetPoint( pointSet->GetSize(), point );

    }

  }

}

mitk::Point3D PointSetConverterView::PlanarCircleToPoint( const mitk::PlanarCircle* circle)
{
  mitk::Point3D point;
  point.Fill(0.0);

  int numControlPoints = circle->GetNumberOfControlPoints();
  for( unsigned int i=0;i<numControlPoints;i++ )
  {
    for(unsigned int j=0;j< mitk::Point3D::Dimension;j++ )
      point[j] += circle->GetWorldControlPoint(i)[j]/numControlPoints;
  }

  return point;
}


mitk::PointSet::Pointer PointSetConverterView::FindPointSetNode( const std::string& name )
{
    //get all the children nodes of the reference image
  typedef itk::VectorContainer<unsigned int, mitk::DataNode::Pointer > DataNodeContainerType;
  DataNodeContainerType::ConstPointer nodes = this->GetDataStorage()->GetDerivations( m_ReferenceImage );

  for( unsigned int i=0;i<nodes->Size();i++ )
  {
    mitk::DataNode::Pointer currentDataNode = nodes->at(i);
    if( currentDataNode->GetName().compare(name) == 0 )
    {
      mitk::PointSet::Pointer pointSet = dynamic_cast<mitk::PointSet *> ( currentDataNode->GetData() );

      if( pointSet.IsNotNull() )
        return pointSet;
    }

  }

  return NULL;
}
