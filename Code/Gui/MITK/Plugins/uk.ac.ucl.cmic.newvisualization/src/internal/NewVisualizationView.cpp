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
#include "NewVisualizationView.h"
#include "NewVisualizationPluginActivator.h"

#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkImageReadAccessor.h>
#include <mitkDataStorageUtils.h>
#include <mitkNodePredicateDataType.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkDataNodePropertyListener.h>
#include <mitkMessage.h>

// THIS IS VERY IMPORTANT
// If nothing is included from the mitk::OpenCL module the resource service will not get registered
#include <mitkOpenCLActivator.h>

// Qt
#include <QMessageBox>

#include <usModule.h>
#include <usModuleResource.h>
#include <usModuleResourceStream.h>
#include <usModuleRegistry.h>

const std::string NewVisualizationView::VIEW_ID = "uk.ac.ucl.cmic.newvisualization";

NewVisualizationView::NewVisualizationView()
: m_Controls(0)
, m_Parent(0)
, m_RenderApplet(0)
, m_VLQtRenderWindow(0)
{
}

NewVisualizationView::~NewVisualizationView()
{

  if (m_SelectionListener)
  {
    m_SelectionListener->NodeAdded   -=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeAdded);
    m_SelectionListener->NodeRemoved -=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeRemoved);
    m_SelectionListener->NodeDeleted -=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeDeleted);
  }

  if (m_VisibilityListener)
    m_VisibilityListener->NodePropertyChanged -=  mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnVisibilityPropertyChanged);

  if (m_NamePropertyListener)
    m_NamePropertyListener->NodePropertyChanged -=  mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnNamePropertyChanged);

  if (m_ColorPropertyListener)
    m_ColorPropertyListener->NodePropertyChanged -=  mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnColorPropertyChanged);
  
  if (m_OpacityPropertyListener)
    m_OpacityPropertyListener->NodePropertyChanged -= mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>( this, &NewVisualizationView::OnOpacityPropertyChanged);

  MITK_INFO <<"Destructing NewViz plugin";

  m_RenderApplet = 0;
}

void NewVisualizationView::SetFocus()
{
}

void NewVisualizationView::CreateQtPartControl( QWidget *parent )
{
  // setup the basic GUI of this view
  m_Parent = parent;

  if (!m_Controls)
  {
    // Create UI.
    m_Controls = new Ui::NewVisualizationViewControls();
    m_Controls->setupUi(parent);

    connect(m_Controls->hSlider_navigate, SIGNAL(valueChanged(int )), this, SLOT(On_SliderMoved(int )));

    // Init listener
    m_SelectionListener = mitk::DataNodePropertyListener::New(GetDataStorage(), "selected", false);
   // m_SelectionListener->NodePropertyChanged +=  mitk::MessageDelegate2<NewVisualizationView, const mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnSelectionChanged);

    m_SelectionListener-> NodeAdded  +=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeAdded);
    m_SelectionListener->NodeRemoved +=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeRemoved);
    m_SelectionListener->NodeDeleted +=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeDeleted);

    m_VisibilityListener = mitk::DataNodePropertyListener::New(GetDataStorage(), "visible");
    m_VisibilityListener->NodePropertyChanged +=  mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnVisibilityPropertyChanged);

    m_NamePropertyListener = mitk::DataNodePropertyListener::New(GetDataStorage(), "name");
    m_NamePropertyListener->NodePropertyChanged +=  mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnNamePropertyChanged);

    m_ColorPropertyListener = mitk::DataNodePropertyListener::New(GetDataStorage(), "color");
    m_ColorPropertyListener->NodePropertyChanged +=  mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnColorPropertyChanged);

    m_OpacityPropertyListener = mitk::DataNodePropertyListener::New(GetDataStorage(), "opacity");
    m_OpacityPropertyListener->NodePropertyChanged += mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>( this, &NewVisualizationView::OnOpacityPropertyChanged);

    // Init the VL visualization part
    InitVLRendering();
  }
}

void  NewVisualizationView::InitVLRendering()
{
  /* init Visualization Library */
  vl::VisualizationLibrary::init();

  /* setup the OpenGL context format */
  vl::OpenGLContextFormat format;
  format.setDoubleBuffer(true);
  format.setRGBABits( 8,8,8,8 );
  format.setDepthBufferBits(24);
  format.setStencilBufferBits(8);
  format.setFullscreen(false);

  if (m_VLQtRenderWindow == 0)
    m_VLQtRenderWindow = new vlQt4::Qt4Widget;

  /* Initialize the OpenGL context and window properties */
  int x = 10;
  int y = 10;
  int width = 512;
  int height= 512;
  m_VLQtRenderWindow->initQt4Widget( "Visualization Library on Qt4", format, NULL, x, y, width, height );

  if (m_RenderApplet == 0)
    m_RenderApplet = new VLRenderingApplet();
  
  m_RenderApplet->initialize();
  //m_VLQtRenderWindow->initializeGL();

  m_VLQtRenderWindow->addEventListener(m_RenderApplet.get());
  m_RenderApplet->rendering()->as<Rendering>()->renderer()->setFramebuffer( m_VLQtRenderWindow->framebuffer() );
  m_RenderApplet->rendering()->as<Rendering>()->camera()->viewport()->setClearColor( black );

  /* define the camera position and orientation */
  vl::vec3 eye    = vl::vec3(0,10,35); // camera position
  vl::vec3 center = vl::vec3(0,0,0);   // point the camera is looking at
  vl::vec3 up     = vl::vec3(0,1,0);   // up direction
  vl::mat4 view_mat = vl::mat4::getLookAt(eye, center, up);
  m_RenderApplet->rendering()->as<Rendering>()->camera()->setViewMatrix( view_mat );
  m_Controls->viewLayout->addWidget(m_VLQtRenderWindow.get());
  
  /* show the window */
  m_VLQtRenderWindow->show();
}

void NewVisualizationView::On_SliderMoved(int val)
{
  m_RenderApplet->UpdateThresholdVal(val);
}

void NewVisualizationView::OnNodeAdded(mitk::DataNode* node)
{
  if (node == 0 || node->GetData()== 0)
    return;

  bool isHelper = false;
  node->GetPropertyList()->GetBoolProperty("helper object", isHelper);

  if (isHelper)
    return;

  bool isVisible = false;
  node->GetVisibility(isVisible, 0);

  if (!isVisible)
    return;

  m_RenderApplet->AddDataNode(node);
  m_RenderApplet->rendering()->render();

  MITK_INFO <<"Node added";
}

void NewVisualizationView::OnNodeRemoved(mitk::DataNode* node)
{
  if (node == 0 || node->GetData()== 0)
    return;

  bool isHelper = false;
  node->GetPropertyList()->GetBoolProperty("helper object", isHelper);

  if (isHelper)
    return;

  m_RenderApplet->RemoveDataNode(node);
  m_RenderApplet->rendering()->render();

  MITK_INFO <<"Node removed";
}

void NewVisualizationView::OnNodeDeleted(mitk::DataNode* node)
{
  if (node == 0 || node->GetData()== 0)
    return;

  m_RenderApplet->RemoveDataNode(node);
  m_RenderApplet->rendering()->render();

  MITK_INFO <<"Node deleted";
}

void NewVisualizationView::OnNamePropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
}

void NewVisualizationView::OnVisibilityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  if (node == 0 || node->GetData()== 0)
    return;

  m_RenderApplet->UpdateDataNode(node);
  m_RenderApplet->rendering()->render();
  MITK_INFO <<"Visibility Change";
}

void NewVisualizationView::OnColorPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  if (node == 0 || node->GetData()== 0)
    return;

  m_RenderApplet->UpdateDataNode(node);
  m_RenderApplet->rendering()->render();
  MITK_INFO <<"Color Change";
}

void NewVisualizationView::OnOpacityPropertyChanged(mitk::DataNode* node, const mitk::BaseRenderer* renderer)
{
  if (node == 0 || node->GetData()== 0)
    return;

  m_RenderApplet->UpdateDataNode(node);
  m_RenderApplet->rendering()->render();
  MITK_INFO <<"Opacity Change";
}

void NewVisualizationView::Visible()
{
  QmitkBaseView::Visible();

  // Make sure that we show all the nodes that are already present in DataStorage
  UpdateDisplay();
}

void NewVisualizationView::UpdateDisplay(bool viewEnabled)
{
  m_RenderApplet->sceneManager()->tree()->actors()->clear();

    // Set DataNode property accordingly
  typedef mitk::DataNode::Pointer dataNodePointer;
  typedef itk::VectorContainer<unsigned int, dataNodePointer> nodesContainerType;
  nodesContainerType::ConstPointer vc = this->GetDataStorage()->GetAll();

  // Iterate through the DataNodes
  for (unsigned int i = 0; i < vc->Size(); i++)
  {
    dataNodePointer currentDataNode = vc->ElementAt(i);
    if (currentDataNode.IsNull() || currentDataNode->GetData()== 0)
      continue;

    bool isHelper = false;
    currentDataNode->GetPropertyList()->GetBoolProperty("helper object", isHelper);

    if (isHelper)
      continue;

    bool isVisible = false;
    currentDataNode->GetVisibility(isVisible, 0);

    if (!isVisible)
      continue;
    
    m_RenderApplet->AddDataNode(currentDataNode);
    //m_RenderApplet->rendering()->render();
    MITK_INFO <<"Node added";
  }

  m_RenderApplet->rendering()->render();
}