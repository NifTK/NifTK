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
/*
  if (m_SelectionListener)
  {
    m_SelectionListener->NodePropertyChanged -= mitk::MessageDelegate2<NewVisualizationView, mitk::DataNode*, const mitk::BaseRenderer*>(this, &NewVisualizationView::OnSelectionChanged);
    m_SelectionListener->NodeAdded   -=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeAdded);
    m_SelectionListener->NodeRemoved -=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeRemoved);
    m_SelectionListener->NodeDeleted -=  mitk::MessageDelegate1<NewVisualizationView, mitk::DataNode*>(this, &NewVisualizationView::OnNodeDeleted);
  }

  if (m_VisibilityListener)
    m_VisibilityListener->NodePropertyChanged -=  mitk::MessageDelegate2<PlanningManager, mitk::DataNode*, const mitk::BaseRenderer*>(this, &PlanningManager::OnVisibilityChanged);
*/
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

/*
    // Init listener
    m_SelectionListener = mitk::DataNodePropertyListener::New(dataStorage, "selected", false);
    
    m_SelectionListener->NodePropertyChanged +=  mitk::MessageDelegate2<PlanningManager, mitk::DataNode*, const mitk::BaseRenderer*>(this, &PlanningManager::OnSelectionChanged);
    m_SelectionListener->NodeAdded   +=  mitk::MessageDelegate1<PlanningManager, mitk::DataNode*>(this, &PlanningManager::OnNodeAdded);
    m_SelectionListener->NodeRemoved +=  mitk::MessageDelegate1<PlanningManager, mitk::DataNode*>(this, &PlanningManager::OnNodeRemoved);
    m_SelectionListener->NodeDeleted +=  mitk::MessageDelegate1<PlanningManager, mitk::DataNode*>(this, &PlanningManager::OnNodeDeleted);

    m_VisibilityListener = mitk::DataNodePropertyListener::New(dataStorage, "visible");
    m_VisibilityListener->NodePropertyChanged +=  mitk::MessageDelegate2<PlanningManager, mitk::DataNode*, const mitk::BaseRenderer*>(this, &PlanningManager::OnVisibilityChanged);

    m_PropertyListener = mitk::DataNodePropertyListener::New(dataStorage, "name");
    m_PropertyListener->NodePropertyChanged +=  mitk::MessageDelegate2<PlanningManager, mitk::DataNode*, const mitk::BaseRenderer*>(this, &PlanningManager::OnPropertyChanged);
*/
    InitVLRendering();

/*
    OclResourceService* oclService =  mitk::NewVisualizationPluginActivator::GetOpenCLService();

    if (oclService == NULL)
    {
      mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
    }

    vl::OpenGLContext * glContext = m_RenderApplet->openglContext();
    glContext->makeCurrent();

    // Force tests to run on the ATI GPU
    oclService->SpecifyPlatformAndDevice(0, 0, true);


    // Get context 
    cl_context gpuContext = oclService->GetContext();
*/
  }

  // Redraw screen
  //UpdateDisplay();
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

  m_VLQtRenderWindow = new vlQt4::Qt4Widget;


  m_RenderApplet = new VLRenderingApplet();

  m_RenderApplet->initialize();
  m_VLQtRenderWindow->addEventListener(m_RenderApplet.get());
  m_RenderApplet->rendering()->as<Rendering>()->renderer()->setFramebuffer( m_VLQtRenderWindow->framebuffer() );
  m_RenderApplet->rendering()->as<Rendering>()->camera()->viewport()->setClearColor( black );

  /* define the camera position and orientation */
  vl::vec3 eye    = vl::vec3(0,10,35); // camera position
  vl::vec3 center = vl::vec3(0,0,0);   // point the camera is looking at
  vl::vec3 up     = vl::vec3(0,1,0);   // up direction
  vl::mat4 view_mat = vl::mat4::getLookAt(eye, center, up);
  m_RenderApplet->rendering()->as<Rendering>()->camera()->setViewMatrix( view_mat );

  /* Initialize the OpenGL context and window properties */
  int x = 10;
  int y = 10;
  int width = 512;
  int height= 512;
  m_VLQtRenderWindow->initQt4Widget( "Visualization Library on Qt4 - Rotating Cube", format, NULL, x, y, width, height );

  m_Controls->viewLayout->addWidget(m_VLQtRenderWindow.get());

  /* show the window */
  m_VLQtRenderWindow->show();
}

void NewVisualizationView::On_SliderMoved(int val)
{
  m_RenderApplet->UpdateThresholdVal(val);
}



void NewVisualizationView::OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                             const QList<mitk::DataNode::Pointer>& nodes )
{

 
  // Update visibility settings
 // UpdateDisplay();
}
void NewVisualizationView::NodeAdded(const mitk::DataNode* node)
{
  if (node == 0)
    return;
 
  UpdateDisplay();
}

void NewVisualizationView::NodeChanged(const mitk::DataNode* node)
{
  if (node == 0 || m_Controls == 0)
    return;

  //UpdateDisplay();

}

void NewVisualizationView::NodeRemoved(const mitk::DataNode* node)
{
  if (node == 0 || m_Controls == 0)
    return;
  
  m_RenderApplet->sceneManager()->tree()->actors()->clear();
  UpdateDisplay();
}

void NewVisualizationView::OnNodeDeleted(const mitk::DataNode* node)
{
  if (node == 0 || m_Controls == 0)
    return;

  UpdateDisplay();
}



void NewVisualizationView::UpdateDisplay(bool viewEnabled)
{
  //m_RenderApplet->sceneManager()->tree()->actors()->clear();

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

    // Get name of the current node
    QString currNodeName(currentDataNode->GetPropertyList()->GetProperty("name")->GetValueAsString().c_str() );
    
    m_RenderApplet->AddDataNode(currentDataNode);
  }

  m_RenderApplet->rendering()->render();
}