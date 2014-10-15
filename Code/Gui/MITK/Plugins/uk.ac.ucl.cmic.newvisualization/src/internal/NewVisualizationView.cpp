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

// VL
#include <vlCore/VisualizationLibrary.hpp>
#include <vlQt4/Qt4Widget.hpp>
#include <vlVolume/RaycastVolume.hpp>
//#include <Applets/BaseDemo.hpp>

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
, m_RenderWindow(0)
{
    
}

NewVisualizationView::~NewVisualizationView()
{
 
  if (m_RenderWindow != 0)
    delete m_RenderWindow;
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



    /* init Visualization Library */
    vl::VisualizationLibrary::init();

    /* setup the OpenGL context format */
    vl::OpenGLContextFormat format;
    format.setDoubleBuffer(true);
    format.setRGBABits( 8,8,8,8 );
    format.setDepthBufferBits(24);
    format.setStencilBufferBits(8);
    format.setFullscreen(false);

    m_RenderApplet = new App_VolumeSliced();

    m_RenderApplet->initialize();
    /* create a native Qt4 window */
    vl::ref<vlQt4::Qt4Widget> qt4_window = new vlQt4::Qt4Widget;
    /* bind the applet so it receives all the GUI events related to the OpenGLContext */
    qt4_window->addEventListener(m_RenderApplet.get());
    /* target the window so we can render on it */
    m_RenderApplet->rendering()->as<Rendering>()->renderer()->setFramebuffer( qt4_window->framebuffer() );
    /* black background */
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
    qt4_window->initQt4Widget( "Visualization Library on Qt4 - Rotating Cube", format, NULL, x, y, width, height );

    m_Controls->viewLayout->addWidget(qt4_window.get());

    /* show the window */
    qt4_window->show();

    ctkPluginContext* context = mitk::NewVisualizationPluginActivator::GetDefault()->GetPluginContext();

  ctkServiceReference serviceRef = context->getServiceReference<OclResourceService>();
  OclResourceService* oclService = context->getService<OclResourceService>(serviceRef);
  
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

  /*
    // Set up the probe eye view
    m_RenderWindow = new QmitkRenderWindow(m_Controls->groupBox_ProbeView, QString("Probe Eye View"));
    
    // Get rendering manager
    mitk::RenderingManager::Pointer globalRenderingManager = mitk::RenderingManager::GetInstance();
    globalRenderingManager->AddRenderWindow(m_RenderWindow->GetRenderWindow());
    m_Controls->probeEyeLayout->addWidget(m_RenderWindow);

    // Get data storage pointer
    mitk::DataStorage::Pointer dsp =  this->GetDataStorage();
    
    // Tell the RenderWindow which (part of) the datastorage to render
    m_RenderWindow->GetRenderer()->SetDataStorage(dsp);
    
    ///**********************************************************************************
    ///                                 UGLY HACK 2
    ///**********************************************************************************
    // Need to initialize the render window in case it is closed and re-opened
    {
      // get all nodes that have not set "includeInBoundingBox" to false
      mitk::NodePredicateNot::Pointer pred 
        = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox" , mitk::BoolProperty::New(false)));

      mitk::DataStorage::SetOfObjects::ConstPointer rs = this->GetDataStorage()->GetSubset(pred);
      // calculate bounding geometry of these nodes
      mitk::TimeGeometry::Pointer bounds = this->GetDataStorage()->ComputeBoundingGeometry3D(rs);
      // initialize the view to the bounding geometry
      globalRenderingManager->InitializeView(m_RenderWindow->GetRenderWindow(), bounds);
    }
    ///**********************************************************************************

    m_RenderWindow->GetSliceNavigationController()->SetSliceLocked(true);
    m_RenderWindow->GetSliceNavigationController()->SetSliceRotationLocked(true);

*/
  }

  // Visibility set according to trajectory selected.
  UpdateDisplay();
}

void NewVisualizationView::OnSelectionChanged( berry::IWorkbenchPart::Pointer source,
                                             const QList<mitk::DataNode::Pointer>& nodes )
{

  if (nodes.isEmpty())
    return;

  // Get the first selected
  mitk::DataNode::Pointer currentDataNode;
  currentDataNode = nodes[0];

  if (currentDataNode.IsNull())
    return;

  mitk::Image::Pointer img = dynamic_cast<mitk::Image *>(currentDataNode->GetData());

  if (img.IsNull())
    return;

  try
  {
    mitk::ImageReadAccessor readAccess(img, img->GetVolumeData(0));
    const void* cPointer = readAccess.GetData();
    unsigned int * dims = new unsigned int[3];
    dims = img->GetDimensions();
    int bytealign = 1;
    EImageFormat format = IF_LUMINANCE;
    EImageType type = IT_UNSIGNED_SHORT;

    unsigned int size = (dims[0] * dims[1] * dims[2]) * sizeof(unsigned char);

   ref<Image> img = new Image(dims[0], dims[1], dims[2], bytealign, format, type);
   memcpy(img->pixels(), cPointer, img->requiredMemory());

    // let's get started with the default volume!
    //setupVolume( loadImage("VLTest.dat") );
    m_RenderApplet->setupVolume( img );

  }
  catch(mitk::Exception& e)
  {
    // deal with the situation not to have access
  }

  
 


   // Update visibility settings
  //m_RenderWindow->update();
  UpdateDisplay();
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

  UpdateDisplay();

}

void NewVisualizationView::NodeRemoved(const mitk::DataNode* node)
{
  if (node == 0 || m_Controls == 0)
    return;

  UpdateDisplay();
}


void NewVisualizationView::UpdateDisplay(bool viewEnabled)
{
  //m_RenderWindow->update();
 
}
