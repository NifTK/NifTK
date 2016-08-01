/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "VLQtWidget.h"

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include <vlGraphics/OpenGL.hpp>
#include "VLRendererView.h"
#include "VLRendererPluginActivator.h"

#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkImageReadAccessor.h>
#include <mitkDataStorageUtils.h>
#include <mitkNodePredicateDataType.h>
#include <mitkNodePredicateOr.h>
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

#include <niftkSharedOGLContext.h>

#ifdef _USE_CUDA
#include <niftkCUDAImage.h>
#include <niftkEdgeDetectionExampleLauncher.h>
#endif

//-----------------------------------------------------------------------------
const std::string VLRendererView::VIEW_ID = "uk.ac.ucl.cmic.vlrenderer";


//-----------------------------------------------------------------------------
VLRendererView::VLRendererView()
: m_Controls(0)
, m_Parent(0)
, m_VLQtRenderWindow(0)
{
}


//-----------------------------------------------------------------------------
VLRendererView::~VLRendererView()
{
  MITK_INFO << "Destructing VLRenderer plugin";
}

//-----------------------------------------------------------------------------

void VLRendererView::SetFocus()
{
}

//-----------------------------------------------------------------------------
void VLRendererView::CreateQtPartControl(QWidget* parent)
{
  // setup the basic GUI of this view
  m_Parent = parent;

  if (!m_Controls)
  {
    // Create UI.
    m_Controls = new Ui::VLRendererViewControls();
    m_Controls->setupUi(parent);

    bool ok = false;
    ok = QObject::connect(m_Controls->hSlider_navigate, SIGNAL(valueChanged(int )), this, SLOT(On_SliderMoved(int )));
    assert(ok);

    m_Controls->m_BackgroundNode->SetDataStorage(GetDataStorage());
    m_Controls->m_BackgroundNode->SetAutoSelectNewItems(false);
    mitk::TNodePredicateDataType<mitk::Image>::Pointer      isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
#ifdef _USE_CUDA
    mitk::TNodePredicateDataType<niftk::CUDAImage>::Pointer isCuda = mitk::TNodePredicateDataType<niftk::CUDAImage>::New();
    mitk::NodePredicateOr::Pointer                          isSuitable = mitk::NodePredicateOr::New(isImage, isCuda);
    m_Controls->m_BackgroundNode->SetPredicate(isSuitable);
#else
    m_Controls->m_BackgroundNode->SetPredicate(isImage);
#endif
    ok = QObject::connect(m_Controls->m_BackgroundNode, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnBackgroundNodeSelected(const mitk::DataNode*)));
    assert(ok);

    m_Controls->m_CameraNode->SetDataStorage(GetDataStorage());
    m_Controls->m_CameraNode->SetAutoSelectNewItems(false);
    ok = QObject::connect(m_Controls->m_CameraNode, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnCameraNodeSelected(const mitk::DataNode*)));
    assert(ok);

    ok = QObject::connect(m_Controls->m_CameraNodeEnabled, SIGNAL(clicked(bool)), this, SLOT(OnCameraNodeEnabled(bool)));
    assert(ok);

    // Init the VL visualization part
    InitVLRendering();
  }
}

//-----------------------------------------------------------------------------

void VLRendererView::InitVLRendering()
{
  assert(m_VLQtRenderWindow == 0);
  m_VLQtRenderWindow = new VLQtWidget( 0, niftk::SharedOGLContext::GetShareWidget() );
  m_VLQtRenderWindow->vlSceneView()->setDataStorage(GetDataStorage());

  // renderer uses ocl kernels to sort triangles.
  ctkPluginContext*   context     = mitk::VLRendererPluginActivator::GetDefault()->GetPluginContext();
  ctkServiceReference serviceRef  = context->getServiceReference<OclResourceService>();
  OclResourceService* oclService  = context->getService<OclResourceService>(serviceRef);
  if (oclService == NULL)
  {
    mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
  }
  m_VLQtRenderWindow->vlSceneView()->setOclResourceService(oclService);
  // note: m_VLQtRenderWindow will use that service instance in initializeGL(), which will only be called
  // once we have been bounced through the event-loop, i.e. after we return from this method here.


  m_Controls->viewLayout->addWidget(m_VLQtRenderWindow);
  m_VLQtRenderWindow->setSizePolicy(QSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding));
  m_VLQtRenderWindow->show();

  // This state seems to be left dirty so we reset it to it's default else VL will complain.
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

//-----------------------------------------------------------------------------

void VLRendererView::On_SliderMoved(int val)
{
  m_VLQtRenderWindow->vlSceneView()->updateThresholdVal(val);
  m_VLQtRenderWindow->update();
}

//-----------------------------------------------------------------------------
void VLRendererView::OnBackgroundNodeSelected(const mitk::DataNode* node)
{
  m_VLQtRenderWindow->vlSceneView()->setBackgroundNode(node);
  // can fail, but we just ignore that.
}

//-----------------------------------------------------------------------------

void VLRendererView::OnCameraNodeSelected(const mitk::DataNode* node)
{
  OnCameraNodeEnabled(m_Controls->m_CameraNodeEnabled->isChecked());
}

//-----------------------------------------------------------------------------

void VLRendererView::OnCameraNodeEnabled(bool enabled)
{
  if (!enabled)
  {
    m_VLQtRenderWindow->vlSceneView()->setCameraTrackingNode( 0 );
  }
  else
  {
    mitk::DataNode::Pointer node = m_Controls->m_CameraNode->GetSelectedNode();
    m_VLQtRenderWindow->vlSceneView()->setCameraTrackingNode( node.GetPointer() );
  }
}

//-----------------------------------------------------------------------------

void VLRendererView::Visible()
{
  QmitkBaseView::Visible();

  // Make sure that we show all the nodes that are already present in DataStorage
  ReinitDisplay();
}

//-----------------------------------------------------------------------------

void VLRendererView::ReinitDisplay(bool viewEnabled)
{
  m_VLQtRenderWindow->vlSceneView()->scheduleSceneRebuild();
}
