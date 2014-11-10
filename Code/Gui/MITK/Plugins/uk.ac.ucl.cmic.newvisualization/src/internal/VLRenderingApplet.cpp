/**************************************************************************************/
/*                                                                                    */
/*  Visualization Library                                                             */
/*  http://www.visualizationlibrary.org                                               */
/*                                                                                    */
/*  Copyright (c) 2005-2010, Michele Bosi                                             */
/*  All rights reserved.                                                              */
/*                                                                                    */
/*  Redistribution and use in source and binary forms, with or without modification,  */
/*  are permitted provided that the following conditions are met:                     */
/*                                                                                    */
/*  - Redistributions of source code must retain the above copyright notice, this     */
/*  list of conditions and the following disclaimer.                                  */
/*                                                                                    */
/*  - Redistributions in binary form must reproduce the above copyright notice, this  */
/*  list of conditions and the following disclaimer in the documentation and/or       */
/*  other materials provided with the distribution.                                   */
/*                                                                                    */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            */
/*  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR  */
/*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    */
/*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      */
/*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON    */
/*  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     */
/*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      */
/*                                                                                    */
/**************************************************************************************/


#include "VLRenderingApplet.h"
#include "NewVisualizationPluginActivator.h"

// VL
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/Text.hpp>
#include <vlGraphics/FontManager.hpp>
#include <vlCore/String.hpp>
#include <vlGraphics/Camera.hpp>
#include <vlGraphics/RenderingTree.hpp>

// VTK
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>

// MITK


// THIS IS VERY IMPORTANT
// If nothing is included from the mitk::OpenCL module the resource service will not get registered
#include <mitkOpenCLActivator.h>

//using namespace vl;

VLRenderingApplet::VLRenderingApplet(vl::OpenGLContext* context)
{
//  m_FPSTimer.start();
  m_ThresholdVal = new vl::Uniform( "val_threshold" );
  m_ThresholdVal->setUniformF( 0.5f );
  m_OclService = 0;

  // base-class non-virtual method!
  initialize();
}

VLRenderingApplet::~VLRenderingApplet()
{
  MITK_INFO <<"Destructing render applet";
}

void VLRenderingApplet::initEvent()
{
  vl::Log::notify(appletInfo());

//  mTransform1 = new Transform;
//  mTransform2 = new Transform;
//  rendering()->as<Rendering>()->transform()->addChild(mTransform1.get());
//  rendering()->as<Rendering>()->transform()->addChild(mTransform2.get());

/*
  mText = new Text;
  mText->setText("...");
  mText->setFont( defFontManager()->acquireFont (":/NewVisualization/VeraMono.ttf", 10) );
  mText->setAlignment( AlignHCenter | AlignTop );
  mText->setViewportAlignment( AlignHCenter| AlignTop );
  mText->setTextAlignment(TextAlignLeft);
  mText->translate(0,-5,0);
  mText->setColor(white);
  mText->setBackgroundColor(black);
  mText->setBackgroundEnabled(true);
  */
}


void VLRenderingApplet::updateScene()
{


  //mTransform1->setLocalMatrix( mat4::getRotation(Time::currentTime() * 5,      0, 1, 0 ) );
  //mTransform2->setLocalMatrix( mat4::getRotation(Time::currentTime() * 5 + 90, 0, 1, 0 ) );

  //mat4 mat;
  //// light 0 transform.
  //mat = mat4::getRotation( Time::currentTime()*43, 0,1,0 ) * mat4::getTranslation( 20,20,20 );
  //m_LightTr->setLocalMatrix( mat );

  return;

#if 0
  ref<ActorCollection> actors = sceneManager()->tree()->actors();
  int numOfActors = actors->size();

  for (int i = 0; i < numOfActors; i++)
  {
    ref<Actor> act = actors->at(i);
    std::string objName = act->objectName();
    
    size_t found =objName.find("_surface");
    if (found != std::string::npos)
    {
      ref<Renderable> ren = m_ActorToRenderableMap[act];
      ref<Geometry> surface = dynamic_cast<Geometry*>(ren.get());
      if (surface == 0)
        continue;

      
      // Get context 
      cl_context clContext = m_OclService->GetContext();
      cl_command_queue clCmdQue = m_OclService->GetCommandQueue();

      cl_int clErr = 0;
      GLuint bufferHandle = surface->vertexArray()->bufferObject()->handle();
      cl_mem clBuf = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, bufferHandle, &clErr);

      clEnqueueAcquireGLObjects(clCmdQue, 1, &clBuf, 0, NULL, NULL);

      // Do something

      clEnqueueReleaseGLObjects(clCmdQue, 1, &clBuf, 0, NULL, NULL);

    }
  }
#endif

}


void VLRenderingApplet::updateEvent()
{
  vl::Applet::updateEvent();

  if ( m_FPSTimer.elapsed() > 2 )
  {
    m_FPSTimer.start();
    openglContext()->setWindowTitle( vl::Say("[%.1n] %s") << fps() << appletName()  + " - " + vl::String("VL ") + vl::VisualizationLibrary::versionString() );
    vl::Log::print( vl::Say("FPS=%.1n\n") << fps() );
  }
}

vl::String VLRenderingApplet::appletInfo()
{
  return "Applet info: " + appletName() + "\n" +
  "Keys:\n" +
  "- Escape: quits the application.\n" +
  "- T:  enables the TrackballManipulator.\n" +
  "- F:  enables the GhostCameraManipulator (use A/D S/W keys).\n" +
  "- F1: toggles fullscreen mode if supported.\n" +
  "- F5: saves a screenshot of the current OpenGL window.\n" +
  "- C:  toggles the continuous update of the OpenGL window.\n" +
  "- U:  force update of the OpenGL window.\n" +
  "\n";
}



void VLRenderingApplet::UpdateThresholdVal( int val )
{

}






