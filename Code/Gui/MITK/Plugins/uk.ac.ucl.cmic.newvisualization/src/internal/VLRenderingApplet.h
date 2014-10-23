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

#ifndef VLRenderingApplet_INCLUDE_ONCE
#define VLRenderingApplet_INCLUDE_ONCE

// VL
#include <vlCore/ResourceDatabase.hpp>
#include <vlCore/Time.hpp>
#include <vlCore/VisualizationLibrary.hpp>

#include <vlGraphics/Applet.hpp>
#include <vlGraphics/Rendering.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/Light.hpp>
#include <vlGraphics/Text.hpp>
#include <vlGraphics/FontManager.hpp>
#include <vlGraphics/GLSL.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>

#include <vlVolume/RaycastVolume.hpp>
#include <vlVolume/VolumeUtils.hpp>


// MITK
#include <mitkDataNode.h>
#include <mitkImage.h>
#include <mitkSurface.h>
#include <mitkBaseData.h>
#include <mitkProperties.h>

// Microservices
#include <usModuleContext.h>
#include <usGetModuleContext.h>
#include <usModule.h>
#include <usModuleResource.h>
#include <usModuleResourceStream.h>

// VTK
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataNormals.h>
#include <vtkCleanPolyData.h>

using namespace vl;

class VLRenderingApplet: public vl::Applet
{
public:
  VLRenderingApplet();
  virtual ~VLRenderingApplet() {}

  void initEvent();
  void updateEvent();
  virtual vl::String appletInfo();

  void updateScene();

  void keyPressEvent(unsigned short, EKey key);

  void AddDataNode(mitk::DataNode::Pointer node);
  void RemoveDataNode(mitk::DataNode::Pointer node);
  void UpdateDataNode(mitk::DataNode::Pointer node);

  void UpdateThresholdVal( int val );

protected:
  typedef std::vector<const char*> CStringList;
  typedef std::vector<size_t> ClSizeList;

private:
  ref<Actor> AddImageActor(mitk::Image::Pointer mitkImg);
  ref<Actor> AddSurfaceActor(mitk::Surface::Pointer mitkSurf);
  void LoadGLSLSourceFromResources(const char* filename, vl::String &source);

  void ConvertVTKPolyData(vtkPolyData * vtkPoly, ref<vl::Geometry> vlPoly);

private:
  Time           m_FPSTimer;
  ref<Uniform>   m_ThresholdVal;
  ref<Text>      mText;
  ref<Transform> mTransform1;
  ref<Transform> mTransform2;
  ref<Light>     m_Light;
  ref<Transform> m_LightTr;

  std::map< mitk::DataNode::Pointer, ref<Actor>  > m_NodeToActorMap;
};

#endif
