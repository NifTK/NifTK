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

// VTK
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>

// MITK
#include <mitkImageReadAccessor.h>

// THIS IS VERY IMPORTANT
// If nothing is included from the mitk::OpenCL module the resource service will not get registered
#include <mitkOpenCLActivator.h>

using namespace vl;

VLRenderingApplet::VLRenderingApplet()
{
  m_FPSTimer.start();
  m_ThresholdVal = new Uniform( "val_threshold" );
  m_ThresholdVal->setUniformF( 0.5f );
  m_OclService = 0;
}

VLRenderingApplet::~VLRenderingApplet()
{
  MITK_INFO <<"Destructing render applet";
}

void VLRenderingApplet::initEvent()
{
  vl::Log::notify(appletInfo());

  mTransform1 = new Transform;
  mTransform2 = new Transform;
  rendering()->as<Rendering>()->transform()->addChild(mTransform1.get());
  rendering()->as<Rendering>()->transform()->addChild(mTransform2.get());

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

  // erase the scene
  sceneManager()->tree()->actors()->clear();

  m_Light = new Light;
  m_LightTr = new Transform;
  rendering()->as<Rendering>()->transform()->addChild( m_LightTr.get() );

  // scrap previous scene
  sceneManager()->tree()->eraseAllChildren();
  sceneManager()->tree()->actors()->clear();

  m_Light->setAmbient( fvec4( 0.1f, 0.1f, 0.1f, 1.0f ) );
  m_Light->setDiffuse( vl::white );
  m_Light->bindTransform( m_LightTr.get() );

  vec3 cameraPos = rendering()->as<vl::Rendering>()->camera()->modelingMatrix().getT();

  vec4 lightPos;
  lightPos[0] = cameraPos[0];
  lightPos[1] = cameraPos[1];
  lightPos[2] = cameraPos[2];
  lightPos[3] = 0;
  m_Light->setPosition(lightPos);

  ctkPluginContext* context = mitk::NewVisualizationPluginActivator::GetDefault()->GetPluginContext();

  ctkServiceReference serviceRef = context->getServiceReference<OclResourceService>();
  m_OclService = context->getService<OclResourceService>(serviceRef);

  if (m_OclService == NULL)
  {
    mitkThrow() << "Failed to find OpenCL resource service." << std::endl;
  }

  vl::OpenGLContext * glContext = openglContext();
  glContext->makeCurrent();

  // Force tests to run on the ATI GPU
  m_OclService->SpecifyPlatformAndDevice(0, 0, true);
}


void VLRenderingApplet::updateScene()
{
  mat4 cameraMatrix = rendering()->as<vl::Rendering>()->camera()->modelingMatrix();
  m_LightTr->setLocalMatrix(cameraMatrix);

  //mTransform1->setLocalMatrix( mat4::getRotation(Time::currentTime() * 5,      0, 1, 0 ) );
  //mTransform2->setLocalMatrix( mat4::getRotation(Time::currentTime() * 5 + 90, 0, 1, 0 ) );

  //mat4 mat;
  //// light 0 transform.
  //mat = mat4::getRotation( Time::currentTime()*43, 0,1,0 ) * mat4::getTranslation( 20,20,20 );
  //m_LightTr->setLocalMatrix( mat );

  return;

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


}

void VLRenderingApplet::keyPressEvent(unsigned short, EKey key)
{
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

void VLRenderingApplet::AddDataNode(mitk::DataNode::Pointer node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  // Propagate color and opacity down to basedata
  node->GetData()->SetProperty("color", node->GetProperty("color"));
  node->GetData()->SetProperty("opacity", node->GetProperty("opacity"));
  node->GetData()->SetProperty("visible", node->GetProperty("visible"));

  ref<Actor> newActor;
  std::string postFix;

  mitk::Image::Pointer mitkImg = dynamic_cast<mitk::Image *>(node->GetData());
  mitk::Surface::Pointer mitkSurf = dynamic_cast<mitk::Surface *>(node->GetData());

  if (mitkImg.IsNotNull())
  {
    newActor = AddImageActor(mitkImg);
    postFix.append("_image");
  }
  else if (mitkSurf.IsNotNull())
  {
    newActor = AddSurfaceActor(mitkSurf);
    postFix.append("_surface");
  }

  if (newActor.get() != 0)// && sceneManager()->tree()->actors()->find(newActor.get()) == -1)
  {
    std::string objName = newActor->objectName();
    objName.append(postFix);
    newActor->setObjectName(objName.c_str());
    m_NodeToActorMap[node] = newActor;
  }
}

void VLRenderingApplet::RemoveDataNode(mitk::DataNode::Pointer node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  std::map< mitk::DataNode::Pointer, ref<Actor> >::iterator it;

  it = m_NodeToActorMap.find(node);
  if (it == m_NodeToActorMap.end())
    return;

  ref<Actor> vlActor = it->second;
  if (vlActor.get() == 0)
    return;

  sceneManager()->tree()->eraseActor(vlActor.get());
  m_NodeToActorMap.erase(it);
}

void VLRenderingApplet::UpdateDataNode(mitk::DataNode::Pointer node)
{
  if (node.IsNull() || node->GetData() == 0)
    return;

  std::map< mitk::DataNode::Pointer, ref<Actor> >::iterator it;

  MITK_INFO <<m_NodeToActorMap.size();

  it = m_NodeToActorMap.find(node);
  if (it == m_NodeToActorMap.end())
    return;

  ref<Actor> vlActor = it->second;
  if (vlActor.get() == 0)
    return;

  mitk::BoolProperty  * visibleProp = dynamic_cast<mitk::BoolProperty *>( node->GetProperty("visible"));

  if (visibleProp->GetValue() == false)
  {
    vlActor->setEnableMask(0);
    return;
  }
  else
    vlActor->setEnableMask(0xFFFFFFFF);

  ref<Effect> fx = vlActor->effect();

  mitk::ColorProperty * colorProp = dynamic_cast<mitk::ColorProperty*>( node->GetProperty("color"));
  mitk::Color mitkColor = colorProp->GetColor();

  vl::fvec4 color;
  color[0] = mitkColor[0];
  color[1] = mitkColor[1];
  color[2] = mitkColor[2];

  //  SURFACE ONLY
  mitk::FloatProperty * opacityProp = dynamic_cast<mitk::FloatProperty *>( node->GetProperty("opacity"));
  float opacity = opacityProp->GetValue();
  //if (opacity == 1.0f)
  //{
  //  fx->shader()->enable(EN_DEPTH_TEST);
  //  fx->shader()->enable(EN_CULL_FACE);
  //  fx->shader()->enable(EN_LIGHTING);
  //  fx->shader()->setRenderState(m_Light.get(), 0 );
  //  fx->shader()->gocMaterial()->setDiffuse(color);
  //  fx->shader()->gocMaterial()->setTransparency(1.0f);
  //}
  //else
  {
    fx->shader()->enable(EN_BLEND);
    fx->shader()->enable(EN_DEPTH_TEST);
    fx->shader()->enable(EN_CULL_FACE);
    fx->shader()->enable(EN_LIGHTING);
    fx->shader()->setRenderState(m_Light.get(), 0 );
    fx->shader()->gocMaterial()->setDiffuse(color);
    fx->shader()->gocMaterial()->setTransparency(1.0f- opacity);
  }

}

ref<Actor> VLRenderingApplet::AddSurfaceActor(mitk::Surface::Pointer mitkSurf)
{
  ref<Geometry>  vlSurf = new Geometry();
  ref<Effect>    fx     = new Effect;
  ref<Transform> tr     = new Transform();

  ConvertVTKPolyData(mitkSurf->GetVtkPolyData(), vlSurf);

  MITK_INFO <<"Num of vertices: " <<vlSurf->vertexArray()->size();
  //ArrayAbstract* posarr = vertexArray() ? vertexArray() : vertexAttribArray(vl::VA_Position) ? vertexAttribArray(vl::VA_Position)->data() : NULL;
  if (!vlSurf->normalArray())
    vlSurf->computeNormals();

  //vl::ref<vl::ResourceDatabase> res_cortex;
  //res_cortex = vl::loadResource(vl::String("d://_boad_cortex2.stl"));
  //if ( res_cortex && res_cortex->count<vl::Geometry>() )
  //  vlSurf  = res_cortex->get<vl::Geometry>(0);

  //if (!vlSurf->normalArray())
  //  vlSurf->computeNormals();
  ////vlSurf->flipNormals();

  float opacity;
  mitkSurf->GetPropertyList()->GetFloatProperty("opacity", opacity);

  mitk::BaseProperty  * prop = mitkSurf->GetProperty("color");
  mitk::Color mitkColor = dynamic_cast<mitk::ColorProperty* >(prop)->GetColor();

  vl::fvec4 color;
  color[0] = mitkColor[0];
  color[1] = mitkColor[1];
  color[2] = mitkColor[2];
  color[3] = 1.0f- opacity;

  //  SURFACE ONLY

  //if (opacity == 1.0f)
  //{
  //  fx->shader()->enable(EN_DEPTH_TEST);
  //  fx->shader()->enable(EN_CULL_FACE);
  //  fx->shader()->enable(EN_LIGHTING);
  //  fx->shader()->setRenderState(m_Light.get(), 0 );
  //  fx->shader()->gocMaterial()->setDiffuse(color);
  //  fx->shader()->gocMaterial()->setTransparency(0.5f);
  //}
  //else
  {
    fx->shader()->enable(EN_BLEND);
    fx->shader()->enable(EN_DEPTH_TEST);
    fx->shader()->enable(EN_CULL_FACE);
    fx->shader()->enable(EN_LIGHTING);
    fx->shader()->setRenderState(m_Light.get(), 0 );
    fx->shader()->gocMaterial()->setDiffuse(color);
    fx->shader()->gocMaterial()->setTransparency(1.0f- opacity);
  }

  ref<Actor> surfActor = sceneManager()->tree()->addActor(vlSurf.get(), fx.get(), tr.get());

  //ref<RenderQueueSorterStandard> list_sorter = new RenderQueueSorterStandard;
  //list_sorter->setDepthSortMode(AlwaysDepthSort);
  //rendering()->as<Rendering>()->setRenderQueueSorter( list_sorter.get() );

  vtkLinearTransform * nodeVtkTr = mitkSurf->GetGeometry()->GetVtkTransform();
  vtkSmartPointer<vtkMatrix4x4> geometryTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  nodeVtkTr->GetMatrix(geometryTransformMatrix);

  float vals[16];
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      double val = geometryTransformMatrix->GetElement(i, j);
      vals[i*4+j] = val;
    }
  }

  vl::mat4 mat(vals);
  tr->setLocalMatrix(mat);

  trackball()->adjustView( sceneManager(), vl::vec3(0,0,1), vl::vec3(0,1,0), 1.0f );

  // refresh window
  openglContext()->update();

  m_ActorToRenderableMap[surfActor] = vlSurf;
  return surfActor;
}

ref<Actor> VLRenderingApplet::AddImageActor(mitk::Image::Pointer mitkImg)
{
  ref<Image>     vlImg;
  ref<Effect>    fx; 
  ref<Transform> tr;
  EImageFormat   format; 
  EImageType     type;
  
  mitk::PixelType pixType = mitkImg->GetPixelType();
  size_t numOfComponents = pixType.GetNumberOfComponents();

  /*
  mitk::PixelTypepixType = mitkImg->GetPixelType();
  std::cout << "Original pixel type:" << std::endl;
  std::cout << " PixelType: " <<pixType.GetTypeAsString() << std::endl;
  std::cout << " BitsPerElement: " <<pixType.GetBpe() << std::endl;
  std::cout << " NumberOfComponents: " <<pixType.GetNumberOfComponents() << std::endl;
  std::cout << " BitsPerComponent: " <<pixType.GetBitsPerComponent() << std::endl;
*/
  try
  {
    mitk::ImageReadAccessor readAccess(mitkImg, mitkImg->GetVolumeData(0));
    const void* cPointer = readAccess.GetData();
    unsigned int * dims = new unsigned int[3];
    dims = mitkImg->GetDimensions();
    int bytealign = 1;

    if (pixType.GetComponentType() == itk::ImageIOBase::CHAR )
    {
      type = IT_BYTE;
    }
    else if (pixType.GetComponentType() == itk::ImageIOBase::UCHAR)
    {
      type = IT_UNSIGNED_BYTE;
    }
    else if ( pixType.GetComponentType() == itk::ImageIOBase::SHORT )
    {
      type = IT_SHORT;
    }
    else if (  pixType.GetComponentType() == itk::ImageIOBase::USHORT )
    {
      type = IT_UNSIGNED_SHORT;
    }
    else if ( pixType.GetComponentType() == itk::ImageIOBase::INT )
    {
      type = IT_INT;
    }
    else if (  pixType.GetComponentType() == itk::ImageIOBase::UINT )
    {
      type = IT_UNSIGNED_INT;
    }
    else if ( pixType.GetComponentType() == itk::ImageIOBase::FLOAT  )
    {
      type = IT_FLOAT;
    }

    if (type != IT_FLOAT)
    {
      if (numOfComponents == 1)
        format = IF_LUMINANCE;
      else if (numOfComponents == 2)
        format = IF_RG_INTEGER;
      else if (numOfComponents == 3)
        format = IF_RGB_INTEGER;
      else if (numOfComponents == 4)
        format = IF_RGBA_INTEGER;
    }
    else if (type == IT_FLOAT)
    {
      if (numOfComponents == 1)
        format = IF_LUMINANCE;
      else if (numOfComponents == 2)
        format = IF_RG;
      else if (numOfComponents == 3)
        format = IF_RGB;
      else if (numOfComponents == 4)
        format = IF_RGBA;
    }

    unsigned int size = (dims[0] * dims[1] * dims[2]) * sizeof(pixType.GetSize());

    vlImg = new Image(dims[0], dims[1], dims[2], bytealign, format, type);
    memcpy(vlImg->pixels(), cPointer, vlImg->requiredMemory());

    vlImg = vlImg->convertFormat(vl::IF_LUMINANCE)->convertType(vl::IT_UNSIGNED_SHORT);
/*
    ref<KeyValues> tags = new KeyValues;
    tags->set("Origin")    = Say("%n %n %n") << mitkImg->GetGeometry()->GetOrigin()[0]  << mitkImg->GetGeometry()->GetOrigin()[1]  << mitkImg->GetGeometry()->GetOrigin()[2];
    tags->set("Spacing")   = Say("%n %n %n") << mitkImg->GetGeometry()->GetSpacing()[0] << mitkImg->GetGeometry()->GetSpacing()[1] << mitkImg->GetGeometry()->GetSpacing()[2];
    vlImg->setTags(tags.get());
*/
  }
  catch(mitk::Exception& e)
  {
    // deal with the situation not to have access
  }

  //vlImg = loadImage( "e:/Niftike-r/VL-src/data/volume/VLTest.dat" );

  fx = new Effect;
  tr = new Transform();
  
  float opacity;
  mitkImg->GetPropertyList()->GetFloatProperty("opacity", opacity);

  mitk::BaseProperty  * prop = mitkImg->GetProperty("color");
  mitk::Color mitkColor = dynamic_cast<mitk::ColorProperty* >(prop)->GetColor();

  vl::fvec4 color;
  color[0] = mitkColor[0];
  color[1] = mitkColor[1];
  color[2] = mitkColor[2];

  fx->shader()->enable(EN_DEPTH_TEST);
  fx->shader()->enable(EN_BLEND);
  fx->shader()->setRenderState(m_Light.get(), 0 );
  fx->shader()->enable(EN_LIGHTING);
  fx->shader()->gocMaterial()->setDiffuse(color);
  fx->shader()->gocMaterial()->setTransparency(1.0f- opacity);

  vl::String vlFsSource;
  LoadGLSLSourceFromResources("volume_raycast_isosurface_transp.fs", vlFsSource);

  vl::String vlVsSource;
  LoadGLSLSourceFromResources("volume_luminance_light.vs", vlVsSource);

  // The GLSL program used to perform the actual rendering.
  // The \a volume_luminance_light.fs fragment shader allows you to specify how many 
  // lights to use (up to 4) and can optionally take advantage of a precomputed normals texture.
  ref<GLSLProgram> glslShader;
  glslShader = fx->shader()->gocGLSLProgram();
  glslShader->attachShader( new GLSLFragmentShader( vlFsSource));
  glslShader->attachShader( new GLSLVertexShader(vlVsSource) );

  ref<Actor> imageActor = new Actor;
  imageActor->setEffect(fx.get() );
  imageActor->setTransform(tr.get());
  sceneManager()->tree()->addActor( imageActor.get() );
  imageActor->setUniform( m_ThresholdVal.get() );

  ref<vl::RaycastVolume> mRaycastVolume;
  mRaycastVolume = new vl::RaycastVolume;
  mRaycastVolume->bindActor( imageActor.get() );
  //mitkImg->GetDimensions();
  //AABB volume_box( vec3( -10,-10,-10 ), vec3( +10,+10,+10 ) );
  //mRaycastVolume->setBox( volume_box );


  unsigned int * dims = mitkImg->GetDimensions();
  const float * spacing = const_cast<float *>(mitkImg->GetGeometry()->GetFloatSpacing());
  
  //MITK_INFO <<"DIMMMSS: "<<dims[0] <<" " <<dims[1] <<" " <<dims[2];
  float dimX = (float)dims[0]*spacing[0] / 2.0f;
  float dimY = (float)dims[1]*spacing[1] / 2.0f;
  float dimZ = (float)dims[2]*spacing[2] / 2.0f;

  float shiftX = 0.0f;//0.5f * spacing[0];
  float shiftY = 0.0f;//0.5f * spacing[1];
  float shiftZ = 0.0f;//0.5f * spacing[2];

  //MITK_INFO <<"DIMMMSS: "<<dimX <<" " <<dimY <<" " <<dimZ;
  AABB volume_box( vec3( -dimX+shiftX,-dimY+shiftY,-dimZ+shiftZ), vec3(dimX+shiftX,dimY+shiftY,dimZ+shiftZ) );
  mRaycastVolume->setBox( volume_box );


  // Setup image
  
/*
  ref<Image> gradient;
  // note that this can take a while...
  gradient = vl::genGradientNormals( vlImg.get() );
*/
  fx = imageActor->effect();

  // install volume image as textue #0
  //if (format == IF_LUMINANCE_INTEGER)
  //  fx->shader()->gocTextureSampler( 0 )->setTexture( new vl::Texture( vlImg.get(), TF_LUMINANCE8, false, false ) );
  //else if (format == IF_RG_INTEGER)
  //  fx->shader()->gocTextureSampler( 0 )->setTexture( new vl::Texture( vlImg.get(), TF_LUMINANCE16UI_EXT, false, false ) );
  //else if (format == IF_LUMINANCE)
  //  fx->shader()->gocTextureSampler( 0 )->setTexture( new vl::Texture( vlImg.get(), TF_LUMINANCE, false, false ) );

  fx->shader()->gocTextureSampler( 0 )->setTexture( new vl::Texture( vlImg.get(), TF_LUMINANCE8, false, false ) );

  fx->shader()->gocUniform( "volume_texunit" )->setUniformI( 0 );
  mRaycastVolume->generateTextureCoordinates( ivec3(vlImg->width(), vlImg->height(), vlImg->depth()) );

  // generate a simple colored transfer function
  ref<Image> trfunc;
  trfunc = vl::makeColorSpectrum( 1024, vl::blue, vl::royalblue, vl::green, vl::yellow, vl::crimson );

  // installs the transfer function as texture #1
  fx->shader()->gocTextureSampler( 1 )->setTexture( new Texture( trfunc.get() ) );
  fx->shader()->gocUniform( "trfunc_texunit" )->setUniformI( 1 );
/*
  fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( 1);
  fx->shader()->gocTextureSampler( 2 )->setTexture( new Texture( gradient.get(), TF_RGBA, false, false ) );
  fx->shader()->gocUniform( "gradient_texunit" )->setUniformI( 2 );
*/
  fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( 0 );
  // used to compute on the fly the normals based on the volume's gradient
  fx->shader()->gocUniform( "gradient_delta" )->setUniform( fvec3( 0.5f/vlImg->width(), 0.5f/vlImg->height(), 0.5f/vlImg->depth() ) );
  
  fx->shader()->gocUniform( "sample_step" )->setUniformF(1.0f / 512.0f);

  vtkLinearTransform * nodeVtkTr = mitkImg->GetGeometry()->GetVtkTransform();
  vtkSmartPointer<vtkMatrix4x4> geometryTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  nodeVtkTr->GetMatrix(geometryTransformMatrix);

  float vals[16];
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      double val = geometryTransformMatrix->GetElement(i, j);
      vals[i*4+j] = val;
    }
  }
  vl::mat4 mat(vals);
  tr->setLocalMatrix(mat);

  // refresh window
  openglContext()->update();

  m_ActorToRenderableMap[imageActor] = dynamic_cast<Renderable*>( vlImg.get());
  return imageActor;
}

void VLRenderingApplet::UpdateThresholdVal( int val )
{
  float val_threshold = 0.0f;
  m_ThresholdVal->getUniform( &val_threshold );
  val_threshold = val / 10000.0f;
  val_threshold = clamp( val_threshold, 0.0f, 1.0f );
  m_ThresholdVal->setUniformF( val_threshold );
  openglContext()->update();
}

void VLRenderingApplet::LoadGLSLSourceFromResources(const char* filename, vl::String &vlStringSource)
{

  // Check if writing to 3D textures are supported
  QString sourceFilename(filename);
  sourceFilename.prepend(":/NewVisualization/");
  QFile sourceFile;
  sourceFile.setFileName(sourceFilename);
  
  // Read source file
  std::string sourceCode;

  if (sourceFile.exists() && sourceFile.open(QIODevice::ReadOnly))
  {
    // Make a text stream on the doco  
    QTextStream textStream(&sourceFile);

    // Read all the contents
    QString qContents = textStream.readAll();
    sourceCode = qContents.toStdString();
  }
  else
  {
    //error
    MITK_ERROR << "Failed to open OpenCL source file" << std::endl;
    return;
  }

  vlStringSource = vl::String(sourceCode.c_str());
}

void VLRenderingApplet::ConvertVTKPolyData(vtkPolyData * vtkPoly, ref<vl::Geometry> vlPoly)
{

  if (vtkPoly == 0)
    return;

  /// \brief Buffer in host memory to store cell info
  unsigned int * m_IndexBuffer = 0;
  
  /// \brief Buffer in host memory to store vertex points
  float * m_PointBuffer = 0;
  
  /// \brief Buffer in host memory to store normals associated with vertices
  float * m_NormalBuffer = 0;

  /// \brief Buffer in host memory to store scalar info associated with vertices
  char * m_ScalarBuffer = 0;


  unsigned int numOfvtkPolyPoints = vtkPoly->GetNumberOfPoints();

  // A polydata will always have point data
  int pointArrayNum = vtkPoly->GetPointData()->GetNumberOfArrays();

  if (pointArrayNum == 0 && numOfvtkPolyPoints == 0)
  {
    MITK_ERROR <<"No points detected in the vtkPoly data!\n";
    return;
  }

  // We'll have to build the cell data if not present already
  int cellArrayNum  = vtkPoly->GetCellData()->GetNumberOfArrays();
  if (cellArrayNum == 0)
    vtkPoly->BuildCells();

  vtkSmartPointer<vtkCellArray> verts;

  // Try to get access to cells
  if (vtkPoly->GetVerts() != 0 && vtkPoly->GetVerts()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetVerts();
  else if (vtkPoly->GetLines() != 0 && vtkPoly->GetLines()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetLines();
  else if (vtkPoly->GetPolys() != 0 && vtkPoly->GetPolys()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetPolys();
  else if (vtkPoly->GetStrips() != 0 && vtkPoly->GetStrips()->GetNumberOfCells() != 0)
    verts = vtkPoly->GetStrips();

  if (verts->GetMaxCellSize() > 4)
  {
    // Panic and return
    MITK_ERROR <<"More than four vertices / cell detected, can't handle this data type!\n";
    return;
  }
  
  vtkSmartPointer<vtkPoints>     points = vtkPoly->GetPoints();

  if (points == 0)
  {
    MITK_ERROR <<"Corrupt vtkPoly, returning! \n";
    return;
  }

  // Deal with normals
  vtkSmartPointer<vtkDataArray> normals = vtkPoly->GetPointData()->GetNormals();

  if (normals == 0)
  {
    MITK_INFO <<"Generating normals for the vtkPoly data (mitk::OclSurface)";
    
    vtkSmartPointer<vtkPolyDataNormals> normalGen = vtkSmartPointer<vtkPolyDataNormals>::New();
    normalGen->SetInputData(vtkPoly);
    normalGen->AutoOrientNormalsOn();
    normalGen->Update();

    normals = normalGen->GetOutput()->GetPointData()->GetNormals();

    if (normals == 0)
    {
      MITK_ERROR <<"Couldn't generate normals, returning! \n";
      return;
    }

    vtkPoly->GetPointData()->SetNormals(normals);
    vtkPoly->GetPointData()->GetNormals()->Modified();
    vtkPoly->GetPointData()->Modified();
  }
 
  // Check if we have scalars
  vtkSmartPointer<vtkDataArray> scalars = vtkPoly->GetPointData()->GetScalars();

  bool pointsValid  = (points.GetPointer() == 0) ? false : true;
  bool normalsValid = (normals.GetPointer() == 0) ? false : true;
  bool scalarsValid = (scalars.GetPointer() == 0) ? false : true;
  
  unsigned int pointBufferSize = 0;
  unsigned int numOfPoints = static_cast<unsigned int> (points->GetNumberOfPoints());
  pointBufferSize = numOfPoints * sizeof(float) *3;

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with points


  // Allocate memory
  m_PointBuffer = new float[numOfPoints*3];

  // Copy data to buffer
  memcpy(m_PointBuffer, points->GetVoidPointer(0), pointBufferSize);

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with normals

  if (normalsValid)
  {
    // Get the number of normals we have to deal with
    int m_NormalCount = static_cast<unsigned int> (normals->GetNumberOfTuples());

    // Size of the buffer that is required to store all the normals
    unsigned int normalBufferSize = numOfPoints * sizeof(float) * 3;

    // Allocate memory
    m_NormalBuffer = new float[numOfPoints*3];

    // Copy data to buffer
    memcpy(m_NormalBuffer, normals->GetVoidPointer(0), normalBufferSize);
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with scalars (colors or texture coordinates)
  if (scalarsValid)
  {

    // Get the number of scalars we have to deal with
    int m_ScalarCount = static_cast<unsigned int> (scalars->GetNumberOfTuples());

    // Size of the buffer that is required to store all the scalars
    unsigned int scalarBufferSize = numOfPoints * sizeof(char) * 1;

    // Allocate memory
    m_ScalarBuffer = new char[numOfPoints];

    // Copy data to buffer
    memcpy(m_ScalarBuffer, scalars->GetVoidPointer(0), scalarBufferSize);
  }


  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Deal with cells - initialize index buffer
  vtkIdType npts;
  vtkIdType *pts;

  // Get the number of indices we have to deal with
  unsigned int m_IndexCount = static_cast<unsigned int> (verts->GetNumberOfCells());

  // Get the max number of vertices / cell
  int maxPointsPerCell = verts->GetMaxCellSize();


  // Get the number of indices we have to deal with
  unsigned int numOfTriangles = static_cast<unsigned int> (verts->GetNumberOfCells());

  // Allocate memory for the index buffer
  m_IndexBuffer = new unsigned int[numOfTriangles*3];
  memset(m_IndexBuffer, 0, numOfTriangles*3*sizeof(unsigned int));

  verts->InitTraversal();

  unsigned int cellIndex = 0;
  // Iterating through all the cells
  while (cellIndex < numOfTriangles)
  {
    verts->GetNextCell(npts, pts);

    // Copy the indices into the index buffer
    for (size_t i = 0; i < static_cast<size_t>(npts); i++)
      m_IndexBuffer[cellIndex*3 +i] = pts[i];

    cellIndex++;
  }
  MITK_INFO <<"Surface data initialized. Num of Points: " <<points->GetNumberOfPoints() <<" Num of Cells: " <<verts->GetNumberOfCells() <<"\n";

  ref<ArrayFloat3>  vlVerts   = new ArrayFloat3;
  ref<ArrayFloat3>  vlNormals = new ArrayFloat3;

  //vlVerts->resize(numOfPoints *3);
  //vlNormals->resize(numOfPoints *3);
  //ref<DrawArrays> de = new DrawArrays(PT_TRIANGLES,0,numOfPoints*3);

  vlVerts->resize(numOfTriangles *3);
  vlNormals->resize(numOfTriangles *3);
  ref<DrawArrays> de = new DrawArrays(PT_TRIANGLES,0,numOfTriangles*3);
   
  vlPoly->drawCalls()->push_back(de.get());
  vlPoly->setVertexArray(vlVerts.get());
  vlPoly->setNormalArray(vlNormals.get());

/*
    // read triangles
  for(unsigned int i=0; i<numOfPoints; ++i)
  {
    fvec3 n0, n1, n2, v1,v2,v0;
    n0.x() = m_NormalBuffer[i*3 +0];
    n0.y() = m_NormalBuffer[i*3 +1];
    n0.z() = m_NormalBuffer[i*3 +2];
    v0.x() = m_PointBuffer[i*3 +0];
    v0.y() = m_PointBuffer[i*3 +1];
    v0.z() = m_PointBuffer[i*3 +2];

    vlNormals->at(i*3+0) = n0;
    vlVerts->at(i*3+0) = v0;
  }
*/

  // read triangles
  for(unsigned int i=0; i<numOfTriangles; ++i)
  {
    fvec3 n0, n1, n2, v1,v2,v0;
    unsigned int vertIndex = m_IndexBuffer[i*3 +0];
    n0.x() = m_NormalBuffer[vertIndex*3 +0];
    n0.y() = m_NormalBuffer[vertIndex*3 +1];
    n0.z() = m_NormalBuffer[vertIndex*3 +2];
    v0.x() = m_PointBuffer[vertIndex*3 +0];
    v0.y() = m_PointBuffer[vertIndex*3 +1];
    v0.z() = m_PointBuffer[vertIndex*3 +2];

    vertIndex = m_IndexBuffer[i*3 +1];
    n1.x() = m_NormalBuffer[vertIndex*3 +0];
    n1.y() = m_NormalBuffer[vertIndex*3 +1];
    n1.z() = m_NormalBuffer[vertIndex*3 +2];
    v1.x() = m_PointBuffer[vertIndex*3 +0];
    v1.y() = m_PointBuffer[vertIndex*3 +1];
    v1.z() = m_PointBuffer[vertIndex*3 +2];

    vertIndex = m_IndexBuffer[i*3 +2];
    n2.x() = m_NormalBuffer[vertIndex*3 +0];
    n2.y() = m_NormalBuffer[vertIndex*3 +1];
    n2.z() = m_NormalBuffer[vertIndex*3 +2];
    v2.x() = m_PointBuffer[vertIndex*3 +0];
    v2.y() = m_PointBuffer[vertIndex*3 +1];
    v2.z() = m_PointBuffer[vertIndex*3 +2];

    vlNormals->at(i*3+0) = n0;
    vlVerts->at(i*3+0) = v0;
    vlNormals->at(i*3+1) = n1;
    vlVerts->at(i*3+1) = v1;
    vlNormals->at(i*3+2) = n2;
    vlVerts->at(i*3+2) = v2;
  }

  /// \brief Buffer in host memory to store cell info
  if (m_IndexBuffer != 0)
    delete m_IndexBuffer;
  
  /// \brief Buffer in host memory to store vertex points
  if (m_PointBuffer != 0)
    delete m_PointBuffer;
  
  /// \brief Buffer in host memory to store normals associated with vertices
  if (m_NormalBuffer != 0)
    delete m_NormalBuffer;

  /// \brief Buffer in host memory to store scalar info associated with vertices
  if (m_ScalarBuffer != 0)
    delete m_ScalarBuffer;


  MITK_INFO <<"Num of VL vertices: " <<vlPoly->vertexArray()->size();
}


