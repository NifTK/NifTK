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
#include <mitkImageReadAccessor.h>

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


vl::ref<vl::Actor> VLRenderingApplet::AddImageActor(mitk::Image::Pointer mitkImg)
{
  vl::ref<vl::Image>     vlImg;
  vl::ref<vl::Effect>    fx; 
  vl::ref<vl::Transform> tr;
  vl::EImageFormat   format; 
  vl::EImageType     type;
  
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
      type = vl::IT_BYTE;
    }
    else if (pixType.GetComponentType() == itk::ImageIOBase::UCHAR)
    {
      type = vl::IT_UNSIGNED_BYTE;
    }
    else if ( pixType.GetComponentType() == itk::ImageIOBase::SHORT )
    {
      type = vl::IT_SHORT;
    }
    else if (  pixType.GetComponentType() == itk::ImageIOBase::USHORT )
    {
      type = vl::IT_UNSIGNED_SHORT;
    }
    else if ( pixType.GetComponentType() == itk::ImageIOBase::INT )
    {
      type = vl::IT_INT;
    }
    else if (  pixType.GetComponentType() == itk::ImageIOBase::UINT )
    {
      type = vl::IT_UNSIGNED_INT;
    }
    else if ( pixType.GetComponentType() == itk::ImageIOBase::FLOAT  )
    {
      type = vl::IT_FLOAT;
    }

    if (type != vl::IT_FLOAT)
    {
      if (numOfComponents == 1)
        format = vl::IF_LUMINANCE;
      else if (numOfComponents == 2)
        format = vl::IF_RG_INTEGER;
      else if (numOfComponents == 3)
        format = vl::IF_RGB_INTEGER;
      else if (numOfComponents == 4)
        format = vl::IF_RGBA_INTEGER;
    }
    else if (type == vl::IT_FLOAT)
    {
      if (numOfComponents == 1)
        format = vl::IF_LUMINANCE;
      else if (numOfComponents == 2)
        format = vl::IF_RG;
      else if (numOfComponents == 3)
        format = vl::IF_RGB;
      else if (numOfComponents == 4)
        format = vl::IF_RGBA;
    }

    unsigned int size = (dims[0] * dims[1] * dims[2]) * sizeof(pixType.GetSize());

    vlImg = new vl::Image(dims[0], dims[1], dims[2], bytealign, format, type);
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

  fx = new vl::Effect;
  tr = new vl::Transform();
  
  float opacity;
  mitkImg->GetPropertyList()->GetFloatProperty("opacity", opacity);

  mitk::BaseProperty  * prop = mitkImg->GetProperty("color");
  mitk::Color mitkColor = dynamic_cast<mitk::ColorProperty* >(prop)->GetColor();

  vl::fvec4 color;
  color[0] = mitkColor[0];
  color[1] = mitkColor[1];
  color[2] = mitkColor[2];
  color[3] = opacity;

  fx->shader()->enable(vl::EN_DEPTH_TEST);
  fx->shader()->enable(vl::EN_BLEND);
  fx->shader()->setRenderState(m_Light.get(), 0 );
  fx->shader()->enable(vl::EN_LIGHTING);
  fx->shader()->gocMaterial()->setDiffuse(color);
  fx->shader()->gocMaterial()->setTransparency(1.0f- opacity);

  vl::String vlFsSource;
  LoadGLSLSourceFromResources("volume_raycast_isosurface_transp.fs", vlFsSource);

  vl::String vlVsSource;
  LoadGLSLSourceFromResources("volume_luminance_light.vs", vlVsSource);

  // The GLSL program used to perform the actual rendering.
  // The \a volume_luminance_light.fs fragment shader allows you to specify how many 
  // lights to use (up to 4) and can optionally take advantage of a precomputed normals texture.
  vl::ref<vl::GLSLProgram> glslShader;
  glslShader = fx->shader()->gocGLSLProgram();
  glslShader->attachShader( new vl::GLSLFragmentShader( vlFsSource));
  glslShader->attachShader( new vl::GLSLVertexShader(vlVsSource) );

  vl::ref<vl::Actor> imageActor = new vl::Actor;
  imageActor->setEffect(fx.get() );
  imageActor->setTransform(tr.get());
  sceneManager()->tree()->addActor( imageActor.get() );
  imageActor->setUniform( m_ThresholdVal.get() );

  vl::ref<vl::RaycastVolume> mRaycastVolume;
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
  vl::AABB volume_box( vl::vec3( -dimX+shiftX,-dimY+shiftY,-dimZ+shiftZ), vl::vec3(dimX+shiftX,dimY+shiftY,dimZ+shiftZ) );
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

  fx->shader()->gocTextureSampler( 0 )->setTexture( new vl::Texture( vlImg.get(), vl::TF_LUMINANCE8, false, false ) );

  fx->shader()->gocUniform( "volume_texunit" )->setUniformI( 0 );
  mRaycastVolume->generateTextureCoordinates( vl::ivec3(vlImg->width(), vlImg->height(), vlImg->depth()) );

  // generate a simple colored transfer function
  vl::ref<vl::Image> trfunc;
  trfunc = vl::makeColorSpectrum( 1024, vl::blue, vl::royalblue, vl::green, vl::yellow, vl::crimson );

  // installs the transfer function as texture #1
  fx->shader()->gocTextureSampler( 1 )->setTexture( new vl::Texture( trfunc.get() ) );
  fx->shader()->gocUniform( "trfunc_texunit" )->setUniformI( 1 );
/*
  fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( 1);
  fx->shader()->gocTextureSampler( 2 )->setTexture( new Texture( gradient.get(), TF_RGBA, false, false ) );
  fx->shader()->gocUniform( "gradient_texunit" )->setUniformI( 2 );
*/
  fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( 0 );
  // used to compute on the fly the normals based on the volume's gradient
  fx->shader()->gocUniform( "gradient_delta" )->setUniform( vl::fvec3( 0.5f/vlImg->width(), 0.5f/vlImg->height(), 0.5f/vlImg->depth() ) );
  
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

  m_ActorToRenderableMap[imageActor] = dynamic_cast<vl::Renderable*>( vlImg.get());
  return imageActor;
}

void VLRenderingApplet::UpdateThresholdVal( int val )
{
  float val_threshold = 0.0f;
  m_ThresholdVal->getUniform( &val_threshold );
  val_threshold = val / 10000.0f;
  val_threshold = vl::clamp( val_threshold, 0.0f, 1.0f );
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




