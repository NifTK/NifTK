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

// VTK
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>

// MITK
#include <mitkImageReadAccessor.h>

using namespace vl;

VLRenderingApplet::VLRenderingApplet()
{
  m_FPSTimer.start();
  m_ThresholdVal = new Uniform( "val_threshold" );
  m_ThresholdVal->setUniformF( 0.5f );
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
  m_Light->bindTransform( NULL );

  m_Light->setAmbient( fvec4( 0.1f, 0.1f, 0.1f, 1.0f ) );
  m_Light->setDiffuse( vl::gold );
  m_Light->bindTransform( m_LightTr.get() );
  
  populateScene();
}

void VLRenderingApplet::populateScene()
{
  // erase the scene

  sceneManager()->tree()->actors()->clear();

  // regenerate the scene

  ref<Effect> text_fx = new Effect;
  text_fx->shader()->enable(EN_BLEND);
  sceneManager()->tree()->addActor(mText.get(), text_fx.get());

  ref<Geometry> box    = makeBox(vec3(0,0,-2), 1,1,1);
  ref<Geometry> sphere = makeUVSphere(vec3(0,0,0),0.5f);
  ref<Geometry> cone   = makeCone(vec3(0,0,+2), 1, 1, 10, true);
  box   ->computeNormals();
  sphere->computeNormals();
  cone  ->computeNormals();

  ref<Light> light = new Light;

    // rendering order: 
    // red -> yellow
    // box -> sphere -> cone
    mText->setText("red -> yellow\nbox -> sphere -> cone");

    ref<Effect> red_fx = new Effect;
    red_fx->setRenderRank(1);
    red_fx->shader()->disable(EN_DEPTH_TEST);
    red_fx->shader()->enable(EN_CULL_FACE);
    red_fx->shader()->enable(EN_LIGHTING);
    red_fx->shader()->setRenderState( light.get(), 0 );
    red_fx->shader()->gocMaterial()->setDiffuse(red);

    sceneManager()->tree()->addActor( box.get(),    red_fx.get(), mTransform1.get() )->setRenderRank( 1 );
    sceneManager()->tree()->addActor( sphere.get(), red_fx.get(), mTransform1.get() )->setRenderRank( 2 );
    sceneManager()->tree()->addActor( cone.get(),   red_fx.get(), mTransform1.get() )->setRenderRank( 3 );

    ref<Effect> yellow_fx = new Effect;
    yellow_fx->setRenderRank(2);
    yellow_fx->shader()->disable(EN_DEPTH_TEST);
    yellow_fx->shader()->enable(EN_CULL_FACE);
    yellow_fx->shader()->enable(EN_LIGHTING);
    yellow_fx->shader()->setRenderState( light.get(), 0 );
    yellow_fx->shader()->gocMaterial()->setDiffuse(yellow);

    sceneManager()->tree()->addActor( box.get(),  yellow_fx.get(), mTransform2.get() )->setRenderRank( 1 );
    sceneManager()->tree()->addActor( cone.get(), yellow_fx.get(), mTransform2.get() )->setRenderRank( 2 );
/*
  else
  {
    // transp_fx

    ref<Effect> transp_fx = new Effect;
    transp_fx->shader()->enable(EN_BLEND);
    transp_fx->shader()->enable(EN_DEPTH_TEST);
    transp_fx->shader()->enable(EN_CULL_FACE);
    transp_fx->shader()->enable(EN_LIGHTING);
    transp_fx->shader()->setRenderState( light.get(), 0 );
    transp_fx->shader()->gocMaterial()->setDiffuse(blue);
    transp_fx->shader()->gocMaterial()->setTransparency(0.5f);

    // solid_fx

    ref<Effect> solid_fx = new Effect;
    solid_fx->shader()->enable(EN_DEPTH_TEST);
    solid_fx->shader()->enable(EN_CULL_FACE);
    solid_fx->shader()->enable(EN_LIGHTING);
    solid_fx->shader()->setRenderState( light.get(), 0 );
    solid_fx->shader()->gocMaterial()->setDiffuse(yellow);

    // add to the scene in an intertwined way
    sceneManager()->tree()->addActor( box.get(),    transp_fx.get(), mTransform1.get() );
    sceneManager()->tree()->addActor( box.get(),    solid_fx.get(), mTransform2.get() );
    sceneManager()->tree()->addActor( sphere.get(), transp_fx.get(), mTransform1.get() );
    sceneManager()->tree()->addActor( cone.get(),   solid_fx.get(), mTransform2.get() );
    sceneManager()->tree()->addActor( cone.get(),   transp_fx.get(), mTransform1.get() );

    if (mTestNumber == 1) // depth-sort only alpha blended objects (default settings)
    {
      mText->setText("depth-sort only alpha blended objects (default settings)");
      ref<RenderQueueSorterStandard> list_sorter = new RenderQueueSorterStandard;
      list_sorter->setDepthSortMode(AlphaDepthSort);
      rendering()->as<Rendering>()->setRenderQueueSorter( list_sorter.get() );
    }
    else
      if (mTestNumber == 2) // depth-sort solid and alpha blended objects
      {
        solid_fx->shader()->disable(EN_DEPTH_TEST);
        mText->setText("depth-sort solid and alpha blended objects");
        ref<RenderQueueSorterStandard> list_sorter = new RenderQueueSorterStandard;
        list_sorter->setDepthSortMode(AlwaysDepthSort);
        rendering()->as<Rendering>()->setRenderQueueSorter( list_sorter.get() );
      }
      else
        if (mTestNumber == 3) // depth-sort alpha blended back to front | depth-sort solid object front to back
        {
          mText->setText("depth-sort alpha blended back to front\ndepth-sort solid object front to back");
          ref<RenderQueueSorterOcclusion> list_sorter = new RenderQueueSorterOcclusion;
          rendering()->as<Rendering>()->setRenderQueueSorter( list_sorter.get() );
        }
        else
          if (mTestNumber == 4) // no depth sorting
          {
            mText->setText("no depth sorting");
            ref<RenderQueueSorterStandard> list_sorter = new RenderQueueSorterStandard;
            list_sorter->setDepthSortMode(NeverDepthSort);
            rendering()->as<Rendering>()->setRenderQueueSorter( list_sorter.get() );
          }
          else
            if (mTestNumber == 5) // no sorting at all
            {
              mText->setText("no sorting at all");
              rendering()->as<Rendering>()->setRenderQueueSorter( NULL );
            }
  }
*/
}

void VLRenderingApplet::updateScene()
{
  mTransform1->setLocalMatrix( mat4::getRotation(Time::currentTime() * 5,      0, 1, 0 ) );
  mTransform2->setLocalMatrix( mat4::getRotation(Time::currentTime() * 5 + 90, 0, 1, 0 ) );

  mat4 mat;
  // light 0 transform.
  mat = mat4::getRotation( Time::currentTime()*43, 0,1,0 ) * mat4::getTranslation( 20,20,20 );
  m_LightTr->setLocalMatrix( mat );
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

  mitk::Image::Pointer mitkImg = dynamic_cast<mitk::Image *>(node->GetData());

  if (mitkImg.IsNotNull())
    AddImageActor(mitkImg);
  
  mitk::Surface::Pointer mitkSurf = dynamic_cast<mitk::Surface *>(node->GetData());
  if (mitkSurf.IsNotNull())
    AddSurfaceActor(mitkSurf);
}

void VLRenderingApplet::AddSurfaceActor(mitk::Surface::Pointer mitkSurf)
{
  ref<Geometry> vlSurf;
  ref<Effect> fx; 
  ref<Transform> tr;

/*
  try
  {
    mitk::ImageReadAccessor readAccess(mitkImg, mitkImg->GetVolumeData(0));
    const void* cPointer = readAccess.GetData();
    unsigned int * dims = new unsigned int[3];
    dims = mitkImg->GetDimensions();
    int bytealign = 1;
    EImageFormat format = IF_LUMINANCE;
    EImageType type = IT_UNSIGNED_SHORT;

    unsigned int size = (dims[0] * dims[1] * dims[2]) * sizeof(unsigned char);

    vlImg = new Image(dims[0], dims[1], dims[2], bytealign, format, type);
    memcpy(vlImg->pixels(), cPointer, vlImg->requiredMemory());

  }
  catch(mitk::Exception& e)
  {
    // deal with the situation not to have access
  }
*/
  fx = new Effect;

  float opacity;
  mitkSurf->GetPropertyList()->GetFloatProperty("opacity", opacity);

  mitk::BaseProperty  * prop = mitkSurf->GetProperty("color");
  mitk::Color mitkColor = dynamic_cast<mitk::ColorProperty* >(prop)->GetColor();

  vl::fvec4 color;
  color[0] = mitkColor[0];
  color[1] = mitkColor[1];
  color[2] = mitkColor[2];

  //  SURFACE ONLY

  if (opacity == 1.0f)
  {
    fx->shader()->enable(EN_DEPTH_TEST);
    fx->shader()->enable(EN_CULL_FACE);
    fx->shader()->enable(EN_LIGHTING);
    //fx->shader()->setRenderState( light.get(), 0 );
    fx->shader()->gocMaterial()->setDiffuse(color);
  }
  else
  {
    fx->shader()->enable(EN_BLEND);
    fx->shader()->enable(EN_DEPTH_TEST);
    fx->shader()->enable(EN_CULL_FACE);
    fx->shader()->enable(EN_LIGHTING);
    //fx->shader()->setRenderState( light.get(), 0 );
    fx->shader()->gocMaterial()->setDiffuse(color);
    fx->shader()->gocMaterial()->setTransparency(1.0f- opacity);
  }

  tr = new Transform();

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

  ref<RenderQueueSorterStandard> list_sorter = new RenderQueueSorterStandard;
  list_sorter->setDepthSortMode(AlwaysDepthSort);
  rendering()->as<Rendering>()->setRenderQueueSorter( list_sorter.get() );

  sceneManager()->tree()->addActor(vlSurf.get(), fx.get(), tr.get());

  // refresh window
  openglContext()->update();
}

void VLRenderingApplet::AddImageActor(mitk::Image::Pointer mitkImg)
{
  ref<Image>     vlImg;
  ref<Effect>    fx; 
  ref<Transform> tr;

  try
  {
    mitk::ImageReadAccessor readAccess(mitkImg, mitkImg->GetVolumeData(0));
    const void* cPointer = readAccess.GetData();
    unsigned int * dims = new unsigned int[3];
    dims = mitkImg->GetDimensions();
    int bytealign = 1;
    EImageFormat format = IF_LUMINANCE;
    EImageType type = IT_UNSIGNED_SHORT;

    unsigned int size = (dims[0] * dims[1] * dims[2]) * sizeof(unsigned char);

    vlImg = new Image(dims[0], dims[1], dims[2], bytealign, format, type);
    memcpy(vlImg->pixels(), cPointer, vlImg->requiredMemory());

  }
  catch(mitk::Exception& e)
  {
    // deal with the situation not to have access
  }

  //vlImg = loadImage( "e:/Niftike-r/VL-src/data/volume/VLTest.dat" );

  fx = new Effect;

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
  fx->shader()->setRenderState( m_Light.get(), 0 );
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
  AABB volume_box( vec3( -10,-10,-10 ), vec3( +10,+10,+10 ) );
  mRaycastVolume->setBox( volume_box );

  // Setup image
  
/*
  ref<Image> gradient;
  // note that this can take a while...
  gradient = vl::genGradientNormals( vlImg.get() );
*/
  fx = imageActor->effect();

  // install volume image as textue #0
  fx->shader()->gocTextureSampler( 0 )->setTexture( new vl::Texture( vlImg.get(), TF_LUMINANCE8, false, false ) );
  fx->shader()->gocUniform( "volume_texunit" )->setUniformI( 0 );
  mRaycastVolume->generateTextureCoordinates( ivec3(vlImg->width(), vlImg->height(), vlImg->depth()) );

  // generate a simple colored transfer function
  ref<Image> trfunc;
  trfunc = vl::makeColorSpectrum( 128, vl::blue, vl::royalblue, vl::green, vl::yellow, vl::crimson );

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
  tr = new Transform();

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

