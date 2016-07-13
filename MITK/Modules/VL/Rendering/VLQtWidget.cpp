/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if 1
  // MS VS
  #if defined(_MSC_VER)
    #define VIVID_TRAP() { if (IsDebuggerPresent()) { __debugbreak(); } else ::vl::abort_vl(); }
  // GNU GCC
  #elif defined(__GNUG__) || defined(__MINGW32__)
    #define VIVID_TRAP() { fflush(stdout); fflush(stderr); asm("int $0x3"); }
  #else
    #define VIVID_TRAP() { ::vl::abort_vl(); }
  #endif
  #define VIVID_CHECK(expr) { if(!(expr)) { ::vl::log_failed_check(#expr,__FILE__,__LINE__); VIVID_TRAP() } }
  #define VIVID_WARN(expr)  { if(!(expr)) { ::vl::log_failed_check(#expr,__FILE__,__LINE__); } }
#else
  #define VIVID_CHECK(expr) {}
  #define VIVID_WARN(expr) {}
  #define VIVID_TRAP() {}
#endif

#include <QTextStream>
#include <QFile>
#include <QDir>

#include "VLQtWidget.h"
#include <vlQt5/QtDirectory.hpp>
#include <vlQt5/QtFile.hpp>
#include <vlCore/Log.hpp>
#include <vlCore/Time.hpp>
#include <vlCore/Colors.hpp>
#include <vlCore/GlobalSettings.hpp>
#include <vlCore/FileSystem.hpp>
#include <vlCore/ResourceDatabase.hpp>
#include <vlGraphics/GeometryPrimitives.hpp>
#include <vlGraphics/RenderQueueSorter.hpp>
#include <vlGraphics/GLSL.hpp>
#include <vlGraphics/plugins/ioVLX.hpp>
#include <vlGraphics/FramebufferObject.hpp>
#include <vlGraphics/AdjacencyExtractor.hpp>
#include <vlVolume/RaycastVolume.hpp>
#include <cassert>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataNormals.h>
#include <vtkImageData.h>
#include <mitkEnumerationProperty.h>
#include <mitkProperties.h>
#include <mitkImageReadAccessor.h>
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <stdexcept>
#include <sstream>
#include <niftkScopedOGLContext.h>
// #include "TrackballManipulator.h"
#ifdef BUILD_IGI
#include <CameraCalibration/niftkUndistortion.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkCameraIntrinsics.h>
#endif
#include <mitkCoordinateAxesData.h>

#ifdef _USE_PCL
#include <niftkPCLData.h>
#endif

#ifdef _MSC_VER
#ifdef _USE_NVAPI
#include <nvapi.h>
#endif
#endif

//-----------------------------------------------------------------------------
// CUDA stuff
//-----------------------------------------------------------------------------

#ifdef _USE_CUDA

  #include <Rendering/VLFramebufferToCUDA.h>
  #include <niftkCUDAManager.h>
  #include <niftkCUDAImage.h>
  #include <niftkLightweightCUDAImage.h>
  #include <niftkCUDAImageProperty.h>
  #include <niftkFlipImageLauncher.h>
  #include <cuda_gl_interop.h>

  //-----------------------------------------------------------------------------

  struct CUDAInterop
  {
    std::string m_NodeName;
    mitk::DataStorage::Pointer m_DataStorage;

    VLFramebufferAdaptor* m_FBOAdaptor;

    CUDAInterop() : m_FBOAdaptor(0)
    {
    }

    ~CUDAInterop()
    {
      delete m_FBOAdaptor;
    }
  };

  //-----------------------------------------------------------------------------

  struct TextureDataPOD
  {
    vl::ref<vl::Texture>   m_Texture;       // on the vl side
    unsigned int           m_LastUpdatedID; // on cuda-manager side
    cudaGraphicsResource_t m_CUDARes;       // on cuda(-driver) side

    TextureDataPOD(): m_LastUpdatedID(0) , m_CUDARes(0) {
    }
  };

// #else
//   struct CUDAInterop { };
#endif

using namespace vl;

//-----------------------------------------------------------------------------
// VLUserData
//-----------------------------------------------------------------------------

struct VLUserData: public vl::Object
{
  VLUserData()
    : m_TransformModifiedTime(0)
    , m_ImageModifiedTime(0)
  {
  }

  itk::ModifiedTimeType m_TransformModifiedTime;
  itk::ModifiedTimeType m_ImageModifiedTime;
};

//-----------------------------------------------------------------------------
// Util functions
//-----------------------------------------------------------------------------

namespace
{
  vl::EImageType MapITKPixelTypeToVL(int itkComponentType)
  {
    static const vl::EImageType typeMap[] =
    {
      vl::IT_IMPLICIT_TYPE,   // itk::ImageIOBase::UNKNOWNCOMPONENTTYPE = 0
      vl::IT_UNSIGNED_BYTE,   // itk::ImageIOBase::UCHAR = 1
      vl::IT_BYTE,            // itk::ImageIOBase::CHAR = 2
      vl::IT_UNSIGNED_SHORT,  // itk::ImageIOBase::USHORT = 3
      vl::IT_SHORT,           // itk::ImageIOBase::SHORT = 4
      vl::IT_UNSIGNED_INT,    // itk::ImageIOBase::UINT = 5
      vl::IT_INT,             // itk::ImageIOBase::INT = 6
      vl::IT_IMPLICIT_TYPE,   // itk::ImageIOBase::ULONG = 7
      vl::IT_IMPLICIT_TYPE,   // itk::ImageIOBase::LONG = 8
      vl::IT_FLOAT,           // itk::ImageIOBase::FLOAT = 9
      vl::IT_IMPLICIT_TYPE    // itk::ImageIOBase::DOUBLE = 10
    };

    return typeMap[itkComponentType];
  }

  //-----------------------------------------------------------------------------

  vl::EImageFormat MapComponentsToVLColourFormat(int components)
  {
    // this assumes the image data is a normal colour image, not encoding pointers or indices, or similar stuff.

    switch (components)
    {
      default:
      case 1:
        return vl::IF_LUMINANCE;
      case 2:
        return vl::IF_RG;
      case 3:
        return vl::IF_RGB;
      case 4:
        return vl::IF_RGBA;
    }
  }

  //-----------------------------------------------------------------------------

  VLUserData* GetUserData(vl::Actor* actor)
  {
    VIVID_CHECK( actor );
    ref<VLUserData> userdata = actor->userData()->as<VLUserData>();
    if ( ! userdata )
    {
      userdata = new VLUserData;
      actor->setUserData( userdata.get() );
    }

    return userdata.get();
  }

  //-----------------------------------------------------------------------------

  vl::mat4 GetVLMatrixFromData(const mitk::BaseData::ConstPointer& data)
  {
    vl::mat4  mat;
    // Intentionally not setIdentity()
    mat.setNull();

    if ( data )
    {
      mitk::BaseGeometry::Pointer geom = data->GetGeometry();
      if ( geom ) {
        if ( geom->GetVtkTransform() ) {
          vtkSmartPointer<vtkMatrix4x4> vtkmat = vtkSmartPointer<vtkMatrix4x4>::New();
          geom->GetVtkTransform()->GetMatrix(vtkmat);
          if ( vtkmat.GetPointer() ) {
            for (int i = 0; i < 4; i++) {
              for (int j = 0; j < 4; j++) {
                double val = vtkmat->GetElement(i, j);
                mat.e(i, j) = val;
              }
            }
          }
        }
      }
    }

    return mat;
  }

  //-----------------------------------------------------------------------------

  void UpdateTransformFromData(vl::Transform* txf, const mitk::BaseData::ConstPointer& data)
  {
    vl::mat4  mat = GetVLMatrixFromData(data);

    if (!mat.isNull())
    {
      txf->setLocalMatrix(mat);
      txf->computeWorldMatrix();
    }
  }

  //-----------------------------------------------------------------------------

  void UpdateActorTransformFromNode(vl::Actor* actor, const mitk::DataNode::ConstPointer& node)
  {
    if (node.IsNotNull())
    {
      mitk::BaseData::Pointer data = node->GetData();
      if (data.IsNotNull())
      {
        mitk::BaseGeometry::Pointer geom = data->GetGeometry();
        if (geom.IsNotNull())
        {
          ref<VLUserData> userdata = GetUserData(actor);
          if (geom->GetMTime() > userdata->m_TransformModifiedTime)
          {
            UpdateTransformFromData(actor->transform(), data.GetPointer());
            userdata->m_TransformModifiedTime = geom->GetMTime();
          }
        }
      }
    }
  }

  //-----------------------------------------------------------------------------

  void UpdateTransformFromNode(vl::Transform* txf, const mitk::DataNode::ConstPointer& node)
  {
    if (node.IsNotNull())
    {
      UpdateTransformFromData(txf, node->GetData());
    }
  }

  //-----------------------------------------------------------------------------

  ref<vl::Geometry> CreateGeometryFor2DImage(int width, int height)
  {
    ref<vl::Geometry>    geom = new vl::Geometry;
    ref<vl::ArrayFloat3> vert  = new vl::ArrayFloat3;
    vert->resize(4);
    geom->setVertexArray( vert.get() );

    ref<vl::ArrayFloat2> tex_coord = new vl::ArrayFloat2;
    tex_coord->resize(4);
    geom->setTexCoordArray(0, tex_coord.get());

    //  1---2 image-top
    //  |   |
    //  0---3 image-bottom

    vert->at(0).x() = 0;     vert->at(0).y() = 0;      vert->at(0).z() = 0; tex_coord->at(0).s() = 0; tex_coord->at(0).t() = 1;
    vert->at(1).x() = 0;     vert->at(1).y() = height; vert->at(1).z() = 0; tex_coord->at(1).s() = 0; tex_coord->at(1).t() = 0;
    vert->at(2).x() = width; vert->at(2).y() = height; vert->at(2).z() = 0; tex_coord->at(2).s() = 1; tex_coord->at(2).t() = 0;
    vert->at(3).x() = width; vert->at(3).y() = 0;      vert->at(3).z() = 0; tex_coord->at(3).s() = 1; tex_coord->at(3).t() = 1;

    ref<vl::DrawArrays> polys = new vl::DrawArrays(vl::PT_QUADS, 0, 4);
    geom->drawCalls().push_back( polys.get() );

    return geom;
  }

  //-----------------------------------------------------------------------------

  ref<vl::Geometry> ConvertVTKPolyData(vtkPolyData* vtkPoly)
  {
    if ( ! vtkPoly ) {
      return NULL;
    }

    ref<vl::Geometry> vlPoly = new vl::Geometry;

    // Buffer in host memory to store cell info
    unsigned int* m_IndexBuffer = 0;

    // Buffer in host memory to store vertex points
    float* m_PointBuffer = 0;

    // Buffer in host memory to store normals associated with vertices
    float* m_NormalBuffer = 0;

    // Buffer in host memory to store scalar info associated with vertices
    char* m_ScalarBuffer = 0;

    unsigned int numOfvtkPolyPoints = vtkPoly->GetNumberOfPoints();

    // A polydata will always have point data
    int pointArrayNum = vtkPoly->GetPointData()->GetNumberOfArrays();

    if (pointArrayNum == 0 && numOfvtkPolyPoints == 0)
    {
      MITK_ERROR << "No points detected in the vtkPoly data!\n";
      return NULL;
    }

    // We'll have to build the cell data if not present already
    int cellArrayNum  = vtkPoly->GetCellData()->GetNumberOfArrays();
    if ( cellArrayNum == 0 ) {
      vtkPoly->BuildCells();
    }

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

    if (verts->GetMaxCellSize() > 3)
    {
      // Panic and return
      MITK_ERROR << "More than three vertices / cell detected, can't handle this data type!\n";
      return NULL;
    }

    vtkSmartPointer<vtkPoints> points = vtkPoly->GetPoints();

    if (points == 0)
    {
      MITK_ERROR << "Corrupt vtkPoly, returning! \n";
      return NULL;
    }

    // Deal with normals
    vtkSmartPointer<vtkDataArray> normals = vtkPoly->GetPointData()->GetNormals();

    if (normals == 0)
    {
      MITK_INFO << "Generating normals for the vtkPoly data (mitk::OclSurface)";

      vtkSmartPointer<vtkPolyDataNormals> normalGen = vtkSmartPointer<vtkPolyDataNormals>::New();
      normalGen->SetInputData(vtkPoly);
      normalGen->AutoOrientNormalsOn();
      normalGen->Update();

      normals = normalGen->GetOutput()->GetPointData()->GetNormals();

      if (normals == 0)
      {
        MITK_ERROR << "Couldn't generate normals, returning! \n";
        return NULL;
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
    pointBufferSize = numOfPoints * sizeof(float) * 3;

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
      VIVID_CHECK(m_NormalCount == numOfPoints);

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
    MITK_INFO << "Surface data initialized. Num of Points: " <<points->GetNumberOfPoints() << " Num of Cells: " <<verts->GetNumberOfCells() << "\n";

    ref<vl::ArrayFloat3>  vl_verts   = new vl::ArrayFloat3;
    ref<vl::ArrayFloat3>  vlNormals = new vl::ArrayFloat3;
    ref<vl::DrawElementsUInt> vlTriangles = new vl::DrawElementsUInt(vl::PT_TRIANGLES);

    vl_verts->resize(numOfPoints * 3);
    vlNormals->resize(numOfPoints * 3);

    vlPoly->drawCalls().push_back(vlTriangles.get());
    vlTriangles->indexBuffer()->resize(numOfTriangles*3);

    vlPoly->setVertexArray(vl_verts.get());
    vlPoly->setNormalArray(vlNormals.get());

    float* vertBufFlotPtr = reinterpret_cast<float *>(vl_verts->ptr());
    float* normBufFlotPtr = reinterpret_cast<float *>(vlNormals->ptr());

    // Vertices and normals
    for (unsigned int i=0; i<numOfPoints; ++i)
    {
      vertBufFlotPtr[3*i + 0] = m_PointBuffer[i*3 +0];
      vertBufFlotPtr[3*i + 1] = m_PointBuffer[i*3 +1];
      vertBufFlotPtr[3*i + 2] = m_PointBuffer[i*3 +2];

      normBufFlotPtr[3*i + 0] = m_NormalBuffer[i*3 +0];
      normBufFlotPtr[3*i + 1] = m_NormalBuffer[i*3 +1];
      normBufFlotPtr[3*i + 2] = m_NormalBuffer[i*3 +2];
    }

    // Make sure that the values are copied onto GPU memory
    //vlPoly->vertexArray()->updateBufferObject();
    //glFinish();

    // Read triangles
    for(unsigned int i=0; i<numOfTriangles; ++i)
    {
      vlTriangles->indexBuffer()->at(i*3+0) = m_IndexBuffer[i*3 +0];
      vlTriangles->indexBuffer()->at(i*3+1) = m_IndexBuffer[i*3 +1];
      vlTriangles->indexBuffer()->at(i*3+2) = m_IndexBuffer[i*3 +2];
    }

    // Make sure that the values are copied onto GPU memory
    vl_verts->updateBufferObject();
    vlNormals->updateBufferObject();
    vlTriangles->indexBuffer()->updateBufferObject();
    glFinish();

    // Buffer in host memory to store cell info
    if (m_IndexBuffer != 0)
      delete m_IndexBuffer;

    // Buffer in host memory to store vertex points
    if (m_PointBuffer != 0)
      delete m_PointBuffer;

    // Buffer in host memory to store normals associated with vertices
    if (m_NormalBuffer != 0)
      delete m_NormalBuffer;

    // Buffer in host memory to store scalar info associated with vertices
    if (m_ScalarBuffer != 0)
      delete m_ScalarBuffer;

    // MITK_INFO << "Num of VL vertices: " << vlPoly->vertexArray()->size() / 3;

    // Finally convert to adjacency format so we can render silhouettes etc.
    return vl::AdjacencyExtractor::extract( vlPoly.get() );
  }

  //-----------------------------------------------------------------------------

  void dumpNodeInfo( const std::string& prefix, const mitk::DataNode::ConstPointer& node ) {
    printf( "\n%s: ", prefix.c_str() );
    const char* class_name = node->GetData() ? node->GetData()->GetNameOfClass() : "<unknown-class>";
    mitk::StringProperty* name_prop = dynamic_cast<mitk::StringProperty*>(node->GetProperty("name"));
    const char* object_name2 = "<unknown-name>";
    if (name_prop != 0) {
      object_name2 = name_prop->GetValue();
    }
    printf( "%s <%s>\n", object_name2, class_name );

    const mitk::PropertyList::PropertyMap* propList = node->GetPropertyList()->GetMap();
    mitk::PropertyList::PropertyMap::const_iterator it = node->GetPropertyList()->GetMap()->begin();
    for( ; it != node->GetPropertyList()->GetMap()->end(); ++it ) {
      const std::string name = it->first;
      const mitk::BaseProperty::Pointer prop = it->second;
      printf( "\t%s: %s <%s>\n", name.c_str(), prop->GetValueAsString().c_str(), prop->GetNameOfClass() );
      if ( name == "size" ) {
        VIVID_CHECK( node->GetProperty( "size") );
      }
    }
  }
}

//-----------------------------------------------------------------------------
// Init and shutdown VL
//-----------------------------------------------------------------------------

namespace
{
  class VLInit
  {
  public:
    VLInit() { vl::VisualizationLibrary::init(); }
    ~VLInit() { vl::VisualizationLibrary::shutdown(); }
  };

  VLInit s_ModuleInit;
}

//-----------------------------------------------------------------------------
// VLMapper
//-----------------------------------------------------------------------------

void VLMapper::initVLPropertiesGlobal() {
}

void VLMapper::initVLPropertiesPointSet() {
  // Init only once
  if ( m_DataNode->GetProperty( "VL.PointMode" ) ) {
    return;
  }

  mitk::EnumerationProperty::Pointer point_set_mode = mitk::EnumerationProperty::New();
  point_set_mode->AddEnum("3D", 0);
  point_set_mode->AddEnum("2D", 1);
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.PointMode", point_set_mode);
  point_set_mode->SetValue( 0 );
  
  mitk::FloatProperty::Pointer point_size_2d = mitk::FloatProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.PointSize2D", point_size_2d);
  point_size_2d->SetValue( 5 );
  
  mitk::FloatProperty::Pointer point_size_3d = mitk::FloatProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.PointSize3D", point_size_3d);
  point_size_3d->SetValue( 5 );
  
  mitk::FloatProperty::Pointer point_opacity = mitk::FloatProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.PointOpacity", point_opacity);
  point_opacity->SetValue( 1 );
  
  mitk::ColorProperty::Pointer point_color = mitk::ColorProperty::New();
  const_cast<mitk::DataNode*>(m_DataNode)->SetProperty("VL.PointColor", point_color);
  point_color->SetValue( mitk::Color( 1 ) );
  
  // MIC FIXME: need to somehow update the DataStorage views
  m_DataStorage->Modified();
}

//-----------------------------------------------------------------------------

vl::ref<vl::Actor> VLMapper::initActor(vl::Geometry* geom, vl::Effect* effect = NULL) {
  VIVID_CHECK( m_DataNode );
  VIVID_CHECK( m_VividRendering );
  ref<vl::Effect> fx = effect ? effect : vl::VividRendering::makeVividEffect();
  ref<vl::Transform> tr = new vl::Transform;
  UpdateTransformFromData( tr.get(), m_DataNode->GetData() );
  ref<vl::Actor> actor = new vl::Actor( geom, fx.get(), tr.get() );
  actor->setEnableMask( vl::VividRenderer::DefaultEnableMask );
  return actor;
}

//-----------------------------------------------------------------------------

void VLMapper::updateCommon() {
  if ( ! m_Actor ) {
    return;
  }

  // Update visibility
  bool visible = true;
  m_DataNode->GetBoolProperty( "visible", visible );
  m_Actor->setEnabled( visible );
  
  // Update opacity
  float opacity = 1.0f;
  m_DataNode->GetFloatProperty( "opacity", opacity );

  // Update color
  float rgb[3] = { 1, 1, 1 };
  m_DataNode->GetColor( rgb );

  // MIC FIXME:
  // This won't work when vl_Vivid.enableLighting is off -> create special uniform
  m_Actor->effect()->shader()->getMaterial()->setDiffuse( vl::vec4( rgb[0], rgb[1], rgb[2], opacity ) );

  // Update transform
  UpdateTransformFromData( m_Actor->transform(), m_DataNode->GetData() );
}

//-----------------------------------------------------------------------------

class VLMapperSurface: public VLMapper {
public:
  VLMapperSurface( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    m_MitkSurf = dynamic_cast<mitk::Surface*>( node->GetData() );
    VIVID_CHECK( m_MitkSurf );
  }

  virtual void init() {
    VIVID_CHECK( m_MitkSurf );

    ref<vl::Geometry> geom = ConvertVTKPolyData( m_MitkSurf->GetVtkPolyData() );
    if ( ! geom->normalArray() ) {
      geom->computeNormals();
    }

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  }

  virtual void update() {}

protected:
  mitk::Surface::Pointer m_MitkSurf;
};

//-----------------------------------------------------------------------------

class VLMapper2DImage: public VLMapper {
public:
  VLMapper2DImage( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
    VIVID_CHECK( m_MitkImage.IsNotNull() );
  }

  virtual void init() {
    VIVID_CHECK( m_MitkImage.IsNotNull() );

    mitk::PixelType  mitk_pixel_type = m_MitkImage->GetPixelType();
    vl::EImageType   vl_type         = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
    vl::EImageFormat vl_format       = MapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());
    unsigned int*    dims            = m_MitkImage->GetDimensions();

    ref<vl::Image> vl_img;

    try {
      unsigned int buffer_bytes = dims[0] * dims[1] * dims[2] * mitk_pixel_type.GetSize();
      mitk::ImageReadAccessor readAccess( m_MitkImage, m_MitkImage->GetVolumeData(0) );
      void* buffer_ptr = const_cast<void*>( readAccess.GetData() );
      // std::memcpy( vl_img->pixels(), ptr, byte_count );
      // Use VTK buffer directly instead of allocating one
      vl_img = new vl::Image( buffer_ptr, buffer_bytes );
      vl_img->allocate2D(dims[0], dims[1], 1, vl_format, vl_type);
      VIVID_CHECK( vl_img->requiredMemory() == buffer_bytes );
    }
    catch (...) {
      // FIXME: error handling?
      MITK_ERROR << "Did not get pixel read access to 2D image.";
    }

    ref<vl::Geometry> geom = CreateGeometryFor2DImage(dims[0], dims[1]);

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    // These must be present as part of the default Vivid material
    VIVID_CHECK( fx->shader()->getTextureSampler( vl::VividRendering::UserTexture ) )
    VIVID_CHECK( fx->shader()->getTextureSampler( vl::VividRendering::UserTexture )->texture() )
    VIVID_CHECK( fx->shader()->getUniform("vl_UserTexture")->getUniformI() == vl::VividRendering::UserTexture );
    ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::VividRendering::UserTexture )->texture();
    texture->createTexture2D( vl_img.get(), vl::TF_UNKNOWN, false, false );
    fx->shader()->getUniform("vl_Vivid.enableTextureMapping")->setUniformI( 1 );
    fx->shader()->getUniform("vl_Vivid.enableLighting")->setUniformI( 0 );
    // When texture mapping is enabled the texture is modulated by the vertex color, including the alpha
    geom->setColorArray( vl::white );
  }

  virtual void update() {
    VIVID_CHECK( m_MitkImage.IsNotNull() );

    if ( m_MitkImage->GetVtkImageData()->GetMTime() <= GetUserData( m_Actor.get() )->m_ImageModifiedTime ) {
      return;
    }

    ref<vl::Texture> tex = m_Actor->effect()->shader()->gocTextureSampler( vl::VividRendering::UserTexture )->texture();
    if ( tex )
    {
      mitk::PixelType  mitk_pixel_type = m_MitkImage->GetPixelType();
      vl::EImageType   vl_type         = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
      vl::EImageFormat vl_format       = MapComponentsToVLColourFormat(mitk_pixel_type.GetNumberOfComponents());
      unsigned int*    dims            = m_MitkImage->GetDimensions();

      ref<vl::Image> vl_img;

      try
      {
        unsigned int buffer_bytes = dims[0] * dims[1] * dims[2] * mitk_pixel_type.GetSize();
        mitk::ImageReadAccessor readAccess( m_MitkImage, m_MitkImage->GetVolumeData(0) );
        void* buffer_ptr = const_cast<void*>( readAccess.GetData() );
        // std::memcpy( vl_img->pixels(), ptr, byte_count );
        // Use VTK buffer directly instead of allocating one
        vl_img = new vl::Image( buffer_ptr, buffer_bytes );
        vl_img->allocate2D(dims[0], dims[1], 1, vl_format, vl_type);
        VIVID_CHECK( vl_img->requiredMemory() == buffer_bytes );
      }
      catch (...)
      {
        // FIXME: error handling?
        MITK_ERROR << "Did not get pixel read access to 2D image.";
      }

      tex->setMipLevel(0, vl_img.get(), false);

      GetUserData( m_Actor.get() )->m_ImageModifiedTime = m_MitkImage->GetVtkImageData()->GetMTime();
    }
  }

protected:
  mitk::Image::Pointer m_MitkImage;
};

//-----------------------------------------------------------------------------

class VLMapper3DImage: public VLMapper {
public:
  VLMapper3DImage( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    m_MitkImage = dynamic_cast<mitk::Image*>( node->GetData() );
    VIVID_CHECK( m_MitkImage.IsNotNull() );
  }

  virtual void init() {
    // This is all legacy code that will go away when we use the vivid volume class
    // MIC FIXME:
    throw std::runtime_error("VLSceneView::Add3DImageActor(): to be implemented!");

    mitk::PixelType mitk_pixel_type = m_MitkImage->GetPixelType();
    size_t numOfComponents = mitk_pixel_type.GetNumberOfComponents();

    if (false)
    {
      std::cout << "Original pixel type:" << std::endl;
      std::cout << " PixelType: " <<mitk_pixel_type.GetTypeAsString() << std::endl;
      std::cout << " BitsPerElement: " <<mitk_pixel_type.GetBpe() << std::endl;
      std::cout << " NumberOfComponents: " << numOfComponents << std::endl;
      std::cout << " BitsPerComponent: " <<mitk_pixel_type.GetBitsPerComponent() << std::endl;
    }

    ref<vl::Image> vl_img;

    try
    {
      mitk::ImageReadAccessor readAccess(m_MitkImage, m_MitkImage->GetVolumeData(0));
      const void* cPointer = readAccess.GetData();

      vl::EImageType     type = MapITKPixelTypeToVL(mitk_pixel_type.GetComponentType());
      vl::EImageFormat   format;

      if (type != vl::IT_FLOAT)
      {
        if (numOfComponents == 1)
          format = vl::IF_LUMINANCE;
        else if (numOfComponents == 2)
          format = vl::IF_RG_INTEGER;
        else if (numOfComponents == 3)
          format = vl::IF_RGB_INTEGER;
        else if (numOfComponents == 4)
          // FIXME: not sure whether we really want integer formats here!
          //        for now, dont do integer for rgba, we have quite a few rgba images.
          format = vl::IF_RGBA;//_INTEGER;
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

      unsigned int* dims = 0;
      dims = m_MitkImage->GetDimensions();
      // we do not own dims!

      int bytealign = 1;
      if (dims[2] <= 1)
        vl_img = new vl::Image(dims[0], dims[1], 0, bytealign, format, type);
      else
        vl_img = new vl::Image(dims[0], dims[1], dims[2], bytealign, format, type);

      // sanity check
      unsigned int size = (dims[0] * dims[1] * dims[2]) * mitk_pixel_type.GetSize();
      VIVID_CHECK(vl_img->requiredMemory() == size);
      std::memcpy(vl_img->pixels(), cPointer, vl_img->requiredMemory());

      vl_img = vl_img->convertFormat(vl::IF_LUMINANCE)->convertType(vl::IT_UNSIGNED_SHORT);
  /*
      ref<KeyValues> tags = new KeyValues;
      tags->set("Origin")    = Say("%n %n %n") << mitk_img->GetGeometry()->GetOrigin()[0]  << mitk_img->GetGeometry()->GetOrigin()[1]  << mitk_img->GetGeometry()->GetOrigin()[2];
      tags->set("Spacing")   = Say("%n %n %n") << mitk_img->GetGeometry()->GetSpacing()[0] << mitk_img->GetGeometry()->GetSpacing()[1] << mitk_img->GetGeometry()->GetSpacing()[2];
      vl_img->setTags(tags.get());
  */
    }
    catch(mitk::Exception& e)
    {
      // deal with the situation not to have access
      VIVID_CHECK(false);
    }

    float opacity;
    m_MitkImage->GetPropertyList()->GetFloatProperty("opacity", opacity);

    mitk::BaseProperty::Pointer   colourProp = m_MitkImage->GetProperty("color");
    mitk::Color                   mitkColor;
    if (colourProp.IsNotNull())
      mitkColor = dynamic_cast<mitk::ColorProperty*>(colourProp.GetPointer())->GetColor();

    vl::fvec4 color;
    color[0] = mitkColor[0];
    color[1] = mitkColor[1];
    color[2] = mitkColor[2];
    color[3] = opacity;

    ref<vl::Effect> fx = vl::VividRendering::makeVividEffect();
    fx->shader()->enable(vl::EN_DEPTH_TEST);
    fx->shader()->enable(vl::EN_BLEND);
    // fx->shader()->setRenderState(m_Light.get(), 0);
    fx->shader()->enable(vl::EN_LIGHTING);
    fx->shader()->gocMaterial()->setDiffuse(color);
    fx->shader()->gocMaterial()->setTransparency(opacity);

    //vl::String fragmentShaderSource   = LoadGLSLSourceFromResources("volume_raycast_isosurface_transp.fs");
    //vl::String vertexShaderSource     = LoadGLSLSourceFromResources("volume_luminance_light.vs");

    //// The GLSL program used to perform the actual rendering.
    //// The \a volume_luminance_light.fs fragment shader allows you to specify how many
    //// lights to use (up to 4) and can optionally take advantage of a precomputed normals texture.
    //ref<vl::GLSLProgram>    glslShader = fx->shader()->gocGLSLProgram();
    //glslShader->attachShader(new vl::GLSLFragmentShader(fragmentShaderSource));
    //glslShader->attachShader(new vl::GLSLVertexShader(vertexShaderSource));

    ref<vl::Actor> imageActor = new vl::Actor;
    imageActor->setEffect(fx.get());
    // imageActor->setUniform(m_ThresholdVal.get());

    ref<vl::Transform>    tr = new vl::Transform;
    //UpdateTransfromFromData(tr, cudaImg);       // FIXME: needs proper thinking through
    imageActor->setTransform(tr.get());
    m_VividRendering->sceneManager()->tree()->addActor(imageActor.get());
    imageActor->setEnableMask( vl::VividRenderer::VolumeEnableMask );

    // this is a callback: gets triggered everytime its bound actor is to be rendered.
    // during that callback it updates the uniforms of our glsl shader to match fixed-function state.
    ref<vl::RaycastVolume>    raycastVolume = new vl::RaycastVolume;
    // this stuffs the proxy geometry onto our actor, as lod-slot zero.
    raycastVolume->bindActor(imageActor.get());

    // we do not own dims!
    unsigned int*   dims    = m_MitkImage->GetDimensions();
    mitk::Vector3D  spacing = m_MitkImage->GetGeometry()->GetSpacing();

    float dimX = (float) dims[0] * spacing[0] / 2.0f;
    float dimY = (float) dims[1] * spacing[1] / 2.0f;
    float dimZ = (float) dims[2] * spacing[2] / 2.0f;
    float shiftX = 0.0f;//0.5f * spacing[0];
    float shiftY = 0.0f;//0.5f * spacing[1];
    float shiftZ = 0.0f;//0.5f * spacing[2];

    vl::AABB volume_box(vl::vec3(-dimX + shiftX, -dimY + shiftY, -dimZ + shiftZ)
                         , vl::vec3( dimX + shiftX,  dimY + shiftY,  dimZ + shiftZ));
    raycastVolume->setBox(volume_box);
    raycastVolume->generateTextureCoordinates(vl::ivec3(vl_img->width(), vl_img->height(), vl_img->depth()));


    // note img has been converted unconditionally to IT_UNSIGNED_SHORT above!
    fx->shader()->gocTextureSampler(0)->setTexture(new vl::Texture(vl_img.get(), vl::TF_LUMINANCE16, false, false));
    fx->shader()->gocUniform("volume_texunit")->setUniformI(0);

    // generate a simple colored transfer function
    ref<vl::Image>  trfunc = vl::makeColorSpectrum(1024, vl::blue, vl::royalblue, vl::green, vl::yellow, vl::crimson);
    // installs the transfer function as texture #1
    fx->shader()->gocTextureSampler(1)->setTexture(new vl::Texture(trfunc.get()));
    fx->shader()->gocUniform("trfunc_texunit")->setUniformI(1);
  /*
    ref<Image> gradient;
    // note that this can take a while...
    gradient = vl::genGradientNormals( vl_img.get() );
    fx->shader()->gocUniform( "precomputed_gradient" )->setUniformI( 1);
    fx->shader()->gocTextureSampler( 2 )->setTexture( new Texture( gradient.get(), TF_RGBA, false, false ) );
    fx->shader()->gocUniform( "gradient_texunit" )->setUniformI( 2 );
  */
    fx->shader()->gocUniform("precomputed_gradient")->setUniformI(0);
    // used to compute on the fly the normals based on the volume's gradient
    fx->shader()->gocUniform("gradient_delta")->setUniform(vl::fvec3(0.5f / vl_img->width(), 0.5f / vl_img->height(), 0.5f / vl_img->depth()));

    fx->shader()->gocUniform( "sample_step" )->setUniformF(1.0f / 512.0f);

    vtkLinearTransform * nodeVtkTr = m_MitkImage->GetGeometry()->GetVtkTransform();
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
    //openglContext()->update();

    m_Actor = imageActor;
  }

  virtual void update() {}

protected:
  mitk::Image::Pointer m_MitkImage;
};

//-----------------------------------------------------------------------------

class VLMapperCoordinateAxes: public VLMapper {
public:
  VLMapperCoordinateAxes( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    m_MitkAxes = dynamic_cast<mitk::CoordinateAxesData*>( node->GetData() );
    VIVID_CHECK( m_MitkAxes );
  }

  virtual void init() {
    VIVID_CHECK( m_MitkAxes );

    ref<vl::ArrayFloat3> verts  = m_Vertices = new vl::ArrayFloat3;
    ref<vl::ArrayFloat4> colors = new vl::ArrayFloat4;
    verts->resize(6);
    colors->resize(6);
  
    // Axis length
    int S = 100;
    mitk::IntProperty::Pointer size_prop = dynamic_cast<mitk::IntProperty*>(m_DataNode->GetProperty("size"));
    if ( size_prop ) {
      S = size_prop->GetValue();
    }

    // X Axis
    verts ->at(0) = vec3(0, 0, 0);
    verts ->at(1) = vec3(S, 0, 0);
    colors->at(0) = vl::red;
    colors->at(1) = vl::red;
    // Y Axis
    verts ->at(2) = vec3(0, 0, 0);
    verts ->at(3) = vec3(0, S, 0);
    colors->at(2) = vl::green;
    colors->at(3) = vl::green;
    // Z Axis
    verts ->at(4) = vec3(0, 0, 0);
    verts ->at(5) = vec3(0, 0, S);
    colors->at(4) = vl::blue;
    colors->at(5) = vl::blue;

    ref<vl::Geometry> geom = new vl::Geometry;
    geom->drawCalls().push_back( new vl::DrawArrays( vl::PT_LINES, 0, 6 ) );
    geom->setVertexArray(verts.get());
    geom->setColorArray(colors.get());

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    fx->shader()->getLineWidth()->set( 2 );
    // Use color array instead of lighting
    fx->shader()->gocUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );
  }

  virtual void update() {
    mitk::IntProperty::Pointer size_prop = dynamic_cast<mitk::IntProperty*>(m_DataNode->GetProperty("size"));
    if ( size_prop ) {
      ref<vl::ArrayFloat3> verts = m_Vertices;
      int S = size_prop->GetValue();
      // X Axis
      verts ->at(0) = vec3(0, 0, 0);
      verts ->at(1) = vec3(S, 0, 0);
      // Y Axis
      verts ->at(2) = vec3(0, 0, 0);
      verts ->at(3) = vec3(0, S, 0);
      // Z Axis
      verts ->at(4) = vec3(0, 0, 0);
      verts ->at(5) = vec3(0, 0, S);
      // Update VBO
      m_Vertices->updateBufferObject();
    }
  }

protected:
  mitk::CoordinateAxesData::Pointer m_MitkAxes;
  ref<vl::ArrayFloat3> m_Vertices;
};

//-----------------------------------------------------------------------------

class VLMapperPointSet: public VLMapper {
public:
  VLMapperPointSet( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    m_MitkPointSet = dynamic_cast<mitk::PointSet*>( node->GetData() );
    m_3DSphereMode = true;
    initVLPropertiesPointSet();
    VIVID_CHECK( m_MitkPointSet );
  }

  virtual void init() { }

  virtual void init3D() {
    VIVID_CHECK( m_MitkPointSet );
    VIVID_CHECK( m_3DSphereMode );

    // Remove 2D data and init 3D data.
    remove();
    m_SphereActors = new vl::ActorTree;
    m_VividRendering->sceneManager()->tree()->addChild( m_SphereActors.get() );

    m_3DSphereGeom = vl::makeIcosphere( vec3(0,0,0), 1, 2, true );
    m_3DSphereFX = vl::VividRendering::makeVividEffect();
    int j = 0;
    for (mitk::PointSet::PointsConstIterator i = m_MitkPointSet->Begin(); i != m_MitkPointSet->End(); ++i, ++j)
    {
      mitk::PointSet::PointType p = i->Value();
      vl::vec3 pos( p[0], p[1], p[2] );
      ref<Actor> actor = initActor( m_3DSphereGeom.get(), m_3DSphereFX.get() );
      ref<vl::Transform> tr = new vl::Transform( vl::mat4::getTranslation( pos ) );
      actor->setTransform( tr.get() );
      m_SphereActors->addActor( actor.get() );
    }
  }

  virtual void init2D() {
    VIVID_CHECK( m_MitkPointSet );
    VIVID_CHECK( ! m_3DSphereMode );

    // Remove 3D data and init 2D data.
    remove();

    ref<vl::ArrayFloat3> verts = new vl::ArrayFloat3;
    verts->resize(m_MitkPointSet->GetSize());
    int j = 0;
    for (mitk::PointSet::PointsConstIterator i = m_MitkPointSet->Begin(); i != m_MitkPointSet->End(); ++i, ++j)
    {
      mitk::PointSet::PointType p = i->Value();
      verts->at(j).x() = p[0];
      verts->at(j).y() = p[1];
      verts->at(j).z() = p[2];
    }

    m_2DGeometry = new vl::Geometry;
    ref<vl::DrawArrays> draw_arrays = new vl::DrawArrays( vl::PT_POINTS, 0, verts->size() );
    m_2DGeometry->drawCalls().push_back(draw_arrays.get());
    m_2DGeometry->setVertexArray( verts.get() );
    m_2DGeometry->setColorArray( vl::white );

    m_Actor = initActor( m_2DGeometry.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<vl::Effect> fx = m_Actor->effect();
    fx->shader()->getUniform( "vl_Vivid.enableLighting" )->setUniformI( 0 );
    fx->shader()->getUniform( "vl_Vivid.enablePointSprite" )->setUniformI( 1 );
    fx->shader()->gocUniform( "vl_Vivid.enableTextureMapping" )->setUniformI( 1 );
    ref<vl::Image> img = new Image("/vivid/images/sphere.png");
    ref<vl::Texture> texture = fx->shader()->getTextureSampler( vl::VividRendering::UserTexture )->texture();
    texture->createTexture2D( img.get(), vl::TF_UNKNOWN, false, false );
  }

  virtual void update() {
    // Get mode
    int mode = 0;
    const mitk::EnumerationProperty* mode_prop = dynamic_cast<const mitk::EnumerationProperty*>( m_DataNode->GetProperty( "VL.PointMode" ) );
    if ( mode_prop ) {
      mode = mode_prop->GetValueAsId();
    }
    m_3DSphereMode = 0 == mode;

    // Get point size
    float pointsize = 1;
    m_DataNode->GetFloatProperty( m_3DSphereMode ? "VL.PointSize3D" : "VL.PointSize2D", pointsize );

    // Get color
    float rgb[3];
    m_DataNode->GetColor( rgb, NULL, "VL.PointColor" );

    // Get opacity
    float opacity = 1;
    m_DataNode->GetFloatProperty( "VL.PointOpacity", opacity );

    if ( m_3DSphereMode ) {
      if ( ! m_SphereActors ) {
        init3D();
      }

      // Set color
      m_3DSphereFX->shader()->getMaterial()->setDiffuse( vl::vec4( rgb[0], rgb[1], rgb[2], opacity ) );
      // Set size
      for( int i = 0; i < m_SphereActors->actors()->size(); ++i ) {
        Transform* tr = m_SphereActors->actors()->at( i )->transform();
        tr->worldMatrix().e(0,0) = pointsize * 2;
        tr->worldMatrix().e(1,1) = pointsize * 2;
        tr->worldMatrix().e(2,2) = pointsize * 2;
      }
    } else {
      if ( ! m_2DGeometry ) {
        init2D();
      }
      
      VIVID_CHECK( m_Actor );

      Shader* shader = m_Actor->effect()->shader();
      // This is part of the standard vivid shader so it must be present.
      VIVID_CHECK( shader->getPointSize() );
      shader->getPointSize()->set( pointsize );
      m_2DGeometry->setColorArray( vl::vec4( rgb[0], rgb[1], rgb[2], opacity ) );
    }
  }

  void remove() {
    VLMapper::remove();
    m_2DGeometry = NULL;
    if ( m_SphereActors ) {
      m_SphereActors->actors()->clear();
      m_VividRendering->sceneManager()->tree()->eraseChild( m_SphereActors.get() );
      m_SphereActors = NULL;
      m_3DSphereGeom = NULL;
      m_3DSphereFX = NULL;
    }
  }

protected:
  mitk::PointSet::Pointer m_MitkPointSet;
  bool m_3DSphereMode;
  ref<vl::ActorTree> m_SphereActors;
  ref<Geometry> m_3DSphereGeom;
  ref<Effect> m_3DSphereFX;
  ref<vl::Geometry> m_2DGeometry;
};

//-----------------------------------------------------------------------------

#ifdef _USE_PCL
/*    
       WARNING: 
never compiled nor tested

     _.--""--._
    /  _    _  \
 _  ( (_\  /_) )  _
{ \._\   /\   /_./ }
/_"=-.}______{.-="_\
 _  _.=("""")=._  _
(_'"_.-"`~~`"-._"'_)
 {_"            "_}

*/                          
class VLMapperPCL: public VLMapper {
public:
  VLMapperPCL( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    m_NiftkPCL = dynamic_cast<niftk::PCLData*>( node->GetData() );
    VIVID_CHECK( m_NiftkPCL );
  }

  virtual void init() {
    VIVID_CHECK( m_NiftkPCL );
    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud = m_NiftkPCL->GetCloud();

    ref<vl::ArrayFloat3> vl_verts = new vl::ArrayFloat3;
    ref<vl::ArrayFloat4> vl_colors = new vl::ArrayFloat4;
    vl_verts->resize(cloud->size());
    vl_colors->resize(cloud->size());
    // We could interleave the color and vert array but do we trust the VTK layout?
    int j = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::const_iterator i = cloud->begin(); i != cloud->end(); ++i, ++j) {
      const pcl::PointXYZRGB& p = *i;

      vl_verts->at(j).x() = p.x;
      vl_verts->at(j).y() = p.y;
      vl_verts->at(j).z() = p.z;

      vl_colors->at(j).r() = (float)p.r / 255.0f;
      vl_colors->at(j).g() = (float)p.g / 255.0f;
      vl_colors->at(j).b() = (float)p.b / 255.0f;
      vl_colors->at(j).a() = 1;
    }

    ref<vl::Geometry> geom = new vl::Geometry;
    ref<vl::DrawArrays> draw_arrays = new vl::DrawArrays( vl::PT_POINTS, 0, vl_verts->size() );
    geom->drawCalls().push_back( draw_arrays.get() );
    geom->setVertexArray( vl_verts.get() );
    geom->setColorArray( vl_colors.get() );

    m_Actor = initActor( geom.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
  }

  virtual void update() {
    // Update point size
    float pointsize = 1;
    m_DataNode->GetFloatProperty( "pointsize", pointsize );
    Shader* shader = m_Actor->effect()->shader();
    // This is part of the standard vivid shader so it must be present.
    VIVID_CHECK( shader->getPointSize() );
    shader->getPointSize()->set( pointsize );
    if ( pointsize > 1 ) {
      shader->enable( vl::EN_POINT_SMOOTH );
    } else {
      shader->disable( vl::EN_POINT_SMOOTH );
    }
  }

protected:
  niftk::PCLData::Pointer m_NiftkPCL;
};

#endif

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA
/*    
       WARNING: 
never compiled nor tested

     _.--""--._
    /  _    _  \
 _  ( (_\  /_) )  _
{ \._\   /\   /_./ }
/_"=-.}______{.-="_\
 _  _.=("""")=._  _
(_'"_.-"`~~`"-._"'_)
 {_"            "_}


This is just stub code, a raw attempt at reorganizing the legacy experimental CUDA code into the new VLMapper logic

*/                          
class VLMapperCUDAImage: public VLMapper {
public:
  VLMapperCUDAImage( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node )
    : VLMapper( gl, vr, ds, node ) {
    niftk::CUDAImage* cuda_image = dynamic_cast<niftk::CUDAImage*>( node->GetData() );
    if ( cuda_image ) {
      m_NiftkLightweightCUDAImage = cuda_image->GetLightweightCUDAImage();
    } else {
      niftk::CUDAImageProperty* cuda_image_prop = dynamic_cast<niftk::CUDAImageProperty*>(m_DataNode->GetProperty("CUDAImageProperty").GetPointer());
      if  (cuda_image_prop ) {
        m_NiftkLightweightCUDAImage = cuda_image_prop->Get();
      }
    }
    VIVID_CHECK(m_NiftkLightweightCUDAImage.GetId() != 0);
  }

  virtual void init() {

    niftk::LightweightCUDAImage lwci;
    const niftk::CUDAImage* cudaImg = dynamic_cast<const niftk::CUDAImage*>(m_NiftkCUDAImage);
    if (cudaImg != 0)
    {
      lwci = cudaImg->GetLightweightCUDAImage();
    }
    else
    {
      niftk::CUDAImageProperty::Pointer prop = dynamic_cast<niftk::CUDAImageProperty*>(m_DataNode->GetProperty("CUDAImageProperty").GetPointer());
      if (prop.IsNotNull())
      {
        lwci = prop->Get();
      }
    }
    VIVID_CHECK(lwci.GetId() != 0);

    ref<vl::Geometry> vlquad = CreateGeometryFor2DImage(m_NiftkLightweightCUDAImage.GetWidth(), m_NiftkLightweightCUDAImage.GetHeight());

    m_Actor = initActor( vlquad.get() );
    m_VividRendering->sceneManager()->tree()->addActor( m_Actor.get() );
    ref<Effect> fx = m_Actor->effect();

    fx->shader()->disable(vl::EN_LIGHTING);
    fx->shader()->gocTextureSampler(1)->setTexture(m_DefaultTexture.get());
    fx->shader()->gocTextureSampler(1)->setTexParameter(m_DefaultTextureParams.get());
  }

  virtual void update() {
    VIVID_CHECK(m_NiftkLightweightCUDAImage.GetId() != 0);

    // BEWARE: 
    // All the logic below is completely outdated especially with regard to accessing the user texture. See VLMapper2DImage for more info.
    // PS. All the horrific code formatting is from the original code...
    // - Michele

    // whatever we had cached from a previous frame.
    TextureDataPOD          texpod    = m_TextureDataPOD;

    // only need to update the vl texture, if content in our cuda buffer has changed.
    // and the cuda buffer can change only when we have a different id.
    if (texpod.m_LastUpdatedID != m_NiftkLightweightCUDAImage.GetId())
    {
      cudaError_t   err = cudaSuccess;
      bool          neednewvltexture = texpod.m_Texture.get() == 0;

      // check if vl-texture size needs to change
      if (texpod.m_Texture.get() != 0)
      {
        neednewvltexture |= m_NiftkLightweightCUDAImage.GetWidth()  != texpod.m_Texture->width();
        neednewvltexture |= m_NiftkLightweightCUDAImage.GetHeight() != texpod.m_Texture->height();
      }

      if (neednewvltexture)
      {
        if (texpod.m_CUDARes)
        {
          err = cudaGraphicsUnregisterResource(texpod.m_CUDARes);
          texpod.m_CUDARes = 0;
          if (err != cudaSuccess)
          {
            MITK_WARN << "Could not unregister VL texture from CUDA. This will likely leak GPU memory.";
          }
        }

        texpod.m_Texture = new vl::Texture(m_NiftkLightweightCUDAImage.GetWidth(), m_NiftkLightweightCUDAImage.GetHeight(), vl::TF_RGBA8, false);
        actor->effect()->shader()->gocTextureSampler(0)->setTexture(texpod.m_Texture.get());
        actor->effect()->shader()->gocTextureSampler(0)->setTexParameter(m_DefaultTextureParams.get());

        err = cudaGraphicsGLRegisterImage(&texpod.m_CUDARes, texpod.m_Texture->handle(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
        if (err != cudaSuccess)
        {
          texpod.m_CUDARes = 0;
          MITK_WARN << "Registering VL texture into CUDA failed. Will not update (properly).";
        }
      }

      if (texpod.m_CUDARes)
      {
        VIVID_CHECK(actor->effect()->shader()->getTextureSampler(0)->texture() == texpod.m_Texture);

        niftk::CUDAManager*  cudamng   = niftk::CUDAManager::GetInstance();
        cudaStream_t         mystream  = cudamng->GetStream("VLSceneView vl-texture update");
        niftk::ReadAccessor  inputRA   = cudamng->RequestReadAccess(m_NiftkLightweightCUDAImage);

        // make sure producer of the cuda-image finished.
        err = cudaStreamWaitEvent(mystream, inputRA.m_ReadyEvent, 0);
        if (err != cudaSuccess)
        {
          // flood the log
          MITK_WARN << "cudaStreamWaitEvent failed with error code " << err;
        }

        // this also guarantees that ogl will have finished doing its thing before mystream starts copying.
        err = cudaGraphicsMapResources(1, &texpod.m_CUDARes, mystream);
        if (err == cudaSuccess)
        {
          // normally we would need to flip image! ogl is left-bottom, whereas everywhere else is left-top origin.
          // but texture coordinates that we have assigned to the quads rendering the current image will do that for us.

          cudaArray_t   arr = 0;
          err = cudaGraphicsSubResourceGetMappedArray(&arr, texpod.m_CUDARes, 0, 0);
          if (err == cudaSuccess)
          {
            err = cudaMemcpy2DToArrayAsync(arr, 0, 0, inputRA.m_DevicePointer, inputRA.m_BytePitch, m_NiftkLightweightCUDAImage.GetWidth() * 4, m_NiftkLightweightCUDAImage.GetHeight(), cudaMemcpyDeviceToDevice, mystream);
            if (err == cudaSuccess)
            {
              texpod.m_LastUpdatedID = m_NiftkLightweightCUDAImage.GetId();
            }
          }

          err = cudaGraphicsUnmapResources(1, &texpod.m_CUDARes, mystream);
          if (err != cudaSuccess)
          {
            MITK_WARN << "Cannot unmap VL texture from CUDA. This will probably kill the renderer. Error code: " << err;
          }
        }
        // make sure Autorelease() and Finalise() are always the last things to do for a stream!
        // otherwise the streamcallback will block subsequent work.
        // in this case here, the callback managed by CUDAManager that keeps track of refcounts could stall
        // the opengl driver if cudaGraphicsUnmapResources() came after Autorelease().
        cudamng->Autorelease(inputRA, mystream);
      }

      // update cache, even if something went wrong.
      m_TextureDataPOD = texpod;

      // helps with debugging
      actor->effect()->shader()->disable(vl::EN_CULL_FACE);
    }
  }

  virtual void remove() {
    if ( m_TextureDataPOD.m_CUDARes ) {
      cudaError_t err = cudaGraphicsUnregisterResource( m_TextureDataPOD.m_CUDARes );
      if (err != cudaSuccess)
      {
        MITK_WARN << "Failed to unregister VL texture from CUDA";
      }
      m_TextureDataPOD.m_CUDARes = 0;
    }

    VLMapper::remove();
  }

protected:
  niftk::LightweightCUDAImage m_NiftkLightweightCUDAImage;
  TextureDataPOD m_TextureDataPOD; // m_NodeToTextureMap
};

#endif

//-----------------------------------------------------------------------------

vl::ref<VLMapper> VLMapper::create( vl::OpenGLContext* gl, vl::VividRendering* vr, mitk::DataStorage* ds, const mitk::DataNode* node ) {
  
  // Map DataNode type to VLMapper type
  vl::ref<VLMapper> vl_node;

  mitk::Surface*            mitk_surf = dynamic_cast<mitk::Surface*>(node->GetData());
  mitk::Image*              mitk_image = dynamic_cast<mitk::Image*>( node->GetData() );
  mitk::CoordinateAxesData* mitk_axes = dynamic_cast<mitk::CoordinateAxesData*>( node->GetData() );
  mitk::PointSet*           mitk_pset = dynamic_cast<mitk::PointSet*>( node->GetData() );
#ifdef _USE_PCL
  niftk::PCLData*           mitk_pcld = dynamic_cast<niftk::PCLData*>( node->GetData() );
#endif
#ifdef _USE_CUDA
  mitk::BaseData*           cuda_img = dynamic_cast<niftk::CUDAImage*>( node->GetData() );
#endif

  if ( mitk_surf ) {
    vl_node = new VLMapperSurface( gl, vr, ds, node );
  } 
  else if ( mitk_image ) {
    unsigned int depth = mitk_image->GetDimensions()[2];
    // In VTK a NxMx1 image is 2D (in VL a 2D image is NxMx0)
    if ( depth <= 1 ) {
      vl_node = new VLMapper2DImage( gl, vr, ds, node );
    } else {
      vl_node = new VLMapper3DImage( gl, vr, ds, node );
    }
  } 
  else  if ( mitk_axes ) {
    vl_node = new VLMapperCoordinateAxes( gl, vr, ds, node );
  } 
  else if ( mitk_pset ) {
    vl_node = new VLMapperPointSet( gl, vr, ds, node );
  }
#ifdef _USE_PCL
  else if ( mitk_pcld ) {
    vl_node = new VLMapperPCL( gl, vr, ds, node );
  }
#endif
#ifdef _USE_CUDA
  else if ( mitk_pcld ) {
    vl_node = new VLMapperCUDAImage( gl, vr, ds, node );
  }
#endif
  return vl_node;
}

//-----------------------------------------------------------------------------
// VLSceneView
//-----------------------------------------------------------------------------

VLSceneView::VLSceneView() : 
  // Qt5Widget(parent, shareWidget, f)
  m_BackgroundWidth( 0 )
  , m_BackgroundHeight( 0 )
  , m_ScheduleTrackballAdjustView( true )
  , m_ScheduleInitScene ( true )
  , m_OclService( 0 )
#ifdef _USE_CUDA
  , m_CUDAInteropPimpl(0)
#endif
{
}

//-----------------------------------------------------------------------------

 void VLSceneView::destroyEvent()
{
  openglContext()->makeCurrent();

  RemoveDataStorageListeners();

#ifdef _USE_CUDA
  FreeCUDAInteropTextures();
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::AddDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.    AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeAdd));
    m_DataStorage->ChangedNodeEvent.AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeUpdate));
    m_DataStorage->RemoveNodeEvent. AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeRemove));
    m_DataStorage->DeleteNodeEvent. AddListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeRemove));
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::RemoveDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->AddNodeEvent.    RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeAdd));
    m_DataStorage->ChangedNodeEvent.RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeUpdate));
    m_DataStorage->RemoveNodeEvent. RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeRemove));
    m_DataStorage->DeleteNodeEvent. RemoveListener(mitk::MessageDelegate1<VLSceneView, const mitk::DataNode*>(this, &VLSceneView::ScheduleNodeRemove));
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::SetDataStorage(const mitk::DataStorage::Pointer& dataStorage)
{
  openglContext()->makeCurrent();

  RemoveDataStorageListeners();

#ifdef _USE_CUDA
  FreeCUDAInteropTextures();
#endif

  m_DataStorage = dataStorage;
  AddDataStorageListeners();

  ClearScene();

  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::SetOclResourceService(OclResourceService* oclserv)
{
 // no idea if this is really a necessary restriction.
 // if it is then maybe the ocl-service should be a constructor parameter.
 if (m_OclService != 0)
   throw std::runtime_error("Can set OpenCL service only once");

 m_OclService = oclserv;
}

//-----------------------------------------------------------------------------

void VLSceneView::ScheduleNodeAdd( const mitk::DataNode* node )
{
  if ( ! node || ! node->GetData() ) {
    return;
  }

  // m_NodesToRemove.erase( node ); // remove it first
  m_NodesToAdd.insert( mitk::DataNode::ConstPointer ( node ) ); // then add
  // m_NodesToUpdate.erase( node ); // then update
  openglContext()->update();

  const char* noc = node->GetData() ? node->GetData()->GetNameOfClass() : "<name-of-class>";
  printf("ScheduleNodeAdd: %s (%s)\n", node->GetName().c_str(), noc );
}

//-----------------------------------------------------------------------------

void VLSceneView::ScheduleNodeUpdate( const mitk::DataNode* node )
{
  if ( ! node || ! node->GetData() ) {
    return;
  }

  m_NodesToRemove.erase( node ); // abort the removal
  // m_NodesToAdd.erase( node ); // let it add it first
  m_NodesToUpdate.insert( mitk::DataNode::ConstPointer ( node ) ); // then update
  openglContext()->update();

  const char* noc = node->GetData() ? node->GetData()->GetNameOfClass() : "<unknown-class>";
  printf("ScheduleNodeUpdate: %s (%s)\n", node->GetName().c_str(), noc );
}

//-----------------------------------------------------------------------------

void VLSceneView::ScheduleNodeRemove( const mitk::DataNode* node )
{
  if ( ! node /* || ! node->GetData() */ ) {
    return;
  }

  m_NodesToRemove.insert( mitk::DataNode::ConstPointer ( node ) ); // remove it
  m_NodesToAdd.erase( node );    // abort the addition
  m_NodesToUpdate.erase( node ); // abort the update
  openglContext()->update();

  const char* noc = node->GetData() ? node->GetData()->GetNameOfClass() : "<name-of-class>";
  printf("ScheduleNodeRemove: %s (%s)\n", node->GetName().c_str(), noc );
}

//-----------------------------------------------------------------------------

void VLSceneView::InitSceneFromDataStorage()
{
  // Make sure the system is initialized
  VIVID_CHECK( m_VividRendering.get() );

  openglContext()->makeCurrent();

  ClearScene();

  if ( m_DataStorage.IsNull() ) {
    return;
  }

  typedef itk::VectorContainer<unsigned int, mitk::DataNode::Pointer> NodesContainerType;
  NodesContainerType::ConstPointer vc = m_DataStorage->GetAll();

  for (unsigned int i = 0; i < vc->Size(); ++i)
  {
    mitk::DataNode::Pointer currentDataNode = vc->ElementAt(i);
    if (currentDataNode.IsNull() || currentDataNode->GetData()== 0) {
      continue;
    } else {
      AddDataNode( mitk::DataNode::ConstPointer( currentDataNode.GetPointer() ) );
    }
  }

  #if 0
    // dump scene to VLB/VLT format for debugging
    ref< vl::ResourceDatabase > db = new vl::ResourceDatabase;
    for( int i = 0; i < m_SceneManager->tree()->actors()->size(); ++i ) {
      vl::Actor* act = m_SceneManager->tree()->actors()->at(i);
      if ( act->enableMask() ) {
        // db->resources().push_back( act );
        // vl::String fname = filename( files[i] );
        db->resources().push_back( act );
        vl::String fname = "niftk-liver";
        vl::saveVLT( "C:/git-ucl/VisualizationLibrary/data/tmp/" + fname + ".vlt", db.get() );
        vl::saveVLB( "C:/git-ucl/VisualizationLibrary/data/tmp/" + fname + ".vlb", db.get() );
      }
    }
  #endif
}

//-----------------------------------------------------------------------------

void VLSceneView::AddDataNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  // Add only once and only if valid
  if ( ! node || ! node->GetData() || GetVLMapper( node ) != NULL ) {
    return;
  }

  #if 0
    dumpNodeInfo( "AddDataNode()", node );
  #endif

  ref<VLMapper> vl_node = VLMapper::create( openglContext(), m_VividRendering.get(), m_DataStorage.GetPointer(), node.GetPointer() );
  if ( vl_node ) {
    m_DataNodeVLMapperMap[ node ] = vl_node;
    vl_node->init();
    vl_node->updateCommon();
    vl_node->update();
    return;
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::RemoveDataNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  // dont leave a dangling update behind.
  m_NodesToUpdate.erase(node);
  m_NodesToAdd.erase(node);

  // Remove VLMapper and VL data
  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  if ( it != m_DataNodeVLMapperMap.end() ) {
    VLMapper* vl_node = it->second.get();
    VIVID_CHECK( vl_node );
    if ( vl_node ) {
      vl_node->remove();
      m_DataNodeVLMapperMap.erase(it);
    }
  }
}

void VLSceneView::UpdateDataNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  if ( node.IsNull() || node->GetData() == 0 ) {
    return;
  }

  #if 1
    dumpNodeInfo( "UpdateDataNode()", node );
  #endif

  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  if ( it != m_DataNodeVLMapperMap.end() ) {
    it->second->updateCommon();
    it->second->update();
    return;
  }

  // Update camera
  if (node == m_CameraNode) {
    UpdateCameraParameters();
  }
}

//-----------------------------------------------------------------------------

VLMapper* VLSceneView::GetVLMapper( const mitk::DataNode::ConstPointer& node )
{
  DataNodeVLMapperMapType::iterator it = m_DataNodeVLMapperMap.find( node );
  return it == m_DataNodeVLMapperMap.end() ? NULL : it->second.get();
}

//-----------------------------------------------------------------------------

void VLSceneView::SetBackgroundColour(float r, float g, float b)
{
  m_VividRendering->camera()->viewport()->setClearColor(vl::fvec4(r, g, b, 1));
  openglContext()->update();
}

//-----------------------------------------------------------------------------

void VLSceneView::initEvent()
{
  VIVID_CHECK( contextIsCurrent() );

  // vl::OpenGLContext::initGLContext();

  // Interface VL with Qt's resource system to load GLSL shaders.
  vl::defFileSystem()->directories().clear();
  vl::defFileSystem()->directories().push_back( new vl::QtDirectory( ":/VL/" ) );


#if 0
  // Mic: this seems to be failing for me.
  // use the device that is running our opengl context as the compute-device
  // for sorting triangles in the correct order.
  if (m_OclService)
  {
    // Force tests to run on the first GPU with shared context
    m_OclService->SpecifyPlatformAndDevice(0, 0, true);
    // Calling this to make sure that the context is created right at startup
    cl_context clContext = m_OclService->GetContext();
  }
#endif

#ifdef _MSC_VER
  // NvAPI_OGL_ExpertModeSet(NVAPI_OGLEXPERT_DETAIL_ALL, NVAPI_OGLEXPERT_DETAIL_BASIC_INFO, NVAPI_OGLEXPERT_OUTPUT_TO_ALL, 0);
#endif

  // Create our VividRendering!
  m_VividRendering = new vl::VividRendering;
  m_VividRendering->setRenderingMode( vl::VividRendering::FrontToBackDepthPeeling ); /* (default) */
  m_VividRendering->setCullingEnabled( false );
  // This creates some flickering on the skin for some reason
  m_VividRendering->setNearFarClippingPlanesOptimized( false );

  // VividRendering nicely prepares for us all the structures we need to use ;)
  m_VividRenderer = m_VividRendering->vividRenderer();
  m_VividVolume = m_VividRendering->vividVolume();
  m_SceneManager = m_VividRendering->sceneManager();

  // In the future Camera (and Trackball) should belong in VLView and be set upon rendering.
  m_Camera = m_VividRendering->calibratedCamera();

  // Initialize the trackball manipulator
  m_Trackball = new VLTrackballManipulator;
  m_Trackball->setEnabled( true );
  m_Trackball->setCamera( m_Camera.get() );
  m_Trackball->setTransform( NULL );
  m_Trackball->setPivot( vl::vec3(0,0,0) );
  openglContext()->addEventListener( m_Trackball.get() );
  // Schedule reset of the camera based on the scene content
  ScheduleTrackballAdjustView();

  // This is only used by the CUDA stuff
  CreateAndUpdateFBOSizes( openglContext()->width(), openglContext()->height() );

#if 0
  // Point cloud data test
  mitk::DataNode::Pointer n = mitk::DataNode::New();
  mitk::PCLData::Pointer  p = niftk::PCLData::New();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  c(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int i = 0; i < 100; ++i)
  {
    pcl::PointXYZRGB  q(std::rand() % 255, std::rand() % 255, std::rand() % 255);
    q.x = std::rand() % 255;
    q.y = std::rand() % 255;
    q.z = std::rand() % 255;
    c->push_back(q);
  }
  p->SetCloud(c);
  n->SetData(p);

  m_DataStorage->Add(n);
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::resizeEvent( int w, int h )
{
   VIVID_CHECK( contextIsCurrent() );

  // dont do anything if window is zero size.
  // it's an opengl error to have a viewport like that!
  if ( w <= 0 || h <= 0 ) {
    return;
  }

  m_VividRendering->camera()->viewport()->set( 0, 0, w, h );
  m_VividRendering->camera()->setProjectionPerspective();

  CreateAndUpdateFBOSizes( w, h );

  // MIC FIXME: update calibrated camera setup
  UpdateViewportAndCameraAfterResize();
}

//-----------------------------------------------------------------------------

void VLSceneView::updateEvent()
{
  VIVID_CHECK( contextIsCurrent() );

  RenderScene();
}

//-----------------------------------------------------------------------------

void VLSceneView::CreateAndUpdateFBOSizes( int width, int height )
{
  openglContext()->makeCurrent();

#ifdef _USE_CUDA
  // sanitise dimensions. depending on how windows are resized we can get zero here.
  // but that breaks on the ogl side.
  width  = std::max(1, width);
  height = std::max(1, height);

  ref<vl::FramebufferObject> opaqueFBO = vl::OpenGLContext::createFramebufferObject(width, height);
  opaqueFBO->setObjectName("opaqueFBO");
  opaqueFBO->addDepthAttachment(new vl::FBODepthBufferAttachment(vl::DBF_DEPTH_COMPONENT24));
  opaqueFBO->addColorAttachment(vl::AP_COLOR_ATTACHMENT0, new vl::FBOColorBufferAttachment(vl::CBF_RGBA));   // this is a renderbuffer
  opaqueFBO->setDrawBuffer(vl::RDB_COLOR_ATTACHMENT0);

  if (m_CUDAInteropPimpl)
  {
    delete m_CUDAInteropPimpl->m_FBOAdaptor;
    m_CUDAInteropPimpl->m_FBOAdaptor = new VLFramebufferAdaptor(opaqueFBO.get());
  }
#endif
}

//-----------------------------------------------------------------------------

void VLSceneView::UpdateViewportAndCameraAfterResize()
{
  // some sane defaults
  // m_Camera->viewport()->set( 0, 0, QWidget::width(), QWidget::height() );
  // m_BackgroundCamera->viewport()->set(0, 0, QWidget::width(), QWidget::height());

  if ( m_BackgroundNode.IsNotNull() )
  {
    //NodeActorMapType::iterator ni = m_NodeActorMap.find(m_BackgroundNode);
    //if (ni == m_NodeActorMap.end())
    //{
    //  // actor not ready yet, try again later.
    //  // this is getting messy... but stuffing our widget here into an editor causes various methods
    //  // to be called at the wrong time.
    //  QMetaObject::invokeMethod(this, "UpdateViewportAndCameraAfterResize", Qt::QueuedConnection);
    //}
    //else
    //{
      // ref<vl::Actor> backgroundactor = ni->second;

      // this is based on my old araknes video-ar app.
      // FIXME: aspect ratio?
      float   width_scale  = (float) openglContext()->width()  / (float) m_BackgroundWidth;
      float   height_scale = (float) openglContext()->height() / (float) m_BackgroundHeight;
      int     vpw = openglContext()->width();
      int     vph = openglContext()->height();
      if (width_scale < height_scale)
        vph = (int) ((float) m_BackgroundHeight * width_scale);
      else
        vpw = (int) ((float) m_BackgroundWidth * height_scale);

      int   vpx = openglContext()->width()  / 2 - vpw / 2;
      int   vpy = openglContext()->height() / 2 - vph / 2;

      // m_BackgroundCamera->viewport()->set(vpx, vpy, vpw, vph);
      // the main-scene-camera should conform to this viewport too!
      // otherwise geometry would never line up with the background (for overlays, etc).
      m_Camera->viewport()->set(vpx, vpy, vpw, vph);
    //}
  }
  // this default perspective depends on the viewport!
  m_Camera->setProjectionPerspective();

  UpdateCameraParameters();
}

void VLSceneView::UpdateScene() {
  // Make sure the system is initialized
  VIVID_CHECK( m_VividRendering.get() );
  VIVID_CHECK( contextIsCurrent() );

  if ( m_ScheduleInitScene ) {
    InitSceneFromDataStorage();
    m_ScheduleInitScene = false;
  } else {
    // Execute scheduled removals
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToRemove.begin(); it != m_NodesToRemove.end(); ++it)
    {
      RemoveDataNode(*it);
    }
    m_NodesToRemove.clear();

    // Execute scheduled additions
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToAdd.begin(); it != m_NodesToAdd.end(); ++it)
    {
      AddDataNode(*it);
    }
    m_NodesToAdd.clear();

    // Execute scheduled updates
    for ( std::set<mitk::DataNode::ConstPointer>::const_iterator it = m_NodesToUpdate.begin(); it != m_NodesToUpdate.end(); ++it)
    {
      UpdateDataNode(*it);
    }
    m_NodesToUpdate.clear();
  }

  // Reset trackball view on demand

  if ( m_ScheduleTrackballAdjustView ) {
    m_Trackball->adjustView( m_VividRendering.get(), vl::vec3(0,0,1), vl::vec3(0,1,0), 1.0f );
    m_ScheduleTrackballAdjustView = false;
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::RenderScene()
{
  VIVID_CHECK( contextIsCurrent() );

  UpdateScene();

  // Set frame time for all the rendering
  vl::real now_time = vl::Time::currentTime();
  m_VividRendering->setFrameClock( now_time );

  // Execute rendering
  m_VividRendering->render( openglContext()->framebuffer() );

  // Show rendering
  if ( openglContext()->hasDoubleBuffer() ) {
#ifdef _USE_CUDA
    cudaSwapBuffers();
#endif
    openglContext()->swapBuffers();
  }

  VL_CHECK_OGL();
}

//-----------------------------------------------------------------------------

void VLSceneView::ClearScene()
{
  openglContext()->makeCurrent();

  if ( m_SceneManager )
  {
    if ( m_SceneManager->tree() ) {
      m_SceneManager->tree()->actors()->clear();
      m_SceneManager->tree()->eraseAllChildren();
    }
  }

  m_CameraNode = 0;
  m_BackgroundNode = 0;
  m_DataNodeVLMapperMap.clear();
  m_NodesToUpdate.clear();
  m_NodesToAdd.clear();
  m_NodesToRemove.clear();
}

//-----------------------------------------------------------------------------

void VLSceneView::UpdateThresholdVal( int isoVal )
{
  float iso = isoVal / 10000.0f;
  iso = vl::clamp( iso, 0.0f, 1.0f );
  m_VividRendering->vividVolume()->setIsoValue( iso );
}

//-----------------------------------------------------------------------------

bool VLSceneView::SetCameraTrackingNode(const mitk::DataNode::ConstPointer& node)
{
  VIVID_CHECK( m_Trackball );

  // Whenever we set the camera node to NULL we recenter the scene using the trackball

  m_CameraNode = node;

  if (m_CameraNode.IsNull())
  {
    m_Trackball->setEnabled( true );
    ScheduleTrackballAdjustView( true );
  } else {
    dumpNodeInfo( "CameraNode()", node );
    m_Trackball->setEnabled( false );
    ScheduleTrackballAdjustView( false );
    UpdateCameraParameters();
  }

  openglContext()->update();

  return true;
}

//-----------------------------------------------------------------------------

void VLSceneView::UpdateCameraParameters()
{
  // calibration parameters come from the background node.
  // so no background, no camera parameters.
  if (m_BackgroundNode.IsNotNull())
  {
#ifdef BUILD_IGI
    mitk::BaseProperty::Pointer cambp = m_BackgroundNode->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName);
    if (cambp.IsNotNull())
    {
      mitk::CameraIntrinsicsProperty::Pointer cam = dynamic_cast<mitk::CameraIntrinsicsProperty*>(cambp.GetPointer());
      if (cam.IsNotNull())
      {
        mitk::CameraIntrinsics::Pointer nodeIntrinsic = cam->GetValue();

        if (nodeIntrinsic.IsNotNull())
        {
          // based on niftkCore/Rendering/vtkOpenGLMatrixDrivenCamera
          float znear = 1;
          float zfar  = 10000;
          float pixelaspectratio = 1;   // FIXME: depends on background image

          vl::mat4  proj;
          proj.setNull();
          proj.e(0, 0) =  2 * nodeIntrinsic->GetFocalLengthX() / (float) m_BackgroundWidth;
          //proj.e(0, 1) = -2 * 0 / m_ImageWidthInPixels;
          proj.e(0, 2) = ((float) m_BackgroundWidth - 2 * nodeIntrinsic->GetPrincipalPointX()) / (float) m_BackgroundWidth;
          proj.e(1, 1) = 2 * (nodeIntrinsic->GetFocalLengthY() / pixelaspectratio) / ((float) m_BackgroundHeight / pixelaspectratio);
          proj.e(1, 2) = (-((float) m_BackgroundHeight / pixelaspectratio) + 2 * (nodeIntrinsic->GetPrincipalPointY() / pixelaspectratio)) / ((float) m_BackgroundHeight / pixelaspectratio);
          proj.e(2, 2) = (-zfar - znear) / (zfar - znear);
          proj.e(2, 3) = -2 * zfar * znear / (zfar - znear);
          proj.e(3, 2) = -1;

          m_Camera->setProjectionMatrix(proj.transpose(), vl::PMT_UserProjection);
        }
      }
    }
#endif
  }

  if (m_CameraNode.IsNotNull())
  {
    vl::mat4 mat = GetVLMatrixFromData(m_CameraNode->GetData());
    if ( ! mat.isNull() ) {
      // beware: there is also a view-matrix! the inverse of modelling-matrix.
      m_Camera->setModelingMatrix(mat);
    }
  }
}

//-----------------------------------------------------------------------------

// MIC FIXME: remove this
void VLSceneView::PrepareBackgroundActor(const mitk::Image* img, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node)
{
  /*
  openglContext()->makeCurrent();

  // nasty
  mitk::Image::Pointer imgp(const_cast<mitk::Image*>(img));
  ref<vl::Actor> actor = Add2DImageActor(imgp);


  // essentially copied from vl::makeGrid()
  ref<vl::Geometry>         vlquad = new vl::Geometry;

  ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  ref<vl::ArrayFloat2> text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = -1; vert3->at(0).y() =  1; vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = -1; vert3->at(1).y() = -1; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() =  1; vert3->at(2).y() = -1; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() =  1; vert3->at(3).y() =  1; vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls().push_back(polys.get());

  // replace original quad with ours.
  actor->setLod(0, vlquad.get());
  actor->effect()->shader()->disable(vl::EN_LIGHTING);

  std::string   objName = actor->objectName() + "_background";
  actor->setObjectName(objName.c_str());

  m_NodeActorMap[node] = actor;
  */
}

//-----------------------------------------------------------------------------

bool VLSceneView::SetBackgroundNode(const mitk::DataNode::ConstPointer& node)
{
  openglContext()->makeCurrent();

  // clear up after previous background node.
  if (m_BackgroundNode.IsNotNull())
  {
    const mitk::DataNode::ConstPointer    oldbackgroundnode = m_BackgroundNode;
    m_BackgroundNode = 0;
    RemoveDataNode(oldbackgroundnode);
    // add back as normal node.
    AddDataNode(oldbackgroundnode);
  }

  // default "no background" value.
  m_BackgroundWidth  = 0;
  m_BackgroundHeight = 0;

  bool    result = false;
  mitk::BaseData::Pointer   basedata;
  if (node.IsNotNull())
    basedata = node->GetData();
  if (basedata.IsNotNull())
  {
    // clear up whatever we had cached for the new background node.
    // it's very likely that it was a normal node before.
    RemoveDataNode(node);

    mitk::Image::Pointer imgdata = dynamic_cast<mitk::Image*>(basedata.GetPointer());
    if (imgdata.IsNotNull())
    {
#ifdef _USE_CUDA
      niftk::CUDAImageProperty::Pointer    cudaimgprop = dynamic_cast<niftk::CUDAImageProperty*>(imgdata->GetProperty("CUDAImageProperty").GetPointer());
      if (cudaimgprop.IsNotNull())
      {
        niftk::LightweightCUDAImage    lwci = cudaimgprop->Get();

        // does the size of cuda-image have to match the mitk-image where it's attached to?
        // i think it does: it is supposed to be the same data living in cuda.
        VIVID_CHECK(lwci.GetWidth()  == imgdata->GetDimension(0));
        VIVID_CHECK(lwci.GetHeight() == imgdata->GetDimension(1));

        PrepareBackgroundActor(&lwci, imgdata->GetGeometry(), node);
        result = true;
      }
      else
#endif
      {
        PrepareBackgroundActor(imgdata.GetPointer(), imgdata->GetGeometry(), node);
        result = true;
      }

      m_BackgroundWidth  = imgdata->GetDimension(0);
      m_BackgroundHeight = imgdata->GetDimension(1);
    }
    else
    {
#ifdef _USE_CUDA
      niftk::CUDAImage::Pointer    cudaimgdata = dynamic_cast<niftk::CUDAImage*>(basedata.GetPointer());
      if (cudaimgdata.IsNotNull())
      {
        niftk::LightweightCUDAImage    lwci = cudaimgdata->GetLightweightCUDAImage();
        PrepareBackgroundActor(&lwci, cudaimgdata->GetGeometry(), node);
        result = true;

        m_BackgroundWidth  = lwci.GetWidth();
        m_BackgroundHeight = lwci.GetHeight();
      }
      // no else here
#endif
    }

    // UpdateDataNode() depends on m_BackgroundNode.
    m_BackgroundNode = node;
    UpdateDataNode(node);
  }

  UpdateViewportAndCameraAfterResize();

  // now that the camera may have changed, fit-view-to-scene again.
  //if (m_CameraNode.IsNull())
  //{
  //  m_Trackball->setEnabled( true );
  //  m_Trackball->adjustView(m_SceneManager.get(), vl::vec3(0, 0, 1), vl::vec3(0, 1, 0), 1.0f);
  //}

  return result;
}

//-----------------------------------------------------------------------------

#ifdef _USE_CUDA
void VLSceneView::cudaSwapBuffers()
{
  if (m_CUDAInteropPimpl)
  {
    cudaError_t          err         = cudaSuccess;
    niftk::CUDAManager*  cudamanager = niftk::CUDAManager::GetInstance();
    cudaStream_t         mystream    = cudamanager->GetStream(m_CUDAInteropPimpl->m_NodeName);
    niftk::WriteAccessor outputWA    = cudamanager->RequestOutputImage(QWidget::width(), QWidget::height(), 4);
    cudaArray_t          fboarr      = m_CUDAInteropPimpl->m_FBOAdaptor->Map(mystream);

    // side note: cuda-arrays are always measured in bytes, never in pixels.
    err = cudaMemcpy2DFromArrayAsync(outputWA.m_DevicePointer, outputWA.m_BytePitch, fboarr, 0, 0, outputWA.m_PixelWidth * 4, outputWA.m_PixelHeight, cudaMemcpyDeviceToDevice, mystream);
    // not sure what to do if it fails. do not throw an exception, that's for sure.
    if (err != cudaSuccess)
    {
      VIVID_CHECK(false);
    }

    // the opengl-interop side is done, renderer can continue from now on.
    m_CUDAInteropPimpl->m_FBOAdaptor->Unmap(mystream);

    // need to flip the image! ogl is left-bottom, but everywhere else is left-top origin!
    niftk::WriteAccessor  flippedWA   = cudamanager->RequestOutputImage(outputWA.m_PixelWidth, outputWA.m_PixelHeight, 4);
    // FIXME: instead of explicitly flipping we could bind the fboarr to a texture, and do a single write out.
    niftk::FlipImageLauncher(outputWA, flippedWA, mystream);

    niftk::LightweightCUDAImage lwciFlipped = cudamanager->Finalise(flippedWA, mystream);
    // Finalise() needs to come before Autorelease(), for performance reasons.
    cudamanager->Autorelease(outputWA, mystream);

    bool    isNewNode = false;
    mitk::DataNode::Pointer node = m_CUDAInteropPimpl->m_DataStorage->GetNamedNode(m_CUDAInteropPimpl->m_NodeName);
    if (node.IsNull())
    {
      isNewNode = true;
      node = mitk::DataNode::New();
      node->SetName(m_CUDAInteropPimpl->m_NodeName);
      node->SetVisibility(false);
      //node->SetBoolProperty("helper object", true);
    }
    niftk::CUDAImage::Pointer  img = dynamic_cast<niftk::CUDAImage*>(node->GetData());
    if (img.IsNull())
      img = niftk::CUDAImage::New();
    img->SetLightweightCUDAImage(lwciFlipped);
    node->SetData(img);
    if (isNewNode)
      m_CUDAInteropPimpl->m_DataStorage->Add(node);
    else
      node->Modified();
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::FreeCUDAInteropTextures()
{
  openglContext()->makeCurrent();

  for (std::map<mitk::DataNode::ConstPointer, TextureDataPOD>::iterator i = m_NodeToTextureMap.begin(); i != m_NodeToTextureMap.end(); )
  {
    if (i->second.m_CUDARes != 0)
    {
      cudaError_t err = cudaGraphicsUnregisterResource(i->second.m_CUDARes);
      if (err != cudaSuccess)
      {
        MITK_WARN << "Failed to unregister VL texture from CUDA";
      }
    }

    i = m_NodeToTextureMap.erase(i);
  }

  // if no cuda is available then this is most likely a nullptr.
  // and if not a nullptr then it's only a dummy. so unconditionally delete it.
  delete m_CUDAInteropPimpl;
  m_CUDAInteropPimpl = 0;

}

//-----------------------------------------------------------------------------

void VLSceneView::EnableFBOCopyToDataStorageViaCUDA(bool enable, mitk::DataStorage* datastorage, const std::string& nodename)
{
  openglContext()->makeCurrent();

  if (enable)
  {
    if (datastorage == 0)
      throw std::runtime_error("Need data storage object");

    delete m_CUDAInteropPimpl;
    m_CUDAInteropPimpl = new CUDAInterop;
    m_CUDAInteropPimpl->m_FBOAdaptor = 0;
    m_CUDAInteropPimpl->m_DataStorage = datastorage;
    m_CUDAInteropPimpl->m_NodeName = nodename;
    if (m_CUDAInteropPimpl->m_NodeName.empty())
    {
      std::ostringstream    n;
      n << "0x" << std::hex << (void*) this;
      m_CUDAInteropPimpl->m_NodeName = n.str();
    }
  }
  else
  {
    delete m_CUDAInteropPimpl;
    m_CUDAInteropPimpl = 0;
  }
}

//-----------------------------------------------------------------------------

void VLSceneView::PrepareBackgroundActor(const niftk::LightweightCUDAImage* lwci, const mitk::BaseGeometry* geom, const mitk::DataNode::ConstPointer node)
{
  openglContext()->makeCurrent();

  VIVID_CHECK(lwci != 0);

  vl::mat4  mat;
  mat = mat.setIdentity();
  ref<vl::Transform> tr     = new vl::Transform();
  tr->setLocalMatrix(mat);


  // essentially copied from vl::makeGrid()
  ref<vl::Geometry>         vlquad = new vl::Geometry;

  ref<vl::ArrayFloat3> vert3 = new vl::ArrayFloat3;
  vert3->resize(4);
  vlquad->setVertexArray(vert3.get());

  ref<vl::ArrayFloat2> text2 = new vl::ArrayFloat2;
  text2->resize(4);
  vlquad->setTexCoordArray(0, text2.get());

  //  0---3
  //  |   |
  //  1---2
  vert3->at(0).x() = -1; vert3->at(0).y() =  1; vert3->at(0).z() = 0;  text2->at(0).s() = 0; text2->at(0).t() = 0;
  vert3->at(1).x() = -1; vert3->at(1).y() = -1; vert3->at(1).z() = 0;  text2->at(1).s() = 0; text2->at(1).t() = 1;
  vert3->at(2).x() =  1; vert3->at(2).y() = -1; vert3->at(2).z() = 0;  text2->at(2).s() = 1; text2->at(2).t() = 1;
  vert3->at(3).x() =  1; vert3->at(3).y() =  1; vert3->at(3).z() = 0;  text2->at(3).s() = 1; text2->at(3).t() = 0;


  ref<vl::DrawElementsUInt> polys = new vl::DrawElementsUInt(vl::PT_QUADS);
  polys->indexBuffer()->resize(4);
  polys->indexBuffer()->at(0) = 0;
  polys->indexBuffer()->at(1) = 1;
  polys->indexBuffer()->at(2) = 2;
  polys->indexBuffer()->at(3) = 3;
  vlquad->drawCalls().push_back(polys.get());


  ref<vl::Effect>    fx = new vl::Effect;
  fx->shader()->disable(vl::EN_LIGHTING);
  // UpdateDataNode() takes care of assigning colour etc.

  ref<vl::Actor> actor = m_VividRendering->sceneManager()->tree()->addActor(vlquad.get(), fx.get(), tr.get());
  actor->setEnableMask( vl::VividRenderer::DefaultEnableMask );


  std::string   objName = actor->objectName() + "_background";
  actor->setObjectName(objName.c_str());

  m_NodeActorMap[node] = actor;
  m_NodeToTextureMap[node] = TextureDataPOD();
}

//-----------------------------------------------------------------------------

#endif