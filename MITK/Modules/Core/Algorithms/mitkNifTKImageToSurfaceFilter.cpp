/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkNifTKImageToSurfaceFilter.h"

#include "mitkException.h"
#include <vtkImageData.h>
#include <vtkDecimatePro.h>
#include <vtkImageChangeInformation.h>
#include <vtkLinearTransform.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkQuadricDecimation.h>
#include <vtkPolyDataNormals.h>
#include <vtkCleanPolyData.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkImageMedian3D.h>
#include <vtkWindowedSincPolyDataFilter.h>
#include <vtkPolyDataNormals.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkSTLWriter.h>
#include <vtkPolyDataConnectivityFilter.h>

#include <mitkImageCast.h>
#include <mitkProgressBar.h>
#include <mitkGlobalInteraction.h>
#include "mitkNifTKMeshSmoother.h"

mitk::NifTKImageToSurfaceFilter::NifTKImageToSurfaceFilter():
  m_Threshold(100.0f),
  m_SurfaceExtractionType(StandardExtractor),
  m_InputSmoothingType(NoInputSmoothing),
  m_PerformInputSmoothing(false),
  m_InputSmoothingIterations(1),
  m_InputSmoothingRadius(0.5),
  m_SurfaceSmoothingType(NoSurfaceSmoothing),
  m_PerformSurfaceSmoothing(false),
  m_SurfaceSmoothingIterations(1),
  m_SurfaceSmoothingRadius(0.5),
  m_SurfaceDecimationType(NoDecimation),
  m_PerformSurfaceDecimation(false),
  m_TargetReduction(0.1),
  m_PerformSurfaceCleaning(true),
  m_SurfaceCleaningThreshold(1000),
  m_VTKNormalCompute(true),
  m_FlipNormals(false),
  m_SamplingRatio(1.0)
{
}

mitk::NifTKImageToSurfaceFilter::~NifTKImageToSurfaceFilter()
{
}

void mitk::NifTKImageToSurfaceFilter::VTKSurfaceExtraction(mitk::Image *inputImage, vtkSmartPointer<vtkPolyData> vtkSurface)
{
  vtkImageData *vtkimage = inputImage->GetVtkImageData();
  vtkImageChangeInformation *indexCoordinatesImageFilter = vtkImageChangeInformation::New();
  indexCoordinatesImageFilter->SetInputData(vtkimage);
  indexCoordinatesImageFilter->SetOutputOrigin(0.0,0.0,0.0);
  indexCoordinatesImageFilter->Update();
  ProgressBar::GetInstance()->Progress();

  //MarchingCube -->create Surface
  vtkMarchingCubes *vtkMC = vtkMarchingCubes::New();
  vtkMC->ComputeScalarsOff();
  vtkMC->SetInputData(indexCoordinatesImageFilter->GetOutput());
  vtkMC->SetValue(0, m_Threshold);

  vtkMC->Update();

  vtkSurface->DeepCopy(vtkMC->GetOutput());

  ProgressBar::GetInstance()->Progress();

  if(vtkSurface->GetNumberOfPoints() > 0)
  {
    mitk::Vector3D spacing = inputImage->GetGeometry()->GetSpacing();

    vtkPoints * points = vtkSurface->GetPoints();
    vtkMatrix4x4 *vtkmatrix = vtkMatrix4x4::New();
    inputImage->GetGeometry()->GetVtkTransform()->GetMatrix(vtkmatrix);
    double (*matrix)[4] = vtkmatrix->Element;

    unsigned int i,j;
    for(i=0;i<3;++i)
      for(j=0;j<3;++j)
        matrix[i][j]/=spacing[j];

    unsigned int n = points->GetNumberOfPoints();
    double point[3];

    for (i = 0; i < n; i++)
    {
      points->GetPoint(i, point);
      mitkVtkLinearTransformPoint(matrix,point,point);
      points->SetPoint(i, point);
    }
    vtkmatrix->Delete();
  }

  indexCoordinatesImageFilter->Delete();
  vtkMC->Delete();
}

void mitk::NifTKImageToSurfaceFilter::CMC33SurfaceExtraction(mitk::Image *inputImage, mitk::MeshData * meshData)
{
  itk::Image<float, 3>::Pointer inputItkImage;
  mitk::CastToItkImage(inputImage, inputItkImage);

  mitk::CMC33 * cmcExtractor = new mitk::CMC33(inputImage->GetDimension(0), inputImage->GetDimension(1), inputImage->GetDimension(2));
  cmcExtractor->set_input_data(inputItkImage->GetBufferPointer());
  cmcExtractor->set_output_data(meshData);
  cmcExtractor->enable_normal_computing(false);

  cmcExtractor->init_all();
  cmcExtractor->run(m_Threshold);

  delete cmcExtractor;
}

void mitk::NifTKImageToSurfaceFilter::MeshSmoothing(MeshData * mesh)
{
  //clock_t time0 = clock();
  
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Smooth the surface mesh and fix normals
  mitk::MeshSmoother * meshSmoother = new mitk::MeshSmoother();
  meshSmoother->InitWithExternalData(mesh);
  
  //clock_t time1 = clock();
  
  float lambda = 0.5f;
  float mu = -0.53f;

  switch (m_SurfaceSmoothingType)
  {
    case InverseEdgeLengthSmooth:
      meshSmoother->SetSmoothingMethod(1);
      break;
    case CurvatureNormalSmooth:
      meshSmoother->SetSmoothingMethod(2);
      break;
    case TaubinSmoothing:
    default:
      meshSmoother->SetSmoothingMethod(0);
      break;
  }
  meshSmoother->TaubinSmooth(lambda, mu, m_SurfaceSmoothingIterations);
  //clock_t time2 = clock();
  meshSmoother->SetFlipNormals(m_FlipNormals);
  meshSmoother->GenerateVertexAndTriangleNormals();
  meshSmoother->RescaleMesh(1.0/m_SamplingRatio);
  //meshSmoother->ReOrientFaces();

  delete meshSmoother;

  //clock_t time3 = clock();
  //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
  //printf("Mesh smoothing took %lf secs.\n", (double) (time3 - time0) / CLOCKS_PER_SEC);
  //printf("  Initialization took %lf secs.\n", (double) (time1 - time0) / CLOCKS_PER_SEC);
  //printf("  Smoothing took %lf secs.\n", (double) (time2 - time1) / CLOCKS_PER_SEC);
  //printf("  Generating normals took %lf secs.\n", (double) (time3 - time2) / CLOCKS_PER_SEC);
  //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
}

void mitk::NifTKImageToSurfaceFilter::ComputeSmoothNormals(MeshData * meshData)
{
  //clock_t time = clock();

   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Smooth the surface mesh and fix normals
  mitk::MeshSmoother * meshSmoother = new mitk::MeshSmoother();
  meshSmoother->InitWithExternalData(meshData);

  //clock_t time2 = clock();
  meshSmoother->SetFlipNormals(m_FlipNormals);
  meshSmoother->GenerateVertexAndTriangleNormals();
  meshSmoother->RescaleMesh(1.0/m_SamplingRatio);
  //clock_t time4 = clock();

  delete meshSmoother;
  //printf("Normal computation - initialization took %lf secs.\n", (double) (time2 - time) / CLOCKS_PER_SEC);
  //printf("Normal computation - computation took %lf secs.\n", (double) (time3 - time2) / CLOCKS_PER_SEC);
  //printf("Normal computation - getting output took %lf secs.\n", (double) (time4 - time3) / CLOCKS_PER_SEC);
}

void mitk::NifTKImageToSurfaceFilter::EditSurface(
  MeshData * meshData,
  bool recomputeNormals,
  bool flipNormals,
  bool reorientFaces,
  bool fixCracks)
{
  //clock_t time = clock();

   //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Smooth the surface mesh and fix normals
  mitk::MeshSmoother * meshSmoother = new mitk::MeshSmoother();
  meshSmoother->InitWithExternalData(meshData);

  //clock_t time2 = clock();

  if (fixCracks)
    meshSmoother->FixCracks();

  if (reorientFaces)
    meshSmoother->ReOrientFaces();

  if (recomputeNormals)
  {
    meshSmoother->SetFlipNormals(flipNormals);
    meshSmoother->GenerateVertexAndTriangleNormals();
  }

  //clock_t time3 = clock();

  delete meshSmoother;
  //printf("Normal computation - initialization took %lf secs.\n", (double) (time2 - time) / CLOCKS_PER_SEC);
  //printf("Normal computation - computation took %lf secs.\n", (double) (time3 - time2) / CLOCKS_PER_SEC);
}


void mitk::NifTKImageToSurfaceFilter::SurfaceSmoothingVTK(vtkSmartPointer<vtkPolyData> vtkSurface)
{
  switch (m_SurfaceSmoothingType)
  {
    case WindowedSincSmoothing:
    {
      vtkWindowedSincPolyDataFilter * smoother = vtkWindowedSincPolyDataFilter::New();
      smoother->SetInputData(vtkSurface);
      smoother->SetNumberOfIterations(m_SurfaceSmoothingIterations);
      //smoother->SetNumberOfIterations(10);
      smoother->FeatureEdgeSmoothingOn();
      smoother->BoundarySmoothingOn();
      smoother->FeatureEdgeSmoothingOn();
      smoother->SetFeatureAngle(120.0);
      smoother->SetPassBand(0.001);
      smoother->NonManifoldSmoothingOff();
      smoother->NormalizeCoordinatesOn();

      smoother->Update();
      vtkSurface->DeepCopy(smoother->GetOutput());

      smoother->Delete();
    }
    break;

    case StandardVTKSmoothing:
    {
      vtkSmoothPolyDataFilter *smoother = vtkSmoothPolyDataFilter::New();
      smoother->SetInputData(vtkSurface);
      smoother->SetNumberOfIterations(m_SurfaceSmoothingIterations);
      smoother->SetFeatureAngle( 60 );
      smoother->FeatureEdgeSmoothingOff();
      smoother->BoundarySmoothingOff();
      smoother->SetConvergence( 0 );

      smoother->Update();
      vtkSurface->DeepCopy(smoother->GetOutput());

      smoother->Delete();
    }
    break;

    default:
      break;
  }
}

void mitk::NifTKImageToSurfaceFilter::CreateSurface(mitk::Image *inputImage, mitk::Surface *surface)
{
  // Create working image / surface
  mitk::Image::Pointer workingImage = 0;
  vtkSmartPointer<vtkPolyData> vtkSurface = vtkSmartPointer<vtkPolyData>::New();

  clock_t time0 = clock();

  if (m_PerformInputSmoothing)
  {
    vtkSmartPointer<vtkImageData> vtkImage = inputImage->GetVtkImageData();

    vtkSmartPointer<vtkImageGaussianSmooth> gaussian = 0;
    vtkSmartPointer<vtkImageMedian3D>       median   = 0;
    int radius = 0;

    switch (m_InputSmoothingType)
    {
    case GaussianSmoothing:
      MITK_INFO << "Performing Gaussian smoothing on the input data: " <<m_InputSmoothingRadius <<"\n";
      gaussian = vtkImageGaussianSmooth::New();
      gaussian->SetInputData(vtkImage);
      //gaussian->SetDimensionality(3);
      //gaussian->SetRadiusFactor(0.49);
      gaussian->SetStandardDeviation(m_InputSmoothingRadius);
      gaussian->Update();

      vtkImage = gaussian->GetOutput();
      break;
    case MedianSmoothing:
      median = vtkImageMedian3D::New();
      median->SetInputData(vtkImage);
      radius = static_cast<int>(floor(m_InputSmoothingRadius+0.5));
      MITK_INFO <<"Performing Median smoothing on the input data: " <<radius <<"\n";
      median->SetKernelSize(radius, radius, radius);//Std: 3x3x3
      median->Update();

      vtkImage = median->GetOutput();
      break;
    case NoInputSmoothing:
      break;
    }

    workingImage = mitk::Image::New();
    workingImage->Initialize(vtkImage.GetPointer());
    workingImage->SetVolume(vtkImage->GetScalarPointer());
    workingImage->SetGeometry(inputImage->GetGeometry());
  }
  else
  {
    workingImage = inputImage->Clone();
  }

  clock_t time1 = clock();
  clock_t time1_5 = 0;
  clock_t time2 = 0;
  clock_t time3 = 0;
  clock_t time4 = 0;


  if (!workingImage)
  {
    MITK_INFO << "SurfaceExtractorView::createSurface(): No reference image. Should not arrive here.";
    return;
  }

  ProgressBar::GetInstance()->Progress();

  // By default we want VTK to compute the normals
  m_VTKNormalCompute = true;


  // Here we decide which extraction method to use
  switch (m_SurfaceExtractionType)
  {
    case StandardExtractor: 
    {
      // Extract surface VTK style
       VTKSurfaceExtraction(workingImage, vtkSurface);
       time2 = time3 = clock();
    }
    break;

    case EnhancedCPUExtractor:
    {
      // Create a new instance of meshdata 
      mitk::MeshData * meshData = new MeshData();

      std::vector<BasicVertex> vertices; 
      std::vector<BasicTriangle> triangles;
      int numOfVertices = 0;
      int numOfTriangles = 0;
      
      // In this case we do not want VTK to compute the normals
      m_VTKNormalCompute = false;

      if (m_SamplingRatio != 1.0)
      {
        if (m_SamplingRatio < 0.125)
          m_SamplingRatio = 0.125;
        else if (m_SamplingRatio > 2.0)
          m_SamplingRatio = 2.0;

        mitk::mitkBasicImageProcessor * imgProc = new mitk::mitkBasicImageProcessor();
        workingImage = imgProc->ProcessImage(workingImage, mitk::mitkBasicImageProcessor::DOWNSAMPLING, 0, 0, 1.0/m_SamplingRatio, 0, 0);
        time1_5 = clock();
        delete imgProc;
      }

      // Run the Corrected Marching Cubes33 algorithm to extract the surface 
      // and get the resulting vertices and triangles
      CMC33SurfaceExtraction(workingImage, meshData);
      time2 = clock();
      ProgressBar::GetInstance()->Progress();

      // Perform the custom smoothing on the raw mesh
      if (m_PerformSurfaceSmoothing && (m_SurfaceSmoothingType == TaubinSmoothing || m_SurfaceSmoothingType == CurvatureNormalSmooth || m_SurfaceSmoothingType == InverseEdgeLengthSmooth))
        MeshSmoothing(meshData);
      else if (!m_PerformSurfaceDecimation)
        ComputeSmoothNormals(meshData);

      time3 = clock();
      ProgressBar::GetInstance()->Progress();

      // Let's build a VTK surface
      vtkSurface = BuildVTKPolyData(meshData);
      ProgressBar::GetInstance()->Progress();

      triangles.clear();
      vertices.clear();
      delete meshData;

      ProgressBar::GetInstance()->Progress();
    }
    break;

    case GPUExtractor:
    break;

    default:
    break;
  }

  time4 = clock();

  if (m_VTKNormalCompute)
  {
    // Instanciating the VTK normal generator. Even if we don't create the normals we'll still need to oreient them
    vtkSmartPointer<vtkPolyDataNormals> normalGen = vtkSmartPointer<vtkPolyDataNormals>::New();
    normalGen->SetInputData(vtkSurface);
    normalGen->AutoOrientNormalsOn();
    normalGen->ComputeCellNormalsOn();
    normalGen->ComputePointNormalsOn();
    normalGen->Update();

    vtkSurface->DeepCopy(normalGen->GetOutput());
  }

  clock_t time5 = clock();

  // Smooth the surface mesh a'la VTK
  if (m_PerformSurfaceSmoothing && (m_SurfaceSmoothingType == StandardVTKSmoothing))
    SurfaceSmoothingVTK(vtkSurface);

  clock_t time6 = clock();

  ProgressBar::GetInstance()->Progress();

   // Decimate the surface mesh
  if (m_PerformSurfaceDecimation)
  {

    //decimate = to reduce number of polygons
    if (m_SurfaceDecimationType == DecimatePro)
    {
      vtkSmartPointer<vtkDecimatePro> decimate = vtkDecimatePro::New();
      decimate->SetInputData(vtkSurface);

      decimate->SetTargetReduction(m_TargetReduction);
      decimate->SplittingOff();
      decimate->PreserveTopologyOn();
      decimate->BoundaryVertexDeletionOff();

      decimate->SetErrorIsAbsolute(5);
      decimate->SetFeatureAngle(30);
      decimate->SetDegree(10);
      decimate->SetMaximumError(0.002);
      decimate->Update();

      vtkSurface = decimate->GetOutput();
    }
    else if (m_SurfaceDecimationType == QuadricVTK)
    {
      vtkSmartPointer<vtkQuadricDecimation> decimate = vtkQuadricDecimation::New();

      decimate->SetTargetReduction(m_TargetReduction);
      decimate->SetInputData(vtkSurface);

      decimate->Update();
      vtkSurface = decimate->GetOutput();
    }
  }

  ProgressBar::GetInstance()->Progress();

  clock_t time7 = clock();

  if (m_PerformSurfaceCleaning)
  {

    vtkSmartPointer<vtkPolyDataConnectivityFilter> connectivityFilter = 
      vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
    connectivityFilter->SetInputData(vtkSurface);
    connectivityFilter->SetExtractionModeToAllRegions();
    connectivityFilter->ScalarConnectivityOn();
    connectivityFilter->Update();

    int numOfRegions = connectivityFilter->GetNumberOfExtractedRegions();
    vtkSmartPointer<vtkIdTypeArray> sizes = connectivityFilter->GetRegionSizes();
    vtkSmartPointer<vtkPolyData> regions = connectivityFilter->GetOutput();

    connectivityFilter->SetExtractionModeToSpecifiedRegions(); 

    //MITK_INFO <<"Num of extracted regions: " <<numOfRegions;

    int id = 0;
    vtkIdTypeArray::Iterator sizeIter;
    for (sizeIter = sizes->Begin(); sizeIter < sizes->End(); sizeIter++)
    {
      if (*sizeIter > m_SurfaceCleaningThreshold)
      {
        //MITK_INFO <<"Current region size: " <<*sizeIter;
        connectivityFilter->AddSpecifiedRegion(id); //select the region to extract here
      }
      id++;
    }

    connectivityFilter->Update();
  
    vtkSurface = connectivityFilter->GetOutput();

    // Clean the results - not sure if this does any good
    //  vtkSmartPointer<vtkCleanPolyData> cleanPolyDataFilter = vtkSmartPointer<vtkCleanPolyData>::New();
    //  cleanPolyDataFilter->SetInput(vtkSurface);
    //  cleanPolyDataFilter->PieceInvariantOff();
    //  cleanPolyDataFilter->ConvertLinesToPointsOff();
    //  cleanPolyDataFilter->ConvertPolysToLinesOff();
    //  cleanPolyDataFilter->ConvertStripsToPolysOff();
    //  cleanPolyDataFilter->PointMergingOff();
    //  cleanPolyDataFilter->Update();

    //  // Get output
    //  vtkSurface = cleanPolyDataFilter->GetOutput();

  }

  clock_t time8 = clock();

  ProgressBar::GetInstance()->Progress();

  // Set the output with geometry
  surface->SetVtkPolyData(vtkSurface);

  if (m_SurfaceExtractionType != StandardExtractor)
    surface->SetGeometry(inputImage->GetGeometry());

  clock_t time9 = clock();


  printf("________________________________________________________\n");
  printf("Total time to create surface: %lf secs.\n", (double) (time9 - time0) / CLOCKS_PER_SEC);
  printf("   Input smoothing & working copy creation took %lf secs.\n", (double) (time1 - time0) / CLOCKS_PER_SEC);
  printf("   Downsampling took %lf secs.\n", (double) (time1_5 - time1) / CLOCKS_PER_SEC);
  printf("   Surface extraction took %lf secs.\n", (double) (time2 - time1_5) / CLOCKS_PER_SEC);
  printf("   Surface smoothing or Normal computation took %lf secs.\n", (double) (time3 - time2) / CLOCKS_PER_SEC);
  printf("   Building VTK PolyData structure took %lf secs.\n", (double) (time4 - time3) / CLOCKS_PER_SEC);
  printf("   VTK Normal computation took %lf secs.\n", (double) (time5 - time4) / CLOCKS_PER_SEC);
  printf("   VTK Surface Smoothing took %lf secs.\n", (double) (time6 - time5) / CLOCKS_PER_SEC);
  printf("   VTK Surface Decimation took %lf secs.\n", (double) (time7 - time6) / CLOCKS_PER_SEC);
  printf("   VTK Surface Cleaning took %lf secs.\n", (double) (time8 - time7) / CLOCKS_PER_SEC);
  printf("   Copying surface to output took %lf secs.\n", (double) (time9 - time8) / CLOCKS_PER_SEC);
  printf("--------------------------------------------------------\n");

  surface->Update();
}

vtkSmartPointer<vtkPolyData> mitk::NifTKImageToSurfaceFilter::BuildVTKPolyData(MeshData * meshData)
{
  vtkSmartPointer<vtkPolyData>  polyData   = vtkSmartPointer<vtkPolyData>::New();
  vtkSmartPointer<vtkPoints>    pointArray = vtkSmartPointer<vtkPoints>::New();
  vtkSmartPointer<vtkCellArray> vertArray  = vtkSmartPointer<vtkCellArray>::New();

  // Copy vertices
  float point[3];
  vtkIdType pid[1];

  int numOfVertices  = meshData->m_Vertices.size();
  int numOfTriangles = meshData->m_Triangles.size();

  for (unsigned int i = 0; i < numOfVertices; i++)
  {
    point[0] = meshData->m_Vertices[i].GetCoordX();
    point[1] = meshData->m_Vertices[i].GetCoordY();
    point[2] = meshData->m_Vertices[i].GetCoordZ();
    pid[0] = pointArray->InsertNextPoint(point);
    vertArray->InsertNextCell ( 1,pid );
  }
  
  // Add the points to a polydata
  if (polyData == 0)
    polyData = vtkSmartPointer<vtkPolyData>::New();
  
  polyData->Reset();

  polyData->SetPoints(pointArray);
  //polyData->SetVerts(vertArray);
  polyData->GetCellData()->Update();
  polyData->GetPointData()->Update();
 
  vtkSmartPointer<vtkCellArray> cells = vtkSmartPointer<vtkCellArray>::New();
  cells->Initialize();

  for (unsigned int i = 0; i < numOfTriangles; i++)
  {
    vtkIdType pid[3];

    pid[0] = meshData->m_Triangles[i].GetVert1Index();
    pid[1] = meshData->m_Triangles[i].GetVert2Index();
    pid[2] = meshData->m_Triangles[i].GetVert3Index();

    //create a vertex cell on the point that was just added.
    cells->InsertNextCell (3, pid);
  }

  cells->Modified();

  // Set the cells to polydata
  polyData->SetPolys(cells);

  if (!m_VTKNormalCompute)
  {
    vtkSmartPointer<vtkFloatArray> pointNormalsArray = vtkSmartPointer<vtkFloatArray>::New();
    pointNormalsArray->SetNumberOfComponents(3); 

    vtkSmartPointer<vtkFloatArray> triangleNormalsArray = vtkSmartPointer<vtkFloatArray>::New();
    triangleNormalsArray->SetNumberOfComponents(3); 

    float normal[3];

    for (unsigned int i = 0; i < numOfVertices; i++)
    {
      normal[0] = meshData->m_Vertices[i].GetNormalX();
      normal[1] = meshData->m_Vertices[i].GetNormalY();
      normal[2] = meshData->m_Vertices[i].GetNormalZ();
      pointNormalsArray->InsertNextTuple(normal);
    }

    // Add the normals to the points in the polydata
    polyData->GetPointData()->SetNormals(pointNormalsArray);
    polyData->GetCellData()->Update();
    polyData->GetPointData()->Update();

    for (unsigned int i = 0; i < numOfTriangles; i++)
    {
      normal[0] = meshData->m_Triangles[i].GetTriNormalX();
      normal[1] = meshData->m_Triangles[i].GetTriNormalY();
      normal[2] = meshData->m_Triangles[i].GetTriNormalZ();
      triangleNormalsArray->InsertNextTuple(normal);
    }

    // Add the normals to the cells in the polydata
    polyData->GetCellData()->SetNormals(triangleNormalsArray);
    polyData->GetCellData()->Update();
    polyData->GetPointData()->Update();
  }
  polyData->Squeeze();
  polyData->Modified();

  try
  {
    polyData->BuildCells();
  }
  catch(const mitk::Exception& e)
  {
    MITK_ERROR << "Caught exception while building links and cells of polydata: " << e.what();
    return 0;
  }

  // Call update
  polyData->GetCellData()->Update();
  polyData->GetPointData()->Update();

  return polyData;
}

void mitk::NifTKImageToSurfaceFilter::GenerateData()
{
  mitk::Surface *surface = this->GetOutput();
  mitk::Image * inputImage = (mitk::Image*)GetInput();
  if (inputImage == NULL || !inputImage->IsInitialized())
    mitkThrow() << "No input image set, please set an valid input image!";

  mitk::Image::RegionType outputRegion = inputImage->GetRequestedRegion();

  int tstart=outputRegion.GetIndex(3);
  int tmax=tstart+outputRegion.GetSize(3);

  if ((tmax-tstart) > 0)
  {
    ProgressBar::GetInstance()->AddStepsToDo( 4 * (tmax - tstart)  );
  }

  //mitk::Stepper* timeStepper = mitk::RenderingManager::GetInstance()->GetTimeNavigationController()->GetTime();
  //unsigned int timeStep = timeStepper->GetPos();

  // We're going to generate the surface for all the time points
  for (int t=tstart; t < tmax; ++t)
  {
    vtkSmartPointer<vtkImageData> vtkImage = inputImage->GetVtkImageData(t);
    mitk::Image::Pointer workingImage = mitk::Image::New();
    workingImage->Initialize(vtkImage.GetPointer());
    workingImage->SetVolume(vtkImage->GetScalarPointer());
    workingImage->SetGeometry(inputImage->GetGeometry());

    CreateSurface(workingImage, surface);

    ProgressBar::GetInstance()->Progress();
  }
}

void mitk::NifTKImageToSurfaceFilter::SetInput(const mitk::Image *image)
{
  // Process object is not const-correct so the const_cast is required here
  this->ProcessObject::SetNthInput(0, const_cast< mitk::Image * >( image ) );
}

const mitk::Image *mitk::NifTKImageToSurfaceFilter::GetInput(void)
{
  if (this->GetNumberOfInputs() < 1)
  {
    return 0;
  }

  return static_cast<const mitk::Image * >
    ( this->ProcessObject::GetInput(0) );
}

void mitk::NifTKImageToSurfaceFilter::GenerateOutputInformation()
{
  mitk::Image::ConstPointer inputImage  =(mitk::Image*) this->GetInput();
  mitk::Surface::Pointer output = this->GetOutput();

  itkDebugMacro(<<"GenerateOutputInformation()");

  if(inputImage.IsNull()) return;
}
