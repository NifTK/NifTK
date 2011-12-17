/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkMeanSquaresMeshToMeshMetricTest1.cxx,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifdef _MSC_VER
#pragma warning ( disable : 4786 )
#endif

#include "itkMeanSquaresMeshToMeshMetric.h"
#include "itkMeshToMeshRegistrationMethod.h"
#include "itkLinearInterpolateMeshFunction.h"
#include "itkVersorTransformOptimizer.h"
#include "itkVersorTransform.h"
#include "itkQuadEdgeMesh.h"

#include "itkCommand.h"
#include "itkVTKPolyDataReader.h"

#include "itkResampleQuadEdgeMeshFilter.h"
#include "itkQuadEdgeMeshTraits.h"
#include "itkQuadEdgeMeshScalarDataVTKPolyDataWriter.h"
#include "itkQuadEdgeMeshVectorDataVTKPolyDataWriter.h"
#include "itkQuadEdgeMeshSphericalDiffeomorphicDemonsFilter.h"
#include "itkDeformationFieldFromTransformMeshFilter.h"
#include "itkResampleDestinationPointsQuadEdgeMeshFilter.h"
#include "itkIdentityTransform.h"


class CommandIterationUpdate : public itk::Command 
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;

  itkNewMacro( Self );

protected:
  CommandIterationUpdate()
   {
   iterationCounter = 0;
   }

public:
  typedef itk::VersorTransformOptimizer  OptimizerType;
  typedef   const OptimizerType   *      OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
    {
    Execute( (const itk::Object *)caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event)
    {
    OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >( object );

    if( ! itk::IterationEvent().CheckEvent( &event ) )
      {
      return;
      }

    std::cout << " Iteration " << ++iterationCounter;
    std::cout << "  Value " << optimizer->GetValue() << "   ";
    std::cout << "  Position " << optimizer->GetCurrentPosition() << std::endl ; 
    }
private:

  unsigned int iterationCounter; 

};




int main( int argc, char * argv [] )
{

  if( argc < 13 )
    {
    std::cerr << "Missing arguments" << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << std::endl;
    std::cerr << "inputFixedMeshRes1 inputMovingMeshRes1 ";
    std::cerr << "outputResampledMeshRes1 ";
    std::cerr << "inputFixedMeshRes2 inputMovingMeshRes2 ";
    std::cerr << "outputResampledMeshRes2 ";
    std::cerr << "inputFixedMeshRes3 inputMovingMeshRes3 ";
    std::cerr << "outputResampledMeshRes3 ";
    std::cerr << "inputFixedMeshRes4 inputMovingMeshRes4 ";
    std::cerr << "outputResampledMeshRes4 ";
    std::cerr << std::endl;
    return EXIT_FAILURE;
    }

  typedef float      MeshPixelType;
  const unsigned int Dimension = 3;

  typedef itk::QuadEdgeMesh< MeshPixelType, Dimension >   FixedMeshType;
  typedef itk::QuadEdgeMesh< MeshPixelType, Dimension >   MovingMeshType;

  typedef itk::VTKPolyDataReader< FixedMeshType >     FixedReaderType;
  typedef itk::VTKPolyDataReader< MovingMeshType >    MovingReaderType;

  FixedReaderType::Pointer fixedMeshReader1 = FixedReaderType::New();
  fixedMeshReader1->SetFileName( argv[1] );

  MovingReaderType::Pointer movingMeshReader1 = MovingReaderType::New();
  movingMeshReader1->SetFileName( argv[2] );

  try
    {
    fixedMeshReader1->Update( );
    movingMeshReader1->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }

  FixedMeshType::ConstPointer  meshFixed  = fixedMeshReader1->GetOutput();
  MovingMeshType::ConstPointer meshMoving = movingMeshReader1->GetOutput();

  typedef itk::MeshToMeshRegistrationMethod< 
                                    FixedMeshType, 
                                    MovingMeshType >    RegistrationType;

  RegistrationType::Pointer   registration  = RegistrationType::New();

  typedef itk::MeanSquaresMeshToMeshMetric< FixedMeshType, 
                                            MovingMeshType >   
                                            MetricType;

  MetricType::Pointer  metric = MetricType::New();

  registration->SetMetric( metric ); 


  registration->SetFixedMesh( meshFixed );
  registration->SetMovingMesh( meshMoving );


  typedef itk::VersorTransform< MetricType::TransformComputationType >  TransformType;

  TransformType::Pointer transform = TransformType::New();

  registration->SetTransform( transform );


  typedef itk::LinearInterpolateMeshFunction< MovingMeshType > InterpolatorType;

  InterpolatorType::Pointer interpolator = InterpolatorType::New();

  registration->SetInterpolator( interpolator );

  const unsigned int numberOfTransformParameters = transform->GetNumberOfParameters();

  typedef TransformType::ParametersType         ParametersType;
  ParametersType parameters( numberOfTransformParameters );

  transform->SetIdentity();
  
  parameters = transform->GetParameters();

  registration->SetInitialTransformParameters( parameters );


  typedef itk::VersorTransformOptimizer     OptimizerType;

  OptimizerType::Pointer      optimizer     = OptimizerType::New();

  registration->SetOptimizer( optimizer );


  typedef OptimizerType::ScalesType             ScalesType;

  ScalesType    parametersScale( numberOfTransformParameters );
  parametersScale[0] = 1.0;
  parametersScale[1] = 1.0;
  parametersScale[2] = 1.0;

  optimizer->SetScales( parametersScale );

  optimizer->MinimizeOn();
  optimizer->SetGradientMagnitudeTolerance( 1e-6 );
  optimizer->SetMaximumStepLength( 0.05 );
  optimizer->SetMinimumStepLength( 1e-9 );
  optimizer->SetRelaxationFactor( 0.9 );
  optimizer->SetNumberOfIterations( 100 );


  CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  try
    {
    registration->StartRegistration();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cerr << "Registration failed" << std::endl;
    std::cout << "Reason " << e << std::endl;
    return EXIT_FAILURE;
    }


  OptimizerType::ParametersType finalParameters = 
                    registration->GetLastTransformParameters();

  std::cout << "final parameters = " << finalParameters << std::endl;
  std::cout << "final value      = " << optimizer->GetValue() << std::endl;


  transform->SetParameters( finalParameters );

  typedef FixedMeshType::Traits   MeshTraits;
  typedef itk::PointSet< MeshPixelType, Dimension, MeshTraits >   PointSetType;

  typedef itk::DeformationFieldFromTransformMeshFilter<
    FixedMeshType, PointSetType >  DeformationFieldFromTransformFilterType;

  DeformationFieldFromTransformFilterType::Pointer deformationFieldFromTransform =
    DeformationFieldFromTransformFilterType::New();

  deformationFieldFromTransform->SetInput( fixedMeshReader1->GetOutput() );
  deformationFieldFromTransform->SetTransform( transform );

  try
    {
    deformationFieldFromTransform->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
    }


  PointSetType::ConstPointer destinationPoints = 
    deformationFieldFromTransform->GetOutput();

  const PointSetType::PointsContainer * dstPoints = destinationPoints->GetPoints();

  const FixedMeshType * fixedMesh = fixedMeshReader1->GetOutput();
  const FixedMeshType::PointsContainer * srcPoints = fixedMesh->GetPoints();

  typedef FixedMeshType::PointType  PointType;
  typedef PointType::VectorType     VectorType;

  typedef itk::QuadEdgeMeshTraits< VectorType, Dimension, bool, bool > VectorPointSetTraits;

  typedef itk::QuadEdgeMesh< VectorType, Dimension, VectorPointSetTraits > MeshWithVectorsType;
  MeshWithVectorsType::Pointer vectorMesh = MeshWithVectorsType::New();

  typedef MeshWithVectorsType::PointDataContainer  PointDataContainer;
  PointDataContainer::Pointer vectors = PointDataContainer::New();

  vectors->Reserve( fixedMesh->GetNumberOfPoints() );

  vectorMesh->SetPoints( const_cast< FixedMeshType::PointsContainer *>( srcPoints ) );
  vectorMesh->SetPointData( vectors );

  PointDataContainer::Iterator  vitr = vectors->Begin();
  PointSetType::PointsContainer::ConstIterator  dstitr = dstPoints->Begin();
  FixedMeshType::PointsContainer::ConstIterator srcitr = srcPoints->Begin();

  while( srcitr != srcPoints->End() )
    {
    vitr.Value() = dstitr.Value() - srcitr.Value();
    ++srcitr;
    ++dstitr;
    ++vitr;
    }

  typedef itk::QuadEdgeMeshVectorDataVTKPolyDataWriter< MeshWithVectorsType >  VectorMeshWriterType;
  VectorMeshWriterType::Pointer vectorMeshWriter = VectorMeshWriterType::New();
  vectorMeshWriter->SetInput( vectorMesh );
  vectorMeshWriter->SetFileName("VectorMesh.vtk");
  vectorMeshWriter->Update(); 



  typedef itk::QuadEdgeMesh< MeshPixelType, Dimension >   RegisteredMeshType;

  typedef itk::QuadEdgeMeshSphericalDiffeomorphicDemonsFilter<
    FixedMeshType, MovingMeshType, RegisteredMeshType >   DemonsFilterType;

  DemonsFilterType::Pointer demonsFilter = DemonsFilterType::New();

  demonsFilter->SetFixedMesh( fixedMeshReader1->GetOutput() );
  demonsFilter->SetMovingMesh( movingMeshReader1->GetOutput() );

  DemonsFilterType::PointType center;
  center.Fill( 0.0 );

  const double radius = 100.0;

  demonsFilter->SetSphereCenter( center );
  demonsFilter->SetSphereRadius( radius );

  const double epsilon = 1.0;
  const double sigmaX = 1.0;
  const double lambda = 1.0;
  const unsigned int maximumNumberOfSmoothingIterations = 10;
  const unsigned int maximumNumberOfIterations = 30;

  demonsFilter->SetEpsilon( epsilon );
  demonsFilter->SetSigmaX( sigmaX );
  demonsFilter->SetMaximumNumberOfIterations( maximumNumberOfIterations );

  demonsFilter->SetLambda( lambda );
  demonsFilter->SetMaximumNumberOfSmoothingIterations( maximumNumberOfSmoothingIterations );

  //
  // Initialize the deformable registration stage with 
  // the results of the Rigid Registration.
  //
  demonsFilter->SetInitialDestinationPoints( destinationPoints );

  try
    {
    demonsFilter->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }


  typedef itk::QuadEdgeMeshScalarDataVTKPolyDataWriter< FixedMeshType >   WriterType;
  WriterType::Pointer writer = WriterType::New();

  writer->SetFileName( argv[3] );
  writer->SetInput( demonsFilter->GetOutput() );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }


  //
  //  Starting process for the second resolution level (IC5).
  // 
  FixedReaderType::Pointer fixedMeshReader2 = FixedReaderType::New();
  fixedMeshReader2->SetFileName( argv[4] );

  MovingReaderType::Pointer movingMeshReader2 = MovingReaderType::New();
  movingMeshReader2->SetFileName( argv[5] );

  try
    {
    fixedMeshReader2->Update( );
    movingMeshReader2->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }


  //
  // Supersample the list of destination points using the mesh at the next resolution level.
  //
  typedef itk::ResampleDestinationPointsQuadEdgeMeshFilter< 
    PointSetType, FixedMeshType, FixedMeshType, PointSetType > UpsampleDestinationPointsFilterType;

  UpsampleDestinationPointsFilterType::Pointer upsampleDestinationPoints = 
    UpsampleDestinationPointsFilterType::New();

  upsampleDestinationPoints->SetInput( demonsFilter->GetFinalDestinationPoints() );
  upsampleDestinationPoints->SetFixedMesh( fixedMeshReader1->GetOutput() );
  upsampleDestinationPoints->SetReferenceMesh( fixedMeshReader2->GetOutput() );
  upsampleDestinationPoints->SetTransform( itk::IdentityTransform<double>::New() );

  try
    {
std::cout << "BEFORE upsampleDestinationPoints Update()" << std::endl;
    upsampleDestinationPoints->Update();
std::cout << "AFTER upsampleDestinationPoints Update()" << std::endl;
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  // Here build a Mesh using the upsampled destination points and
  // the scalar values of the fixed IC5 mesh.

  FixedMeshType::Pointer fixedMesh2 = fixedMeshReader2->GetOutput();
  fixedMesh2->DisconnectPipeline();

  PointSetType::ConstPointer upsampledPointSet = upsampleDestinationPoints->GetOutput();
  
  const PointSetType::PointsContainer * upsampledPoints = upsampledPointSet->GetPoints();

  PointSetType::PointsContainerConstIterator upsampledPointsItr = upsampledPoints->Begin();
  PointSetType::PointsContainerConstIterator upsampledPointsEnd = upsampledPoints->Begin();

  FixedMeshType::PointsContainer::Pointer fixedPoints2 = fixedMesh2->GetPoints();

  FixedMeshType::PointsContainerIterator fixedPoint2Itr = fixedPoints2->Begin();

  while( upsampledPointsItr != upsampledPointsEnd )
    {
    fixedPoint2Itr.Value() = upsampledPointsItr.Value();
    ++fixedPoint2Itr;
    ++upsampledPointsItr;
    }

  // 
  // Now feed this mesh into the Rigid registration of the second resolution level.
  //

  registration->SetFixedMesh( fixedMesh2 );
  registration->SetMovingMesh( movingMeshReader2->GetOutput() );

  transform->SetIdentity();
  parameters = transform->GetParameters();

  registration->SetInitialTransformParameters( parameters );

  // 
  //  Running Second Resolution Level Rigid Registration.
  //

  std::cout << "Running Second Resolution Level Rigid Registration." << std::endl;

  try
    {
    registration->StartRegistration();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cerr << "Registration failed" << std::endl;
    std::cout << "Reason " << e << std::endl;
    return EXIT_FAILURE;
    }

  finalParameters = registration->GetLastTransformParameters();

  std::cout << "final parameters = " << finalParameters << std::endl;
  std::cout << "final value      = " << optimizer->GetValue() << std::endl;

  transform->SetParameters( finalParameters );

  deformationFieldFromTransform->SetInput( fixedMesh2 );
  deformationFieldFromTransform->SetTransform( transform );

  try
    {
    deformationFieldFromTransform->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
    }

  demonsFilter->SetInitialDestinationPoints( deformationFieldFromTransform->GetOutput() );

  demonsFilter->SetFixedMesh( fixedMesh2 );
  demonsFilter->SetMovingMesh( movingMeshReader2->GetOutput() );


  // 
  //  Running Second Resolution Level Demons Registration.
  //
  std::cout << "Running Second Resolution Level Demons Registration." << std::endl;

  try
    {
    demonsFilter->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }

  writer->SetFileName( argv[6] );
  writer->SetInput( demonsFilter->GetOutput() );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }


  //
  //  Starting process for the Third resolution level (IC6).
  // 
  FixedReaderType::Pointer fixedMeshReader3 = FixedReaderType::New();
  fixedMeshReader3->SetFileName( argv[7] );

  MovingReaderType::Pointer movingMeshReader3 = MovingReaderType::New();
  movingMeshReader3->SetFileName( argv[8] );

  try
    {
    fixedMeshReader3->Update( );
    movingMeshReader3->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }


  //
  // Supersample the list of destination points using the mesh at the next resolution level.
  //
  upsampleDestinationPoints->SetInput( demonsFilter->GetFinalDestinationPoints() );
  upsampleDestinationPoints->SetFixedMesh( fixedMesh2 );
  upsampleDestinationPoints->SetReferenceMesh( fixedMeshReader3->GetOutput() );
  upsampleDestinationPoints->SetTransform( itk::IdentityTransform<double>::New() );

  try
    {
std::cout << "BEFORE upsampleDestinationPoints Update()" << std::endl;
    upsampleDestinationPoints->Update();
std::cout << "AFTER upsampleDestinationPoints Update()" << std::endl;
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  // Here build a Mesh using the upsampled destination points and
  // the scalar values of the fixed IC6 mesh.

  FixedMeshType::Pointer fixedMesh3 = fixedMeshReader3->GetOutput();
  fixedMesh3->DisconnectPipeline();

  upsampledPointSet = upsampleDestinationPoints->GetOutput();
  
  upsampledPoints = upsampledPointSet->GetPoints();

  upsampledPointsItr = upsampledPoints->Begin();
  upsampledPointsEnd = upsampledPoints->Begin();

  FixedMeshType::PointsContainer::Pointer fixedPoints3 = fixedMesh3->GetPoints();

  FixedMeshType::PointsContainerIterator fixedPoint3Itr = fixedPoints3->Begin();

  while( upsampledPointsItr != upsampledPointsEnd )
    {
    fixedPoint3Itr.Value() = upsampledPointsItr.Value();
    ++fixedPoint3Itr;
    ++upsampledPointsItr;
    }

  // 
  // Now feed this mesh into the Rigid registration of the third resolution level.
  //

  registration->SetFixedMesh( fixedMesh3 );
  registration->SetMovingMesh( movingMeshReader3->GetOutput() );

  transform->SetIdentity();
  parameters = transform->GetParameters();

  registration->SetInitialTransformParameters( parameters );

  // 
  //  Running Third Resolution Level Rigid Registration.
  //

  std::cout << "Running Third Resolution Level Rigid Registration." << std::endl;

  try
    {
    registration->StartRegistration();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cerr << "Registration failed" << std::endl;
    std::cout << "Reason " << e << std::endl;
    return EXIT_FAILURE;
    }

  finalParameters = registration->GetLastTransformParameters();

  std::cout << "final parameters = " << finalParameters << std::endl;
  std::cout << "final value      = " << optimizer->GetValue() << std::endl;

  transform->SetParameters( finalParameters );

  deformationFieldFromTransform->SetInput( fixedMesh3 );
  deformationFieldFromTransform->SetTransform( transform );

  try
    {
    deformationFieldFromTransform->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
    }

  demonsFilter->SetInitialDestinationPoints( deformationFieldFromTransform->GetOutput() );

  demonsFilter->SetFixedMesh( fixedMesh3 );
  demonsFilter->SetMovingMesh( movingMeshReader3->GetOutput() );


  // 
  //  Running Third Resolution Level Demons Registration.
  //
  std::cout << "Running Third Resolution Level Demons Registration." << std::endl;

  try
    {
    demonsFilter->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }

  writer->SetFileName( argv[9] );
  writer->SetInput( demonsFilter->GetOutput() );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }


  //
  //  Starting process for the Fourth resolution level (IC7).
  // 
  FixedReaderType::Pointer fixedMeshReader4 = FixedReaderType::New();
  fixedMeshReader4->SetFileName( argv[10] );

  MovingReaderType::Pointer movingMeshReader4 = MovingReaderType::New();
  movingMeshReader4->SetFileName( argv[11] );

  try
    {
    fixedMeshReader4->Update( );
    movingMeshReader4->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }


  //
  // Supersample the list of destination points using the mesh at the next resolution level.
  //
  upsampleDestinationPoints->SetInput( demonsFilter->GetFinalDestinationPoints() );
  upsampleDestinationPoints->SetFixedMesh( fixedMesh3 );
  upsampleDestinationPoints->SetReferenceMesh( fixedMeshReader4->GetOutput() );
  upsampleDestinationPoints->SetTransform( itk::IdentityTransform<double>::New() );

  try
    {
std::cout << "BEFORE upsampleDestinationPoints Update()" << std::endl;
    upsampleDestinationPoints->Update();
std::cout << "AFTER upsampleDestinationPoints Update()" << std::endl;
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }

  // Here build a Mesh using the upsampled destination points and
  // the scalar values of the fixed IC6 mesh.

  FixedMeshType::Pointer fixedMesh4 = fixedMeshReader4->GetOutput();
  fixedMesh4->DisconnectPipeline();

  upsampledPointSet = upsampleDestinationPoints->GetOutput();
  
  upsampledPoints = upsampledPointSet->GetPoints();

  upsampledPointsItr = upsampledPoints->Begin();
  upsampledPointsEnd = upsampledPoints->Begin();

  FixedMeshType::PointsContainer::Pointer fixedPoints4 = fixedMesh4->GetPoints();

  FixedMeshType::PointsContainerIterator fixedPoint4Itr = fixedPoints4->Begin();

  while( upsampledPointsItr != upsampledPointsEnd )
    {
    fixedPoint4Itr.Value() = upsampledPointsItr.Value();
    ++fixedPoint4Itr;
    ++upsampledPointsItr;
    }

  // 
  // Now feed this mesh into the Rigid registration of the third resolution level.
  //

  registration->SetFixedMesh( fixedMesh4 );
  registration->SetMovingMesh( movingMeshReader4->GetOutput() );

  transform->SetIdentity();
  parameters = transform->GetParameters();

  registration->SetInitialTransformParameters( parameters );

  // 
  //  Running Fourth Resolution Level Rigid Registration.
  //

  std::cout << "Running Fourth Resolution Level Rigid Registration." << std::endl;

  try
    {
    registration->StartRegistration();
    }
  catch( itk::ExceptionObject & e )
    {
    std::cerr << "Registration failed" << std::endl;
    std::cout << "Reason " << e << std::endl;
    return EXIT_FAILURE;
    }

  finalParameters = registration->GetLastTransformParameters();

  std::cout << "final parameters = " << finalParameters << std::endl;
  std::cout << "final value      = " << optimizer->GetValue() << std::endl;

  transform->SetParameters( finalParameters );

  deformationFieldFromTransform->SetInput( fixedMesh4 );
  deformationFieldFromTransform->SetTransform( transform );

  try
    {
    deformationFieldFromTransform->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cout << excp << std::endl;
    return EXIT_FAILURE;
    }

  demonsFilter->SetInitialDestinationPoints( deformationFieldFromTransform->GetOutput() );

  demonsFilter->SetFixedMesh( fixedMesh4 );
  demonsFilter->SetMovingMesh( movingMeshReader4->GetOutput() );


  // 
  //  Running Fourth Resolution Level Demons Registration.
  //
  std::cout << "Running Fourth Resolution Level Demons Registration." << std::endl;

  try
    {
    demonsFilter->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }

  writer->SetFileName( argv[12] );
  writer->SetInput( demonsFilter->GetOutput() );

  try
    {
    writer->Update();
    }
  catch( itk::ExceptionObject & excp )
    {
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
    }


  return EXIT_SUCCESS;
}
