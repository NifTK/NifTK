/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: QuadEdgeMeshSphericalDiffeomorphicDemonsFilter1.cxx,v $
  Language:  C++
  Date:      $Date: 2010-05-26 10:55:12 +0100 (Wed, 26 May 2010) $
  Version:   $Revision: 3302 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "itkQuadEdgeMeshVTKPolyDataReader.h"
#include "itkQuadEdgeMesh.h"

int main( int argc, char *argv[] )
{
  if( argc < 3 )
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " fixedMeshFile  movingMeshFile " << std::endl;
    return EXIT_FAILURE;
    }
  
  typedef float      MeshPixelType;
  const unsigned int Dimension = 3;

  typedef itk::QuadEdgeMesh< MeshPixelType, Dimension >   FixedMeshType;
  typedef itk::QuadEdgeMesh< MeshPixelType, Dimension >   MovingMeshType;

  typedef itk::QuadEdgeMeshVTKPolyDataReader< FixedMeshType >   FixedReaderType;
  typedef itk::QuadEdgeMeshVTKPolyDataReader< MovingMeshType >  MovingReaderType;

  FixedReaderType::Pointer fixedReader = FixedReaderType::New();
  fixedReader->SetFileName( argv[1] );

  MovingReaderType::Pointer movingReader = MovingReaderType::New();
  movingReader->SetFileName( argv[2] );

  try
    {
    fixedReader->Update( );
    movingReader->Update( );
    }
  catch( itk::ExceptionObject & exp )
    {
    std::cerr << "Exception thrown while reading the input file " << std::endl;
    std::cerr << exp << std::endl;
    return EXIT_FAILURE;
    }

  FixedMeshType::ConstPointer  fixedMesh  = fixedReader->GetOutput();
  MovingMeshType::ConstPointer movingMesh = movingReader->GetOutput();

  if( fixedMesh->GetNumberOfPoints() != movingMesh->GetNumberOfPoints() )
    {
    std::cerr << "Error: the two input meshes do not have the same number of points" << std::endl;
    std::cerr << "Fixed mesh number of points = " << fixedMesh->GetNumberOfPoints() << std::endl;
    std::cerr << "Moving mesh number of points = " << movingMesh->GetNumberOfPoints() << std::endl;
    return EXIT_FAILURE;
    }

  typedef FixedMeshType::PointsContainer      FixedPointsContainer;
  typedef MovingMeshType::PointsContainer     MovingPointsContainer;

  typedef FixedPointsContainer::ConstIterator     FixedPointsConstIterator;
  typedef MovingPointsContainer::ConstIterator    MovingPointsConstIterator;


  FixedPointsContainer::ConstPointer    fixedPoints   = fixedMesh->GetPoints();
  MovingPointsContainer::ConstPointer   movingPoints  = movingMesh->GetPoints();

  FixedPointsConstIterator  fixedPointItr   = fixedPoints->Begin();
  MovingPointsConstIterator movingPointItr  = movingPoints->Begin();

  FixedPointsConstIterator  fixedPointEnd   = fixedPoints->End();
  MovingPointsConstIterator movingPointEnd  = movingPoints->End();

  double sumOfDistances = 0.0;

  while( ( fixedPointItr != fixedPointEnd ) && ( movingPointItr != movingPointEnd ) )
    {
    ++fixedPointItr;
    ++movingPointItr;
    const double distanceSquared = 
      fixedPointItr.Value().SquaredEuclideanDistanceTo( movingPointItr.Value() );
    sumOfDistances += distanceSquared;
    }
  
  const double distancesRMS = vcl_sqrt( sumOfDistances );

  std::cout << "RMS " << distancesRMS << std::endl;

  return EXIT_SUCCESS;
}
