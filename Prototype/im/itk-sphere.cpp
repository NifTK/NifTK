/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : malone@drc.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include <iostream>
#include <vector>

#include <unistd.h>

#include "itkQuadEdgeMesh.h"
#include "itkRegularSphereMeshSource.h"
#include "itkQuadEdgeMeshScalarDataVTKPolyDataWriter.h"

void usage (void) {
  printf ("itk-sphere outname [-r R]\n\n");
  printf ("\t-r R\t resolution level of sphere, integer > 0\n\n");
  return;
}

int main (int argc, char **argv) {
  const int dims = 3;
  const int defaultres = 7;
  int resolution = defaultres;
  int restmp;

  char *optstring  = "r:";
  int opt;
  int argprob = 0;
  char *outname;

  while ( (opt = getopt(argc, argv, optstring)) >= 0 ) {
    switch (opt) {
    case 'r' :
      if ( sscanf(optarg, "%d", &restmp) == 1) {
	resolution = restmp;
      } else {
	argprob = 1;
	printf ("-r needs an integer argument\n");
      }
      break;
    case ':' :
      argprob = 1;
      break;
    case '?' :
      argprob = 1;
      break;
    default :
      printf ("Error processing arguments\n");
      argprob = 1;
  }
  }
  if ( optind != (argc-1) ) {
    printf ("Need a (only one) filename to output\n");
    argprob = 1;
  } else {
    outname = argv[optind];
  }

  if ( argprob ) {
    usage();
    return 0;
  }

  typedef itk::QuadEdgeMesh<float, dims>   MeshT;
  typedef itk::RegularSphereMeshSource< MeshT >  SourceT;
  SourceT::Pointer  mySource = SourceT::New();
  typedef SourceT::VectorType  VectorT;
  typedef SourceT::PointType   PointT;

  PointT myCentre; 
  myCentre.Fill( 0.0 );
  VectorT myScale;
  myScale.Fill (1.0);

  mySource->SetCenter(myCentre);
  mySource->SetScale(myScale);

  mySource->SetResolution(resolution);
  mySource->Modified();
  mySource->Update();

  std::cout << "mySource: " << mySource;

  MeshT::Pointer myMesh = mySource->GetOutput();

  typedef itk::QuadEdgeMeshScalarDataVTKPolyDataWriter< MeshT >   WriterT;
  WriterT::Pointer myWriter = WriterT::New();
  myWriter->SetInput( myMesh );
  myWriter->SetFileName( outname );
  try
    {
      myWriter->Update();
    }
  catch( itk::ExceptionObject &err) {
        std::cerr << err << std::endl;
  }

  return 0;
}
