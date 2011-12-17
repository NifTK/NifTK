/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-20 14:34:44 +0100 (Tue, 20 Sep 2011) $
 Revision          : $Revision: 7333 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#include "itkLogHelper.h"
#include "ConversionUtils.h"
#include "vtkPolyData.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataWriter.h"
#include "vtkIntArray.h"
#include "vtkPointData.h"
#include <limits>
#include <sstream>

void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Transfers label values from one PolyData to another, by taking the closest point" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  " << exec << " -ref inputPolyData.vtk -data inputPolyData2.vtk -o outputPolyData.vtk [options]" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "*** [mandatory] ***" << std::endl << std::endl;
    std::cout << "    -ref    <filename>      Input VTK Poly Data, to determine geometry" << std::endl;
    std::cout << "    -data   <filename>      Input VTK Poly Data, containing scalar values that will be transfered" << std::endl;
    std::cout << "    -o    <filename>        Output VTK Poly Data." << std::endl << std::endl;      
    std::cout << "    -phi   radians          Euler angle to rotate target by." << std::endl << std::endl;
    std::cout << "    -theta radians          Euler angle to rotate target by." << std::endl << std::endl;
    std::cout << "    -psi   radians          Euler angle to rotate target by." << std::endl << std::endl;
    std::cout << "                            Unspecified Euler angles are assumed to be zero." << std::endl << std::endl;
    std::cout << "                            N.B. it is the target model that is rotated to compare against the labels (not the label model)." << std::endl << std::endl;
    std::cout << "*** [options]   ***" << std::endl << std::endl;
  }

struct arguments
{
  std::string inputReferencePolyDataFile;
  std::string inputDataPolyDataFile;
  std::string outputPolyDataFile;
  double phi, theta, psi;
  bool eulerRotate;
};

void setDefaultArgs (struct arguments &args) {
  args.eulerRotate=0;
  args.phi=0;
  args.theta=0;
  args.psi=0;
}


double distanceBetweenPoints(double* a, double *b)
  {
    return sqrt(
          ((a[0]-b[0]) * (a[0]-b[0])) 
        + ((a[1]-b[1]) * (a[1]-b[1]))
        + ((a[2]-b[2]) * (a[2]-b[2]))
        );
  }


class eulerRotate {
protected:
  double a[3][3];
public:
  eulerRotate ( double phi, double theta, double psi );
  void eulerRotatePoint ( double *point );
};


eulerRotate::eulerRotate ( double phi, double theta, double psi )
{
  // Euler rotation, right-handed x-convention, phi around z, then theta
  // around x' and finally psi around z'
  a[0][0] = -sin(psi)*sin(phi)+cos(theta)*cos(phi)*cos(psi);
  a[0][1] = sin(psi)*cos(phi) + cos(theta)*sin(phi)*cos(psi);
  a[0][2] = -cos(psi)*sin(theta);
  a[1][0] = -cos(psi)*sin(phi) - cos(theta)* cos(phi) * sin(psi);
  a[1][1] = cos(psi)*cos(phi) - cos(theta)*sin(phi)*sin(psi);
  a[1][2] = sin(psi)*sin(theta);
  a[2][0] = sin(theta)*cos(phi);
  a[2][1] = sin(theta)*sin(phi);
  a[2][2] = cos(theta);
  return;
}

void eulerRotate::eulerRotatePoint ( double *point ) {
  double newpoint[3];
  for ( unsigned row=0; row<3 ; row++) {
    newpoint[row]=0;
    for ( unsigned col=0; col<3; col++) {
      newpoint[row] += a[row][col] * point[col];
    }
  }
  for ( unsigned el=0; el<3 ; el++) {
    point[el]=newpoint[el];
  }
  return;
}


vtkIdType getClosestPoint(
    vtkPoints *points,
    double *point
    )
  {
    
    vtkIdType closestIndex = -1;
    
    double distance = 0;
    double closestDistance = std::numeric_limits<double>::max();
    double pointOnSurface[3];
    
    vtkIdType numberOfPoints = points->GetNumberOfPoints();
    
    for (vtkIdType i = 0; i < numberOfPoints; i++)
      {
        points->GetPoint(i, pointOnSurface);
        
        distance =  distanceBetweenPoints(point, pointOnSurface);
        if (distance < closestDistance)
          {
            closestIndex = i;
            closestDistance = distance;
          }
      }    
    return closestIndex;
  }

/**
 * \brief Drop lines from a poly data.
 */
int main(int argc, char** argv)
{
  // To pass around command line args
  struct arguments args;
  setDefaultArgs(args);

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-ref") == 0){
      args.inputReferencePolyDataFile=argv[++i];
      std::cout << "Set -ref=" << args.inputReferencePolyDataFile;
    }
    else if(strcmp(argv[i], "-data") == 0){
      args.inputDataPolyDataFile=argv[++i];
      std::cout << "Set -data=" << args.inputDataPolyDataFile;
    }
    else if(strcmp(argv[i], "-o") == 0){
      args.outputPolyDataFile=argv[++i];
      std::cout << "Set -o=" << args.outputPolyDataFile;
    }
    else if(strcmp(argv[i], "-phi") == 0){
      if (++i < argc ) {
	if ( sscanf(argv[i],"%lf",&args.phi) != 1 ) {
	  std::cout << "Not valid -phi=" << argv[i];
	  return -1;
	}
      } else {
	std::cout << "-phi needs an option";
	return -1;
      }
      std::ostringstream msg;
      msg << "Set -phi=" << args.phi;
      std::cout << msg.str();
      args.eulerRotate=true;
    }
    else if(strcmp(argv[i], "-theta") == 0){
      if (++i < argc ) {
	if ( sscanf(argv[i],"%lf",&args.theta) != 1 ) {
	  std::cout << "Not valid -theta=" << argv[i];
	  return -1;
	}
      } else {
	std::cout << "-theta needs an option";
	return -1;
      }
      std::ostringstream msg;
      msg << "Set -theta=" << args.theta;
      std::cout << msg.str();
      args.eulerRotate=true;
    }
    else if(strcmp(argv[i], "-psi") == 0){
      if (++i < argc ) {
	if ( sscanf(argv[i],"%lf",&args.psi) != 1 ) {
	  std::cout << "Not a valid -psi=" << argv[i];
	  return -1;
	}
      } else {
	std::cout << "-psi needs an option";
	return -1;
      }
      std::ostringstream msg;
      msg << "Set -psi=" << args.psi;
      std::cout << msg.str();
      args.eulerRotate=true;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }            
  }
  
  // Validate command line args
  if (args.outputPolyDataFile.length() == 0 || args.inputDataPolyDataFile.length() == 0 || args.inputReferencePolyDataFile.length() == 0)
    {
      Usage(argv[0]);
      return EXIT_FAILURE;
    }

  vtkPolyDataReader *referenceReader = vtkPolyDataReader::New();
  referenceReader->SetFileName(args.inputReferencePolyDataFile.c_str());
  referenceReader->Update();

  std::cout << "Loaded PolyData:" << args.inputReferencePolyDataFile << std::endl;
  
  vtkPolyDataReader *dataReader = vtkPolyDataReader::New();
  dataReader->SetFileName(args.inputDataPolyDataFile.c_str());
  dataReader->Update();

  std::cout << "Loaded PolyData:" << args.inputDataPolyDataFile << std::endl;
  
  vtkPoints *referencePoints = referenceReader->GetOutput()->GetPoints();
  vtkPoints *dataPoints = dataReader->GetOutput()->GetPoints();

  vtkIntArray *referenceLabels = dynamic_cast<vtkIntArray*>(referenceReader->GetOutput()->GetPointData()->GetScalars());
  vtkIntArray *dataLabels = dynamic_cast<vtkIntArray*>(dataReader->GetOutput()->GetPointData()->GetScalars());

  vtkIdType pointNumber = 0;
  vtkIdType numberOfPoints = referencePoints->GetNumberOfPoints();
  double point[3];
  vtkIdType closestIndex;

  // Unused if we didn't set args.eulerRotate, but set it in case we did:
  eulerRotate rotation(args.phi,args.theta,args.psi);

  for (pointNumber = 0; pointNumber < numberOfPoints; pointNumber++)
    {
      referencePoints->GetPoint(pointNumber, point);
      if ( pointNumber % ( numberOfPoints/100) == 0 ||
	   pointNumber+1 == numberOfPoints) {
	std::cout << pointNumber+1 << " of " << numberOfPoints << std::endl;
      }
      if(args.eulerRotate) {
	rotation.eulerRotatePoint(point);
      }
      closestIndex = getClosestPoint(dataPoints, point);
      
      int currentLabel = (int)(referenceLabels->GetTuple1(pointNumber));
      int nextLabel = (int)(dataLabels->GetTuple1(closestIndex));
      
      if ((currentLabel < 36 && nextLabel >= 36) || (currentLabel >= 36 && nextLabel < 36))
        {
          std::cerr << "ERROR:point number=" << pointNumber << ", cl=" << currentLabel << ", nl=" << nextLabel << std::endl;
        }
      
      referenceLabels->SetTuple1(pointNumber, dataLabels->GetTuple1(closestIndex));
    }
  
  vtkPolyDataWriter *writer = vtkPolyDataWriter::New();
  writer->SetFileName(args.outputPolyDataFile.c_str());
  writer->SetInput(referenceReader->GetOutput());
  writer->Update();
  
  return EXIT_SUCCESS;

}
