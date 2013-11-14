/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGIGeometry_h
#define niftkIGIGeometry_h

#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

#include "niftkVTKWin32ExportHeader.h"
namespace niftk
{

class NIFTKVTK_WINEXPORT VTKIGIGeometry
{
public:

  /**
  * \brief For visualisation purposes, creates a representation of the laparoscope.
  * \param the rigid body filename to define the location of the tracking markers
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakeLaparoscope(std::string rigidBodyFilename, std::string handeyeFilename );

  /**
  * \brief For visualisation purposes, creates a representation of the pointer.
  * \param the rigid body filename to define the location of the tracking markers
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakePointer(std::string rigidBodyFilename, std::string handeyeFilename);

  /**
  * \brief For visualisation purposes, creates a representation of the reference.
  * \param the rigid body filename to define the location of the tracking markers
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakeReference(std::string rigidBodyFilename, std::string handeyeFilename );

  /**
  * \brief For visualisation purposes, make a wall of a cube
  * \param the size of the cube in mm 
  * \param which wall to make 
  * \param the xoffset, room will be centred at x= size * xOffset 
  * */
  vtkSmartPointer<vtkPolyData>  MakeAWall( const int& whichwall, const float& size = 4000,
    const float& xOffset = 0.0 , const float& yOffset = 0.0, const float& zOffset = -0.3, 
    const float& thickness = 10.0);

  /**
  * \brief For visualisation purposes, make a nice big axes
  * \param the length of each axis
  * \param whether or not the z axis is symmetric 
  * */
  vtkSmartPointer<vtkPolyData>  MakeAxes( const float& length = 4000,
      const bool& symmetricZ = true);

  /** 
   * \brief a special type of axis useful for cameras
   */
  vtkSmartPointer<vtkPolyData>  MakeLapLensAxes();

  /**
   * \brief for visualisation purposes, make a representation of an Optotrak Certus 
   * camera unit
   * \param the width of the camera unit
   */
  vtkSmartPointer<vtkPolyData>  MakeOptotrak( const float & width);

  /**
  * \brief For visualisation purposes, creates a representation of a transrectal ultrasound probe
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakeTransrectalUSProbe(std::string handeyeFilename );

private:
  /** 
   * \brief get the IRED positions from a rigid body definition file
   * \param the file name
   */
  std::vector<std::vector <float> > ReadRigidBodyDefinitionFile(std::string rigidBodyFilename);

  /**
  * \brief put down spheres to represent each IRED
  */
  vtkSmartPointer<vtkPolyData> MakeIREDs ( std::vector <std::vector <float> > IREDPositions,
      float Radius = 3.0, int ThetaResolution = 8 , int PhiResolution = 8 );
  
  /** 
   * \brief Get the centroid of a vector of floats
   */
  std::vector <float>  Centroid(std::vector < std::vector <float> > );

  /**
  * \brief Connect the IREDS with a line
  */
  vtkSmartPointer<vtkPolyData> ConnectIREDs ( std::vector < std::vector <float> > IREDPositions , bool isPointer = false );

};
} // end namespace
#endif
