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
  * \param the handeye calibrations to define the location of the lens's relative the tracker
  */
  vtkSmartPointer<vtkPolyData> MakeLaparoscope(const std::vector < std::vector <float> >& IREDPositions,
      const vtkSmartPointer<vtkMatrix4x4> leftHandeye, const  vtkSmartPointer<vtkMatrix4x4> rightHandeye,
      const vtkSmartPointer<vtkMatrix4x4> centreHandeyeFilename ,
      const  bool& AddCrossHairs = true , const float& trackerMarkerRadius = 3.0 ,
      const float& LensAngle = 30.0, const float& BodyLength = 550 );

  /**
  * \brief For visualisation purposes, creates a representation of the laparoscope. Overloaded version
  * of the above, handling file IO.
  * \param the rigid body filename to define the location of the tracking markers
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakeLaparoscope(std::string rigidBodyFilename, std::string leftHandeyeFilename, std::string rightHandeyeFilename, std::string centreHandeyeFilename ,  bool AddCrossHairs = true, float trackerMarkerRadius = 3.0 , float LensAngle = 30.0, float BodyLength = 550 );

  /**
  * \brief For visualisation purposes, creates a representation of a point in space, the point has a sphere around it to help measure calibration error
  * \param the location(s) of the point
  */
  vtkSmartPointer<vtkPolyData> MakeInvariantPoint (const std::vector < std::vector <float> >& IREDPositions, const float& sphereRadius = 10);

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
  * \brief For visualisation purposes, creates a representation of the reference.
  * \param the rigid body filename to define the location of the tracking markers
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakeReferencePolaris(std::string rigidBodyFilename, std::string handeyeFilename );

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
  * \param the length of the axis
  * \param whether or not the axis is symmetric
  * */
  vtkSmartPointer<vtkPolyData> MakeXAxes( const float& length = 4000,const bool& symmetric = false);
  /**
  * \brief For visualisation purposes, make a nice big axes
  * \param the length of the axis
  * \param whether or not the axis is symmetric
  * */
  vtkSmartPointer<vtkPolyData> MakeYAxes( const float& length = 4000,const bool& symmetric = false);
  /**
  * \brief For visualisation purposes, make a nice big axes
  * \param the length of the axis
  * \param whether or not the axis is symmetric
  * */
  vtkSmartPointer<vtkPolyData> MakeZAxes( const float& length = 4000,const bool& symmetric = true);

  /**
   * \brief a special type of axis useful for cameras
   */
  vtkSmartPointer<vtkPolyData>  MakeLapLensAxes();

  /**
   * \brief for visualisation purposes, make a representation of an Optotrak Certus
   * camera unit
   * \param the width of the camera unit
   * \param set to true to turn the neck over so it looks (a bit) more like a polaris
   */
  vtkSmartPointer<vtkPolyData>  MakeOptotrak( const float & width = 500, bool Polaris = false);

  /**
  * \brief For visualisation purposes, creates a representation of a transrectal ultrasound probe
  * \param the handeye calibration to define the tool origin
  */
  vtkSmartPointer<vtkPolyData> MakeTransrectalUSProbe(std::string handeyeFilename );

  /**
  * \brief For visualisation purposes, creates a representation of a monitor
  * \param
  */
  vtkSmartPointer<vtkPolyData> MakeMonitor();

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
  vtkSmartPointer<vtkPolyData> ConnectIREDs ( std::vector < std::vector <float> > IREDPositions , bool isPointer = false , float width = 0.0 );

  /**
  * \brief Connect the IREDS to their centroid with a line
  */
  vtkSmartPointer<vtkPolyData> ConnectIREDsToCentroid ( std::vector < std::vector <float> > IREDPositions);

};
} // end namespace
#endif
