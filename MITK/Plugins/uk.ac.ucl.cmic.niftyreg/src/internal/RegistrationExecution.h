/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef RegistrationExecution_h
#define RegistrationExecution_h

#include <QThread>

#include "QmitkNiftyRegView.h"
#include <niftkF3DControlGridToVTKPolyData.h>


class RegistrationExecution : public QThread
{
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
public:

  /// Constructor
  RegistrationExecution( void *param );

  /// Starts the event loop by calling exec() 
  void run();


protected slots:

  /// Run the actual registration
  void ExecuteRegistration( void );


protected:
    
  /// A pointer to the registration plugin
  QmitkNiftyRegView* userData;
  
  /// Create a VTK polydata object to visualise the control points
  void CreateControlPointVisualisation( nifti_image *controlPointGrid );
    
  /// Create a VTK polydata object to visualise the control points using spheres
  void CreateControlPointSphereVisualisation( nifti_image *controlPointGrid );

  /// Create a VTK polydata object to visualise the deformation vector field
  void CreateVectorFieldVisualisation( nifti_image *controlPointGrid,
				       int controlGridSkipFactor );
  
  /// Create a VTK polydata object to visualise the deformation
  void CreateDeformationVisualisationSurface( niftk::PlaneType plane,
					      nifti_image *controlPointGrid,
					      int xSkip,
					      int ySkip,
					      int zSkip );

 };


#endif // RegistrationExecution_h

