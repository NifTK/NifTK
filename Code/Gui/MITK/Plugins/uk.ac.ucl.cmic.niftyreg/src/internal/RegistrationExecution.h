/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-07 07:38:36 +0100 (Sat, 07 Jul 2012) $
 Revision          : $Revision: 9321 $
 Last modified by  : $Author: jhh $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef RegistrationExecution_h
#define RegistrationExecution_h

#include <QThread>

#include "QmitkNiftyRegView.h"


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

  typedef enum {
    PLANE_XY,           //!< Create the 'xy' plane deformation field
    PLANE_YZ,           //!< Create the 'yz' plane deformation field
    PLANE_XZ,           //!< Create the 'xz' plane deformation field
  } PlaneType;                                             

    
  /// A pointer to the registration plugin
  QmitkNiftyRegView* userData;
  
  /// Create a VTK polydata object to visualise the control points
  void CreateControlPointVisualisation( void );
    
  /// Create a VTK polydata object to visualise the control points using spheres
  void CreateControlPointSphereVisualisation( void );
    
  template<class SplineTYPE>
    SplineTYPE GetValue(SplineTYPE *array, int *dim, int x, int y, int z);

  template<class SplineTYPE>
    void SetValue(SplineTYPE *array, int *dim, int x, int y, int z, SplineTYPE value);
  

  template <class DTYPE>
    void reg_bspline_refineControlPointGrid2D( nifti_image *targetImage,
					       nifti_image *splineControlPoint,
					       float xRefineFactor, 
					       float yRefineFactor );
  
  template <class DTYPE>
    void reg_bspline_refineControlPointGrid3D(nifti_image *targetImage,
					      nifti_image *splineControlPoint,
					      float xRefineFactor, 
					      float yRefineFactor, 
					      float zRefineFactor);
  
  void reg_bspline_refineControlPointGrid(nifti_image *referenceImage,
					  nifti_image *controlPointGrid,
					  float xRefineFactor, 
					  float yRefineFactor, 
					  float zRefineFactor);

  /// Create a VTK polydata object to visualise the deformation
  void CreateDeformationVisualisationSurface( PlaneType plane );

 };


#endif // RegistrationExecution_h

