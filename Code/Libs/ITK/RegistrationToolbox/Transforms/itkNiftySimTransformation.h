/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 13:21:33 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8042 $
 Last modified by  : $Author: jhh $

 Original authors  : j.hipwell@ucl.ac.uk, l.han@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/


#ifndef __itkNiftySimTransformation_h
#define __itkNiftySimTransformation_h

#include <iostream>

#include "itkEulerAffineTransform.h"
#include "itkDeformableTransform.h"
#include "itkImage.h"
#include "itkVector.h"
#include "itkExceptionObject.h"
#include "itkMacro.h"

#include "tledSolver.h"
#include "tledSolverCPU.h"
#include "tledMatrixFunctions.h"
#include "tledConstraintManager.h"
#include "tledModel.h"
#include "tledTimer.h"
#include "tledSolutionWriter.h"
#include "tledContactManager.h"

#ifdef _GPU_
   #include "tledSolverGPU.h"
   #include "tledSolverGPU_ROM.h"
   #include <cutil.h>
#endif   // _GPU_

#ifdef _Visualisation_
   #include "tledModelViewer.h"
#endif // _Visualisation_

#include "tledSimulator.h"


namespace itk
{

/**
 * \class NiftySimTransformation
 * \brief Class to apply a NiftySim transformation to an image.
 *
 * \ingroup Transforms
 *
 */


template <
    class TFixedImage,                   // Templated over the image type.
    class TScalarType,                   // Data type for scalars
    unsigned int NDimensions,            // Number of Dimensions i.e. 2D or 3D
    class TDeformationScalar>            // Data type in the deformation field.       
class ITK_EXPORT NiftySimTransformation : 
public DeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >
{
public:
  
  /** Standard class typedefs. */
  typedef NiftySimTransformation                                                            Self;
  typedef DeformableTransform< TFixedImage, TScalarType, NDimensions, TDeformationScalar >  Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );
  
  /** Run-time type information (and related methods). */
  itkTypeMacro( NiftySimTransformation, DeformableTransform );
  
  /** Get the number of dimensions. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  
  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType                      ScalarType;
  
  /** Standard parameters container. */
  typedef typename Superclass::ParametersType                  ParametersType;
  
  /** Standard Jacobian container. */
  typedef typename Superclass::JacobianType                    JacobianType;

  /** Standard coordinate point type for this class. */
  typedef typename Superclass::OutputPointType                 OutputPointType;
  typedef typename Superclass::InputPointType                  InputPointType;

  /** Typedefs for the deformation field. */
  typedef typename Superclass::DeformationFieldType            DeformationFieldType;
  typedef typename Superclass::DeformationFieldPixelType       DeformationFieldPixelType;
  typedef typename Superclass::DeformationFieldSpacingType     DeformationFieldSpacingType;
  typedef typename Superclass::DeformationFieldSizeType        DeformationFieldSizeType;
  typedef typename Superclass::DeformationFieldIndexType       DeformationFieldIndexType;

  /** The deformation field is defined over the fixed image. */
  typedef TFixedImage                                          FixedImageType;
  typedef typename TFixedImage::ConstPointer                   FixedImagePointer;

  /** A mask image defining the region over which the deformation is defined. */
  typedef unsigned char                                       DeformationFieldMaskPixelType;
  typedef Image< DeformationFieldMaskPixelType, NDimensions > DeformationFieldMaskType;
  typedef typename DeformationFieldMaskType::Pointer          DeformationFieldMaskPointer;
  typedef ImageRegion<NDimensions>                            DeformationFieldMaskRegionType;
  typedef typename DeformationFieldMaskRegionType::IndexType  DeformationFieldMaskIndexType;
  typedef typename DeformationFieldMaskRegionType::SizeType   DeformationFieldMaskSizeType;
  typedef typename DeformationFieldMaskType::SpacingType      DeformationFieldMaskSpacingType;
  typedef typename DeformationFieldMaskType::DirectionType    DeformationFieldMaskDirectionType;
  typedef typename DeformationFieldMaskType::PointType        DeformationFieldMaskOriginType;

  /// Global transform for rotation and translation
  typedef typename itk::EulerAffineTransform<double, NDimensions, NDimensions> EulerAffineTransformType;

  typedef typename EulerAffineTransformType::Pointer EulerAffineTransformPointer;
  typedef typename EulerAffineTransformType::ParametersType EulerAffineTransformParametersType;
  typedef typename EulerAffineTransformType::InputPointType EulerAffineTransformPointType;


  /// The input XML model file
  void SetxmlFName( char *file ) { m_xmlFName = file; this->Modified(); }
  char *SetxmlFName( ) { return m_xmlFName; }

  /// Print all nodal forces
  itkSetMacro( printNForces, bool );
  itkGetMacro( printNForces, bool );

  /// Print loaded nodal forces
  itkSetMacro( printNDispForces, bool );
  itkGetMacro( printNDispForces, bool );

  /// Print sums of loaded nodal forces
  itkSetMacro( printNDispForcesSums, bool );
  itkGetMacro( printNDispForcesSums, bool );

  /// Print all nodal displacements
  itkSetMacro( printNDisps, bool );
  itkGetMacro( printNDisps, bool );

  /// Plot the results
  itkSetMacro( plotModel, bool );
  itkGetMacro( plotModel, bool );

  /// Use GPU solver
  itkSetMacro( sportMode, bool );
  itkGetMacro( sportMode, bool );

  /// Report execution time
  itkSetMacro( doTiming, bool );
  itkGetMacro( doTiming, bool );

  /// Verbose mode
  itkSetMacro( Verbose, bool );
  itkGetMacro( Verbose, bool );

  /** Return a pointer to the deformation field mask (the mask needs
      to be rotated to be in the correct space). */
  itkGetObjectMacro( DeformationFieldMask, DeformationFieldMaskType );

  /// Print the simulation results
  virtual void PrintResults();

  /// Plot the mesh
  virtual void PlotMeshes();

  /** 
   * Initialises the deformation field mask and reads the XML model description file..
   */
  virtual void Initialize(FixedImagePointer image);

  /// This method sets the parameters of the transform.
  virtual void SetParameters(const ParametersType & parameters);

  /// Set the global rotation parameters
  void SetRotationParameters( EulerAffineTransformParametersType &rotations);
  /// Set the global rotation center
  void SetRotationCenter( EulerAffineTransformPointType &center);
  /// Set the global translation parameters
  void SetTranslationParameters( EulerAffineTransformParametersType &translations);

  /// Transform a mm coordinate
  virtual OutputPointType TransformPoint(const InputPointType  &point ) const;

  /** 
   * Set the deformation field to Identity.
   * Doesn't affect the Global transform.
   * Doesn't resize anything either.
   */
  virtual void SetIdentity();
  
  /**
   * Return true if the deformable is regriddable. 
   * This then requires the implementation the Regrid function. 
   */
  virtual bool IsRegridable() const { return false; }

  /// Write the deformation field mask image to a file
  void WriteDeformationFieldMask(const char *fname);

  /// Write the model nodal displacements to an ITK vectpr image file
  void WriteDisplacementsToFile(const char *fname);

  /// Write the model nodal displacements to a text file
  void WriteDisplacementsToTextFile(const char *fname);

  /// Write the model to an XML file
  void WriteModelToFile(const char *fname);

  /// Write the model nodal positions to a text file
  void WriteNodePositionsToTextFile(const char *fname);
  /// Write the model original nodal positions and displacements from rotation to a text file
  void WriteNodePositionsAndRotationToTextFile(const char *fname);
  /// Write the model rotated nodal positions and displacements to a text file
  void WriteRotatedNodePositionsAndDisplacementsToTextFile(const char *fname);


protected:

  NiftySimTransformation();
  virtual ~NiftySimTransformation();

  /** Print contents of an NiftySimTransformation. */
  void PrintSelf(std::ostream &os, Indent indent) const;

  /// Flag indicating whether the object has been initialised
  bool m_FlagInitialised;

  /// The input XML model file
  char *m_xmlFName;

  /// Print all nodal forces
  bool m_printNForces;

  /// Print loaded nodal forces
  bool m_printNDispForces;

  /// Print sums of loaded nodal forces
  bool m_printNDispForcesSums;

  /// Print all nodal displacements
  bool m_printNDisps;

  /// Plot the results
  bool m_plotModel;

  /// Use GPU solver
  bool m_sportMode;

  /// Report execution time
  bool m_doTiming;

  /// Verbose mode
  bool m_Verbose;

  /// The number of nodes in the model
  int m_NumberOfOriginalNodes;
	
  /** A copy of the original node coordinates so that we can rotate
      them with the image when we update the global rotation
      transformation. */
  std::vector<float> m_OriginalNodeCoordinates;
  /** Vector to store the transformed node coordinates prior to
      passing to the solver. */
  std::vector<float> m_TransformedNodeCoordinates;
  
  /// The fixed/target image
  FixedImagePointer m_FixedImage;

  /// A mask defining the volume over which the deformation is defined
  DeformationFieldMaskPointer m_DeformationFieldMask;

  /// The biomechanical model
  tledModel* m_Model;

  /// The simulator
  tledSimulator* m_Simulator;

  /// Global transformation center
  EulerAffineTransformPointType m_GlobalTransformationCenter;
  /// Global rotation parameters
  EulerAffineTransformParametersType m_GlobalRotationParameters;
  /// Global translation parameters
  EulerAffineTransformParametersType m_GlobalTranslationParameters;

  /// The global rotation transformation
  EulerAffineTransformPointer m_GlobalRotationTransform;
  /// The global inverse-rotation transformation
  EulerAffineTransformPointer m_GlobalInverseRotationTransform;


  /// Run a simulation
  bool RunSimulation();

  /** Calculate voxel displacements for a given set of node displacements.
      written by Dr Lianghao Han, CMIC, UCL,05/08/2010 */
  void CalculateVoxelDisplacements();
  
  std::vector<float> RotateVector( std::vector<float> vInput );
  std::vector<float> InverseRotateVector( std::vector<float> vInput );

  void MatDet33(double A[3][3], double* R);
  void findVolumeMaxMin(double partVol[4], double* max, double* min,double* sum);
  void findmaxmin(double u[4], double* max, double* min);
  int point_in_tetrahedron(double Node[4][3], double point[3], double uNodes[4][3], 
			   DeformationFieldPixelType &uOut);
  


  /** To get the valid Jacobian region - the deformation around the edge is 0. */
  virtual typename Superclass::JacobianDeterminantFilterType::OutputImageRegionType GetValidJacobianRegion() const 
  { 
    typename Superclass::JacobianDeterminantFilterType::OutputImageRegionType region 
      = this->m_JacobianFilter->GetOutput()->GetLargestPossibleRegion(); 
  
    for (unsigned int i = 0; i < NDimensions; i++)
    {
      region.SetIndex(i, 2); 
      region.SetSize(i, region.GetSize(i)-4); 
    }
    return region; 
  } 

  

private:

  NiftySimTransformation(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};  
  
} // namespace itk.

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkNiftySimTransformation.txx"
#endif


#endif /*  __itkNiftySimTransformation_h */
