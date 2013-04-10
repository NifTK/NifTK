/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkSimultaneousUnconstrainedMatrixReconRegnMethod_h
#define __itkSimultaneousUnconstrainedMatrixReconRegnMethod_h

#include "itkProcessObject.h"
#include "itkImage.h"
#include "itkSimultaneousUnconstrainedMatrixReconRegnMetric.h"
#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkProjectionGeometry.h"
#include "itkReconstructionUpdateCommand.h"

namespace itk
{

  /** \class SimultaneousUnconstrainedMatrixReconRegnMethod
   * \brief Base class for Image Reconstruction Methods
   *
   * This class defines the generic interface for a reconstruction method.
   *
   * This class is templated over the type of the images to be
   * reconstructed. 
   *
   * The method uses a generic optimizer that can
   * be selected at run-time. The only restriction for the optimizer is
   * that it should be able to operate in single-valued cost functions
   * given that the metrics used to compare images provide a single 
   * value as output.
   */
  template <class TScalarType = double, class IntensityType = float>
    class ITK_EXPORT SimultaneousUnconstrainedMatrixReconRegnMethod : public ProcessObject 
  {
    public:
      /** Standard class typedefs. */
      typedef SimultaneousUnconstrainedMatrixReconRegnMethod  															Self;
      typedef ProcessObject  																																Superclass;
      typedef SmartPointer<Self>   																													Pointer;
      typedef SmartPointer<const Self>  																										ConstPointer;

      typedef itk::ReconstructionUpdateCommand     																					ReconstructionUpdateCommandType;
      typedef ReconstructionUpdateCommand::Pointer 																					ReconstructionUpdateCommandPointer;

      /** Method for creation through the object factory. */
      itkNewMacro(Self);

      /** Run-time type information (and related methods). */
      itkTypeMacro(SimultaneousUnconstrainedMatrixReconRegnMethod, ProcessObject);

      // Some convenient typedefs.

      /// Type of the vector to store the two input vectors
      typedef vnl_vector<IntensityType>                   																	VectorType;

      /** Intensity type has to be double because the optimizer expects
        the parameters (intensities) to be double */

      typedef Image<IntensityType, 3>                   																		SimultaneousUnconstrainedReconRegnType;
      typedef typename SimultaneousUnconstrainedReconRegnType::Pointer      								SimultaneousUnconstrainedReconRegnPointer;
      typedef typename SimultaneousUnconstrainedReconRegnType::RegionType   								SimultaneousUnconstrainedReconRegnRegionType;
      typedef typename SimultaneousUnconstrainedReconRegnType::PixelType    								SimultaneousUnconstrainedReconRegnPixelType;
      typedef typename SimultaneousUnconstrainedReconRegnType::SizeType     								SimultaneousUnconstrainedReconRegnSizeType;
      typedef typename SimultaneousUnconstrainedReconRegnType::SpacingType  								SimultaneousUnconstrainedReconRegnSpacingType;
      typedef typename SimultaneousUnconstrainedReconRegnType::PointType   					 				SimultaneousUnconstrainedReconRegnPointType;
      typedef typename SimultaneousUnconstrainedReconRegnType::IndexType    								SimultaneousUnconstrainedReconRegnIndexType;

      /// Type of the optimizer.
      typedef SingleValuedNonLinearOptimizer           																			OptimizerType;
      typedef typename OptimizerType::Pointer       																				OptimizerPointer;

      /// The type of the metric
      typedef SimultaneousUnconstrainedMatrixReconRegnMetric<double, float> 								MetricType;
      typedef typename MetricType::Pointer          																				MetricPointer;

      /// The projection geometry type
      typedef itk::ProjectionGeometry<IntensityType>   																			ProjectionGeometryType;
      typedef typename ProjectionGeometryType::Pointer 																			ProjectionGeometryPointer;

      /** Type of the optimisation parameters (reconstructed intensities).
       *  This is the same type used to represent the search space of the
       *  optimization algorithm */
      typedef typename MetricType::ParametersType    																				ParametersType;

      /** Type for the output: Using Decorator pattern for enabling
       *  the reconstructed volume to be passed in the data pipeline */
      typedef SimultaneousUnconstrainedReconRegnType                               					SimultaneousUnconstrainedReconRegnOutputType;
      typedef typename SimultaneousUnconstrainedReconRegnOutputType::Pointer       					SimultaneousUnconstrainedReconRegnOutputPointer;
      typedef typename SimultaneousUnconstrainedReconRegnOutputType::ConstPointer  					SimultaneousUnconstrainedReconRegnOutputConstPointer;

      /** Set/Get the Optimizer. */
      itkSetObjectMacro( Optimizer, OptimizerType );
      itkGetObjectMacro( Optimizer, OptimizerType );

      /** Set/Get the Metric. */
      itkSetObjectMacro( Metric, MetricType );
      itkGetObjectMacro( Metric, MetricType );

      /** Set/Get the Projection Geometry. */
      itkSetObjectMacro( ProjectionGeometry, ProjectionGeometryType );
      itkGetObjectMacro( ProjectionGeometry, ProjectionGeometryType );

      /** Set/Get the ReconstructionUpdateCommand. */
      itkSetObjectMacro( ReconstructionUpdateCommand, ReconstructionUpdateCommandType );
      itkGetObjectMacro( ReconstructionUpdateCommand, ReconstructionUpdateCommandType );

      /// Set the 3D reconstruction estimate volume 
      //void SetSimultaneousUnconReconRegnEstimate( SimultaneousUnconstrainedReconRegnType *im3D);

      /// Update the 3D reconstruction estimate volume 
      // void UpdateSimultaneousUnconReconRegnEstimate( SimultaneousUnconstrainedReconRegnType *im3D);
      /** Update the 3D reconstruction estimate volume with the average of
        the existing estimate and the supplied volume. */
      // void UpdateSimultaneousUnconReconRegnEstimateWithAverage( SimultaneousUnconstrainedReconRegnType *im3D);
      /// Update the initial optimisation parameters
      // void UpdateInitialParameters(void);

      /// Set the two sets of input projection images (y_1 and y_2)
      // bool SetInputProjectionSetOne( VectorType *im2DOne);
      // bool SetInputProjectionSetTwo( VectorType *im2DTwo);

      /// Set the size, resolution and origin of the reconstructed image
      void SetSimultaneousUnconReconRegnVolumeSize( SimultaneousUnconstrainedReconRegnSizeType &reconRegnSize )
      { m_ReconRegnVolumeSize = reconRegnSize; };
      void SetSimultaneousUnconReconRegnVolumeSpacing( SimultaneousUnconstrainedReconRegnSpacingType &reconRegnSpacing ) 
      {m_ReconRegnVolumeSpacing = reconRegnSpacing;};
      void SetSimultaneousUnconReconRegnVolumeOrigin( SimultaneousUnconstrainedReconRegnPointType &reconRegnOrigin ) 
      {m_ReconRegnVolumeOrigin = reconRegnOrigin;};

      /** Initialise by setting the interconnects between the components. */
      virtual void Initialise() throw (ExceptionObject);

      /** Returns the input image  */
      // VectorType *GetInputOne();
      /** Returns the input image  */
      // VectorType *GetInputTwo();
      /** Returns the image resulting from the reconstruction process  */
      SimultaneousUnconstrainedReconRegnOutputType *GetOutput();

      /** Returns the image resulting from the reconstruction process  */
      SimultaneousUnconstrainedReconRegnOutputType *GetReconstructedVolume() const;

      /** Make a DataObject of the correct type to be used as the specified
       * output. */
      virtual DataObjectPointer MakeOutput(unsigned long int idx);

      /** Method to return the latest modified time of this object or
       * any of its cached ivars */
      unsigned long GetMTime() const;  

    protected:
      SimultaneousUnconstrainedMatrixReconRegnMethod();
      virtual ~SimultaneousUnconstrainedMatrixReconRegnMethod() {};
      void PrintSelf(std::ostream& os, Indent indent) const;

      /** We avoid propagating the input region to the output by
        overloading this function */
      virtual void GenerateOutputInformation() {};

      /** Method that initiates the reconstruction. This will Initialise and ensure
       * that all inputs the registration needs are in place, via a call to 
       * Initialise() will then start the optimization process via a call to 
       * StartOptimization()  */
      void StartSimultaneousUnconReconRegn(void);

      /** Method that initiates the optimization process. This method should not be
       * called directly by the users. Instead, this method is intended to be
       * invoked internally by the StartSimultaneousUnconReconRegn() which is in turn invoked by
       * the Update() method. */
      void StartOptimization(void);

      /** Method invoked by the pipeline in order to trigger the computation of 
       * the reconstruction. */
      void GenerateData ();


    private:
      SimultaneousUnconstrainedMatrixReconRegnMethod(const Self&);		// purposely not implemented
      void operator=(const Self&);	          												// purposely not implemented

      bool                             																m_FlagInitialised;

      OptimizerPointer                 																m_Optimizer;
      MetricPointer                    																m_Metric;
      ProjectionGeometryPointer        																m_ProjectionGeometry;

      // VectorType*     																							m_ProjectionImageSetOne;
      // VectorType*     																							m_ProjectionImageSetTwo;
      SimultaneousUnconstrainedReconRegnPointer            						m_VolumeEstimate;

      ParametersType                   																m_InitialParameters;
      ParametersType                   																m_LastParameters;

      SimultaneousUnconstrainedReconRegnSizeType           						m_ReconRegnVolumeSize;
      SimultaneousUnconstrainedReconRegnSpacingType        						m_ReconRegnVolumeSpacing;
      SimultaneousUnconstrainedReconRegnPointType          						m_ReconRegnVolumeOrigin;

      /** To print out the reconstruction status as we go. */
      ReconstructionUpdateCommandPointer 															m_ReconstructionUpdateCommand;

  };


} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSimultaneousUnconstrainedMatrixReconRegnMethod.txx"
#endif

#endif




