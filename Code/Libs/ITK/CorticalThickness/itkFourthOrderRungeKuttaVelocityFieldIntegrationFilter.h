/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter_h
#define itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter_h

#include <itkVector.h>
#include <itkImage.h>
#include <itkImageToImageFilter.h>
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkPoint.h>
#include <itkContinuousIndex.h>

namespace itk {

/**
 * \class FourthOrderRungeKuttaVelocityFieldIntegrationFilter
 * \brief This filter integrates a time varying velocity field, using fourth order Runge-Kutta.
 * This filter is basically a tidied up version of that in ANTS: http://www.picsl.upenn.edu/ANTS/
 * However, ANTS has the four timepoints back to front. I don't know if this is intentional.
 * 
 * Also note, that I was reading Numerical Recipes in C, page 711, the bit on fourth-order
 * Runge-Kutta method. If you are reading this, bear in mind that we are integrating through time.
 * This means that for a 3D image, with x,y,z ordinates, and a time ordinate, that it is the
 * time ordinate (index 3) that corresponds to x_n in Numerical Recipes. So, for a small
 * shift in time, given by DeltaTime in this class, and h in Numerical Recipes, we calculate
 * a new point location (x,y,z), analagous to y_n in Numerical Recipes.  
 * 
 * Note also that we have start time and end time, normally, 0 and 1 respectively. If you
 * reverse these you are integrating backwards, which computes the inverse. Also delta time controls
 * the size of the step. We essentially start at start time (0), and at each step we add delta time (0.1 say)
 * until we exceed end time (1). So, in effect this can be independent of the size of the time
 * dimension of your velocity field. You could have delta time = 0.1, and have the size of your time
 * dimension = 3.
 *
 * The mask image, set using SetMaskImage, determines which voxels are integrated. Any value in
 * this mask apart from zero, will cause that voxel to be integrated. It is assumed that this
 * is the same size, as the input image (but obviously, only 1 time dimension).
 * 
 * The max distance mask is used to stop the integration if the Euclidean distance of a point
 * has gone above that in the mask. 
 * 
 * Once this class has integrated all the points in the mask, it is possible to compute a
 * thickness image, as described in Das et. al. NeuroImage 2009 doi:10.1016/j.neuroimage.2008.12.016.
 * To do this, a mask is set, currently called the SetGreyWhiteInterfaceMask method. For each of these voxels,
 * the computed thickness value, is propogated through the velocity field. This could be done in
 * another filter, with a small amount of refactoring, but it is in here for now. The result
 * can be obtained by setting CalculateThickness to true, calling an Update and then
 * GetCalculatedThicknessImage after the update.
 * 
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT FourthOrderRungeKuttaVelocityFieldIntegrationFilter : 
public ImageToImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions + 1>, // Input image is time varying velocity field.
                           Image< Vector<TScalarType, NDimensions>,  NDimensions>      // Output image is integrated displacement field.
                         >
{
  
  public:
    
    /** Standard ITK "Self" typedefs. */
    typedef FourthOrderRungeKuttaVelocityFieldIntegrationFilter                           Self;
    typedef ImageToImageFilter<Image< Vector<TScalarType, NDimensions>,  NDimensions + 1>,
                               Image< Vector<TScalarType, NDimensions>,  NDimensions>
                              >                                                           Superclass;
    typedef SmartPointer<Self>                                                            Pointer;
    typedef SmartPointer<const Self>                                                      ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourthOrderRungeKuttaVelocityFieldIntegrationFilter, ImageToImageFilter);

    /** Get the number of dimensions we are working in. */
    itkStaticConstMacro(Dimension, unsigned int, NDimensions);

    /** Standard typedefs. */
    typedef Vector< TScalarType, NDimensions >                                            TimeVaryingVelocityPixelType;
    typedef Image< TimeVaryingVelocityPixelType, NDimensions + 1 >                        TimeVaryingVelocityImageType;
    typedef typename TimeVaryingVelocityImageType::RegionType                             TimeVaryingVelocityRegionType;
    typedef typename TimeVaryingVelocityImageType::IndexType                              TimeVaryingVelocityIndexType;
    typedef typename TimeVaryingVelocityImageType::SizeType                               TimeVaryingVelocitySizeType;
    typedef typename TimeVaryingVelocityImageType::PointType                              TimeVaryingVelocityPointType;
    typedef typename TimeVaryingVelocityImageType::SpacingType                            TimeVaryingVelocitySpacingType;
    typedef typename TimeVaryingVelocityImageType::DirectionType                          TimeVaryingVelocityDirectionType;
    typedef TScalarType                                                                   MaskPixelType;
    typedef Image< MaskPixelType, NDimensions>                                            MaskImageType;
    typedef typename MaskImageType::Pointer                                               MaskImagePointer;
    typedef typename MaskImageType::IndexType                                             MaskImageIndexType;
    typedef typename MaskImageType::PointType                                             MaskImagePointType;
    typedef Vector< TScalarType, NDimensions >                                            DisplacementPixelType;
    typedef Image< DisplacementPixelType, NDimensions>                                    DisplacementImageType;
    typedef typename DisplacementImageType::RegionType                                    DisplacementImageRegionType;
    typedef typename DisplacementImageType::IndexType                                     DisplacementImageIndexType;
    typedef typename DisplacementImageType::SizeType                                      DisplacementImageSizeType;
    typedef typename DisplacementImageType::PointType                                     DisplacementImagePointType;
    typedef typename DisplacementImageType::SpacingType                                   DisplacementImageSpacingType;
    typedef typename DisplacementImageType::DirectionType                                 DisplacementImageDirectionType;
    typedef VectorLinearInterpolateImageFunction< TimeVaryingVelocityImageType, 
                                                  TScalarType >                           TimeVaryingVelocityFieldInterpolatorType;
    typedef typename TimeVaryingVelocityFieldInterpolatorType::Pointer                    TimeVaryingVelocityFieldInterpolatorPointer;
    typedef Point<TScalarType, NDimensions + 1>                                           TimeVaryingPointType;
    typedef Image<TScalarType, NDimensions>                                               ThicknessImageType;
    typedef typename ThicknessImageType::PixelType                                        ThicknessImagePixelType;
    typedef typename ThicknessImageType::Pointer                                          ThicknessImagePointer;
    typedef NearestNeighborInterpolateImageFunction<ThicknessImageType, TScalarType>      ThicknessImageInterpolatorType;
    typedef typename ThicknessImageInterpolatorType::Pointer                              ThicknessImageInterpolatorPointer;
    
    /** Start time. */
    itkSetMacro(StartTime, float);
    itkGetMacro(StartTime, float);

    /** Finish time. */
    itkSetMacro(FinishTime, float);
    itkGetMacro(FinishTime, float);

    /** Delta time. */
    itkSetMacro(DeltaTime, float);
    itkGetMacro(DeltaTime, float);
    
    /** Flag to control thickness calculation. Default false=off. */
    itkSetMacro(CalculateThickness, bool);
    itkGetMacro(CalculateThickness, bool);

    /** Get the maximum thickness after the last update. */
    itkGetMacro(MaxThickness, float);

    /** Get the maximum displacement after the last update. */
    itkGetMacro(MaxDisplacement, float);

    /** Get the field energy, the sum of the Euclidean norm of each velocity vector. */
    itkGetMacro(FieldEnergy, float);

    /** Set a binary mask such that only voxels within mask are integrated. The mask image memory is managed externally to this class. */
    void SetVoxelsToIntegrateMaskImage(MaskImageType* image) { m_MaskImage = image; }
    
    /** Set a distance mask, which is used to stop integrating if euclidean distance is greater than this mask value. The mask image memory is managed externally to this class. */
    void SetMaxDistanceMaskImage(ThicknessImageType* maxDistance) { m_MaxDistanceMask = maxDistance; }

    /** Set the grey white interface mask. The grey white interface mask memory is managed externally to this class. */
    void SetGreyWhiteInterfaceMaskImage(MaskImageType* greyWhiteInterface) { m_GreyWhiteInterface = greyWhiteInterface; }
    
    /** Get the calculated thickness image. */
    ThicknessImageType* GetCalculatedThicknessImage() { return m_ThicknessImage; }
    
  protected:
    
    FourthOrderRungeKuttaVelocityFieldIntegrationFilter();
    ~FourthOrderRungeKuttaVelocityFieldIntegrationFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;
    
    // Called before threaded bit.
    virtual void BeforeThreadedGenerateData();
    
    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData(const DisplacementImageRegionType &regionForThread, int);
    
    // Called after threaded bit.
    virtual void AfterThreadedGenerateData();
    
  private:
    
    /**
     * Prohibited copy and assignment. 
     */
    FourthOrderRungeKuttaVelocityFieldIntegrationFilter(const Self&); 
    void operator=(const Self&); 
    
    /** The start time of the integration. */
    float m_StartTime;
    
    /** The finish time of the integration. */
    float m_FinishTime;
    
    /** Delta T. */
    float m_DeltaTime;

    /** Calculated after the main integration. */
    float m_MaxThickness;

    /** Calculated after the main integration. */
    float m_MaxDisplacement;

    /** Gets the field energy. ie. sum of euclidean norm of each velocity vector. */
    float m_FieldEnergy;

    /** To interpolate the time varying velocity field. */
    TimeVaryingVelocityFieldInterpolatorPointer m_TimeVaryingVelocityFieldInterpolator;

    /** To interpolate the max distance image. */
    ThicknessImageInterpolatorPointer m_MaxDistanceImageInterpolator;

    /** Mask of max distance priors, we stop integrating if distance is > than the mask prior. */
    ThicknessImagePointer m_MaxDistanceMask;
    
    /** Mask image, so we only integrate within masked area. */
    MaskImagePointer m_MaskImage;
    
    /** This is the grey white interface that we propogate when calculating thickness. */
    MaskImagePointer m_GreyWhiteInterface;
    
    /** Flag to turn on/off thickness calculation. Default false. */
    bool m_CalculateThickness;
    
    /** To store a Euclidean thickness image. */
    ThicknessImagePointer m_ThicknessImage;
    
    /** Use to calculate Euclidean thickness. */
    ThicknessImagePointer m_HitsImage;
    
    /** Makes sure the output image has the right direction, spacing, origin, based on input. */
    virtual void GenerateOutputInformation();

    /** This will integrate the whole volume. */
    void IntegrateRegion(const DisplacementImageRegionType &regionForThread, float directionOverride, bool writeToOutput, bool writeToThicknessImage);
    
    /** Method to integrate a single point. */
    void IntegratePoint(const float& startTime, 
        const float& endTime, 
        const float& direction, 
        const unsigned int& numberOfTimePoints,
        const DisplacementImageIndexType& index,
        const TimeVaryingVelocityImageType* velocityField,
        const bool& writeToThicknessImage,
        DisplacementPixelType& displacement);
    
}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFourthOrderRungeKuttaVelocityFieldIntegrationFilter.txx"
#endif

#endif
