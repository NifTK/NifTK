/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $
 Revision          : $Revision: 7310 $
 Last modified by  : $Author: ad $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkAddUpdateToTimeVaryingVelocityFieldFilter_h
#define __itkAddUpdateToTimeVaryingVelocityFieldFilter_h

#include "itkVector.h"
#include "itkImage.h"
#include "itkInPlaceImageFilter.h"

namespace itk {

/**
 * \class AddUpdateToTimeVaryingVelocityFieldFilter
 * \brief Adds a vector displacement field to a time varying velocity field.
 *
 * This class should be run InPlace to save memory. Also, you can set two update fields.
 * If we are integrating from a fixed image to a moving image, using symmetric registration,
 * then you will have two transformations. The first is a forward transformation from
 * F to the midpoint, and the second is an inverse transformation from the midpoint to M.
 * So you can set SetUpdateImage and SetUpdateInverseImage. The memory for these two images
 * should be managed outside of this class (i.e. its not the responsibility of this class).
 */
template < typename TScalarType, unsigned int NDimensions = 3>
class ITK_EXPORT AddUpdateToTimeVaryingVelocityFieldFilter :
public InPlaceImageFilter<
                           Image< Vector<TScalarType, NDimensions>,  NDimensions + 1>, // Input image is time varying velocity field.
                           Image< Vector<TScalarType, NDimensions>,  NDimensions + 1>  // Output image is time varying velocity field.
                         >
{

public:

    /** Standard ITK "Self" typedefs. */
    typedef AddUpdateToTimeVaryingVelocityFieldFilter                                      Self;
    typedef ImageToImageFilter<Image< Vector<TScalarType, NDimensions>,  NDimensions + 1>,
                               Image< Vector<TScalarType, NDimensions>,  NDimensions + 1>
                              >                                                            Superclass;
    typedef SmartPointer<Self>                                                             Pointer;
    typedef SmartPointer<const Self>                                                       ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(AddUpdateToTimeVaryingVelocityFieldFilter, ImageToImageFilter);

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

    typedef Vector< TScalarType, NDimensions >                                            UpdatePixelType;
    typedef Image< UpdatePixelType, NDimensions>                                          UpdateImageType;
    typedef typename UpdateImageType::Pointer                                             UpdateImagePointer;
    typedef typename UpdateImageType::RegionType                                          UpdateImageRegionType;
    typedef typename UpdateImageType::IndexType                                           UpdateImageIndexType;
    typedef typename UpdateImageType::SizeType                                            UpdateImageSizeType;
    typedef typename UpdateImageType::PointType                                           UpdateImagePointType;
    typedef typename UpdateImageType::SpacingType                                         UpdateImageSpacingType;
    typedef typename UpdateImageType::DirectionType                                       UpdateImageDirectionType;

    /** Set/Get the time point that the update corresponds to. */
    itkSetMacro(TimePoint, float);
    itkGetMacro(TimePoint, float);

    /** Set/Get the scale factor, which is multiplied with update vector. Default 1. */
    itkSetMacro(ScaleFactor, float);
    itkGetMacro(ScaleFactor, float);

    /** Set/Get the OverWrite flag, if true, we don't add to existing field, we just overwrite it. Default false. */
    itkSetMacro(OverWrite, bool);
    itkGetMacro(OverWrite, bool);

    /** Set the update image. */
    void SetUpdateImage(UpdateImageType* i) { m_UpdateImage = i; }

    /** Set the inverse image. */
    void SetUpdateInverseImage(UpdateImageType *i) { m_UpdateInverseImage = i; }

    /** Get the max deformation. */
    itkGetMacro(MaxDeformation, float);

protected:

    AddUpdateToTimeVaryingVelocityFieldFilter();
    ~AddUpdateToTimeVaryingVelocityFieldFilter() {};
    void PrintSelf(std::ostream& os, Indent indent) const;

    // Called before threaded bit.
    virtual void BeforeThreadedGenerateData();

    // Called after threaded bit.
    virtual void AfterThreadedGenerateData();

    // The main method to implement in derived classes, note, its threaded.
    virtual void ThreadedGenerateData(const TimeVaryingVelocityRegionType &regionForThread, int);

private:

    /** If true, we don't add, we just copy or overwrite. */
    bool m_OverWrite;

    /** The current timepoint. */
    float m_TimePoint;

    /** Scale factor, so we can scale update field. */
    float m_ScaleFactor;

    /** Max deformation, updated after each iteration. */
    float m_MaxDeformation;

    /** The update to be added. */
    UpdateImagePointer m_UpdateImage;

    /** The inverse update. */
    UpdateImagePointer m_UpdateInverseImage;

}; // end class

} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAddUpdateToTimeVaryingVelocityFieldFilter.txx"
#endif

#endif
