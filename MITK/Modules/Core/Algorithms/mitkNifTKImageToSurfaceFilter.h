/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkNifTKNifTKImageToSurfaceFilter_h
#define mitkNifTKNifTKImageToSurfaceFilter_h

#include "niftkCoreExports.h"

#include <mitkCommon.h>
#include <mitkSurfaceSource.h>
#include <mitkSurface.h>
#include <mitkImage.h>

#include <vtkPolyData.h>
#include <vtkImageData.h>
#include <vtkSmoothPolyDataFilter.h>
#include <vtkMarchingCubes.h>
#include <vtkSmartPointer.h>

#include "mitkBasicVertex.h"
#include "mitkBasicTriangle.h"
#include "mitkNifTKCMC33.h"
#include "mitkBasicImageProcessor.h"

namespace mitk {
  /**
  * @brief Converts pixel data to surface data by using a threshold
  * The mitkNifTKImageToSurfaceFilter is used to create a new surface out of an mitk image. The 
  * filter uses a threshold to define the surface. It can use two algorithms for the extraction: 
  * the vtkMarchingCube algorithm (default) and the Corrected Marching Cubes 33 method:
  * Custodio, Lis, et al. "Practical considerations on Marching Cubes 33 topological correctness."
  * Computers & Graphics 37.7 (2013): 840-850.
  *
  * By default a vtkPolyData surface based on an input threshold for the input image will be created. 
  * Optionally it is possible to:
  *    - Smooth the input image before the extraction take place (Median, Gaussian smoothing methods from VTK)
  *    - Reduce the number of triangles/polygons (Decimate Pro or Quadric decimation methods from VTK)
  *    - Smooth the surface-data (VTK method or various flavors of Taubin smoothing from Meshlab)
  *    - Perform a "Small object removal" that eliminates small, non connected surface fragments that have 
  *      fewer number of ploygons than a selected threshold
  *
  * The resulting vtk-surface has the same size as the input image.
  *
  * @ingroup ImageFilters
  * @ingroup Process
  */

  class NIFTKCORE_EXPORT NifTKImageToSurfaceFilter : public mitk::SurfaceSource
  {
  public:

    enum SurfaceExtractionMethod {StandardExtractor, EnhancedCPUExtractor, GPUExtractor};
    enum SurfaceDecimationMethod {NoDecimation, DecimatePro, QuadricVTK, Quadric, QuadricTri, Melax, ShortestEdge};
    enum SurfaceSmoothingMethod  {NoSurfaceSmoothing, TaubinSmoothing, CurvatureNormalSmooth, InverseEdgeLengthSmooth, WindowedSincSmoothing, StandardVTKSmoothing};
    enum InputSmoothingMethod    {NoInputSmoothing, GaussianSmoothing, MedianSmoothing};

    mitkClassMacro(NifTKImageToSurfaceFilter, SurfaceSource);
    itkNewMacro(Self);

    /// \brief For each image time slice a surface will be created. This method is called by Update().
    virtual void GenerateData();

    /// \brief Initializes the output information ( i.e. the geometry information ) of the output of the filter
    virtual void GenerateOutputInformation();

    /// \brief Returns a const reference to the input image (e.g. the original input image that ist used to create the surface)
    const mitk::Image *GetInput(void);

    /// \brief Set the source image to create a surface for this filter class. As input every mitk 3D or 3D+t image can be used.
    using itk::ProcessObject::SetInput;
    virtual void SetInput(const mitk::Image *image);


    /**
    * Threshold that is used to create the surface. All pixel in the input image that are higher than that
    * value will be considered in the surface. The threshold referees to
    * vtkMarchingCube. Default value is 1. See also SetThreshold (ScalarType _arg)
    */
    /// \brief Set the Marching Cubes threshold value. Threshold can be manipulated by inherited classes.
    itkSetMacro(Threshold, ScalarType);

    /// \brief Get the Marching Cubes threshold value. Threshold can be manipulated by inherited classes.
    itkGetConstMacro(Threshold, ScalarType);

    /// \brief Get the state of the input image smoothing mode.
    itkGetConstMacro(SurfaceExtractionType, SurfaceExtractionMethod);

    /// \brief Sets the input image smoothing mode (Gaussian / Median)
    itkSetMacro(SurfaceExtractionType, SurfaceExtractionMethod);



    /// \brief Get the state of the input image smoothing mode.
    itkGetConstMacro(InputSmoothingType, InputSmoothingMethod);

    /// \brief Sets the input image smoothing mode (Gaussian / Median)
    itkSetMacro(InputSmoothingType, InputSmoothingMethod);

    /// \brief Enables input image smoothing. The preferred method can be specified, along with the radius.
    itkSetMacro(PerformInputSmoothing,bool);

    /// \brief Enable/Disable input image smoothing.
    itkBooleanMacro(PerformInputSmoothing);

    /// \brief Returns if input image smoothing is enabled
    itkGetConstMacro(PerformInputSmoothing,bool);

    /// \brief Sets the smoothing radius of the input image smoothing method
    itkSetMacro(InputSmoothingRadius, float);

    /// \brief Returns the smoothing radius of the input image smoothing method
    itkGetConstMacro(InputSmoothingRadius, float);

    /// \brief Sets the number of iterations for the surface smoothing method
    itkSetMacro(InputSmoothingIterations, int);

    /// \brief Returns the number of iterations for the surface smoothing method
    itkGetConstMacro(InputSmoothingIterations, int);



    /// \brief Get the state of the surface smoothing mode.
    itkGetConstMacro(SurfaceSmoothingType, SurfaceSmoothingMethod);

    /// \brief Sets the surface smoothing mode (Laplacian / Taubin)
    itkSetMacro(SurfaceSmoothingType, SurfaceSmoothingMethod);

    /// \brief Enables surface smoothing. The preferred method can be specified, along with the parameters.
    itkSetMacro(PerformSurfaceSmoothing,bool);

    /// \brief Enable/Disable surface smoothing.
    itkBooleanMacro(PerformSurfaceSmoothing);

    /// \brief Returns if surface smoothing is enabled
    itkGetConstMacro(PerformSurfaceSmoothing,bool);

    /// \brief Sets the smoothing radius of the surface smoothing method
    itkSetMacro(SurfaceSmoothingRadius, float);

    /// \brief Returns the smoothing radius of the surface smoothing method
    itkGetConstMacro(SurfaceSmoothingRadius, float);

    /// \brief Sets the number of iterations for the input image smoothing method
    itkSetMacro(SurfaceSmoothingIterations, int);

    /// \brief Returns the number of iterations for the input image smoothing method
    itkGetConstMacro(SurfaceSmoothingIterations, int);



    /// \brief Get the state of decimation mode to reduce the number of triangles in the surface represantation.
    itkGetConstMacro(SurfaceDecimationType, SurfaceDecimationMethod);

    /// \brief Sets the surface decimation method to reduce the number of triangles in the mesh and produce a good approximation to the original image.
    itkSetMacro(SurfaceDecimationType, SurfaceDecimationMethod);

    /// \brief Enables surface decimation. The preferred method can be specified, along with the radius.
    itkSetMacro(PerformSurfaceDecimation,bool);

    /// \brief Enable/Disable surface decimation.
    itkBooleanMacro(PerformSurfaceDecimation);

    /// \brief Returns if surface decimation is enabled
    itkGetConstMacro(PerformSurfaceDecimation,bool);

    /// \brief Set desired  amount of reduction of triangles in the range from 0.0 to 1.0. For example 0.9 will reduce the data set to 10%.
    itkSetMacro(TargetReduction, float);

    /// \brief Returns the reduction factor as a float value
    itkGetConstMacro(TargetReduction, float);

    /// \brief Set desired downsampling ratio that is applied to the image before surface extraction
    itkSetMacro(SamplingRatio, double);

    /// \brief Returns the downsampling ratio
    itkGetConstMacro(SamplingRatio, double);


    /// \brief Enable/Disable surface cleaning and small object removal.
    itkSetMacro(PerformSurfaceCleaning,bool);
    /// \brief Returns true if surface cleaning is enabled
    itkGetConstMacro(PerformSurfaceCleaning,bool);

    /// \brief Sets the number threshold for small object removal
    itkSetMacro(SurfaceCleaningThreshold, int);

    /// \brief Returns the threshold for small object removal
    itkGetConstMacro(SurfaceCleaningThreshold, int);


    /// \brief Transforms a point by a 4x4 matrix
    template <class T1, class T2, class T3>
    inline void mitkVtkLinearTransformPoint(T1 matrix[4][4], T2 in[3], T3 out[3])
    {
      T3 x = matrix[0][0]*in[0]+matrix[0][1]*in[1]+matrix[0][2]*in[2]+matrix[0][3];
      T3 y = matrix[1][0]*in[0]+matrix[1][1]*in[1]+matrix[1][2]*in[2]+matrix[1][3];
      T3 z = matrix[2][0]*in[0]+matrix[2][1]*in[1]+matrix[2][2]*in[2]+matrix[2][3];
      out[0] = x;
      out[1] = y;
      out[2] = z;
    }

  protected:

    /// \brief Default Constructor
    NifTKImageToSurfaceFilter();

    /// \brief Destructor
    virtual ~NifTKImageToSurfaceFilter();

    /**
    * With the given threshold vtkMarchingCube creates the surface. By default a vtkPolyData surface based
    * on a threshold of the input image will be created. Optionally it is possible to reduce the number of
    * triangles/polygones [SetDecimate(mitk::NifTKImageToSurfaceFilter::DecimatePro) and SetTargetReduction (float _arg)]
    * or smooth the data [SetSmooth(true), SetSmoothingIteration(int smoothIteration) and SetSmoothRelaxation(float smoothRelaxation)].
    *
    * @param time selected slice or "0" for single
    * @param *mitk::Image input image
    * @param *mitk::Surface output
    * @param threshold can be different from SetThreshold()
    */
    void CreateSurface(mitk::Image *image, mitk::Surface *surface);

    /// \brief Creates a surface using the VTK Marching Cubes method.
    void VTKSurfaceExtraction(mitk::Image *image, vtkSmartPointer<vtkPolyData> vtkSurface);
    /// \brief Creates a surface using the Corrected Marching Cubes 33 method (C-MC33)
    void CMC33SurfaceExtraction(mitk::Image *inputImage, mitk::MeshData * meshData);

    /// \brief Performs smoothing of the triangle mesh with the previously selected method (SurfaceSmoothingType)
    void MeshSmoothing(MeshData * meshData);

    /// \brief Performs smoothing of the triangle mesh with the VTK smoothing method
    void SurfaceSmoothingVTK(vtkSmartPointer<vtkPolyData> vtkSurface);
    /// \brief Computes smooth (interpolated) normals for the surface
    void ComputeSmoothNormals(MeshData * meshData);

    /// \brief Utility function for a. Flipping Normals b. Reorienting Faces c. Fixing Cracks
    void EditSurface(MeshData * meshData, bool recomputeNormals, bool flipNormals, bool reorientFaces, bool fixCracks);
    
    /// \brief Creates a VTK Polydata structure from the set of provided vertices and triangles
    vtkSmartPointer<vtkPolyData> BuildVTKPolyData(MeshData * meshData);

    /**
    * Threshold that is used to create the surface. All pixel in the input image that are higher than that
    * value will be considered in the surface. Default value is 1. See also SetThreshold (ScalarType _arg)
    * */
    ScalarType              m_Threshold;

    SurfaceExtractionMethod m_SurfaceExtractionType;

    InputSmoothingMethod    m_InputSmoothingType;
    bool                    m_PerformInputSmoothing;
    int                     m_InputSmoothingIterations;
    float                   m_InputSmoothingRadius;

    SurfaceSmoothingMethod  m_SurfaceSmoothingType;
    bool                    m_PerformSurfaceSmoothing;
    int                     m_SurfaceSmoothingIterations;
    float                   m_SurfaceSmoothingRadius;

    SurfaceDecimationMethod m_SurfaceDecimationType;
    bool                    m_PerformSurfaceDecimation;
    double                  m_TargetReduction;

    bool                    m_PerformSurfaceCleaning;
    int                     m_SurfaceCleaningThreshold;

    bool                    m_VTKNormalCompute;
    bool                    m_FlipNormals;

    double                  m_SamplingRatio;

  };

} // namespace mitk

#endif //mitkNifTKNifTKImageToSurfaceFilter_h


