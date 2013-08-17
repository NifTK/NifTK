/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <string>
#include <vector>
#include <typeinfo>
#include <cstdlib>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryContourImageFilter.h>
#include <itkAndImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkUnaryFunctorImageFilter.h>
#include <itkConnectedThresholdImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkSubtractImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkPointSet.h>
#include <itkMIDASRegionGrowingImageFilter.h>

using namespace std;

/***********************************************************
 * This test deprecated. This test assumes that the
 * region growing goes up to and not including the boundary.
 * As of 2012/05/29, we do include boundary, and this makes
 * it a bit difficult to generate the images in this way.
 **********************************************************/

/*
 * Replaces true with trueValue, false with falseValue
 */
template<typename TOutputPixelType>
class _ReplaceIntensityFunctor {
private:
	TOutputPixelType m_TrueValue, m_FalseValue;

public:
	TOutputPixelType operator()(const bool pxVal) const {
		return pxVal ? m_TrueValue : m_FalseValue;
	}

	bool operator!=(const _ReplaceIntensityFunctor &other) {
		return this != &other;
	}

	bool operator==(const _ReplaceIntensityFunctor &other) {
		return this == &other;
	}

public:
	_ReplaceIntensityFunctor() :
		m_TrueValue(-1), m_FalseValue(-1) {
	}

	_ReplaceIntensityFunctor(const TOutputPixelType trueVal,
			const TOutputPixelType falseVal) :
		m_TrueValue(trueVal), m_FalseValue(falseVal) {
	}
};

template<class TImg>
void _LoadImage(typename TImg::Pointer &rsp_output, const string &path) {
	typedef itk::ImageFileReader<TImg> __ImgLoader;

	typename __ImgLoader::Pointer sp_loader;

	sp_loader = __ImgLoader::New();
	sp_loader->SetFileName(path);
	sp_loader->Update();

	rsp_output = sp_loader->GetOutput();
}

template<class TImg>
void _DumpImg(const string &path, const TImg &img) {
	static bool hasCast = false;

	try {
		typedef itk::ImageFileWriter<TImg> __ImgWriter;

		typename __ImgWriter::Pointer sp_writer;

		sp_writer = __ImgWriter::New();
		sp_writer->SetFileName(path);
		sp_writer->SetInput(&img);
		sp_writer->Update();
		hasCast = false;
	} catch (itk::ExceptionObject &r_ex) {
		if (!hasCast) {
			typedef itk::CastImageFilter<TImg, itk::Image<unsigned short, 2> >
					__CastFilter;

			typename __CastFilter::Pointer sp_cast;

			hasCast = true;
			sp_cast = __CastFilter::New();
			sp_cast->SetInput(&img);
			_DumpImg(path, *sp_cast->GetOutput());
		} else
			throw;
	}
}

template<class TGSImg, class TBinImg, class TPointSet>
void _GenerateImages(typename TGSImg::Pointer &rsp_gsImg,
		typename TBinImg::Pointer &rsp_contimg,
		typename TBinImg::Pointer &rsp_groundtruth, TPointSet &r_seeds,
		const string &gsImgPath, const string &regionImgPath,
		const typename TGSImg::PixelType lowerThrsh,
		const typename TGSImg::PixelType upperThrsh,
		const typename TBinImg::PixelType trueval,
		const typename TBinImg::PixelType falseval)
{
	typedef typename TGSImg::PixelType __GSPixelType;
	typedef typename TBinImg::PixelType __BinPixelType;
	typedef itk::Image<bool, 2> __RegionImageType;

	typename __RegionImageType::Pointer sp_regionImg, sp_contour;

	// Load region image and grey scale image.
	try {
		sp_regionImg = NULL, rsp_gsImg = NULL;
		_LoadImage<__RegionImageType > (sp_regionImg, regionImgPath);
		_LoadImage<TGSImg> (rsp_gsImg, gsImgPath);
	} catch (itk::ExceptionObject &r_ex) {
		cerr << "ITK: Exception " << r_ex;
		return;
	}

	// Extract contour, which is edge pixel of region image.
	{
    typedef itk::BinaryContourImageFilter<
      __RegionImageType, __RegionImageType> __ContourExtractor; typename __ContourExtractor::Pointer sp_extractor;

    sp_extractor = __ContourExtractor::New();
    sp_extractor->SetInput(sp_regionImg);
    sp_extractor->Update();
    sp_contour = sp_extractor->GetOutput();
  }

  {
    typedef itk::SubtractImageFilter<__RegionImageType, __RegionImageType> __SubFilter;

    typename __SubFilter::Pointer sp_filter;

    sp_filter = __SubFilter::New();
    sp_filter->SetInput1(sp_regionImg);
    sp_filter->SetInput2(sp_contour);
    sp_filter->Update();
    sp_regionImg = sp_filter->GetOutput();
  }

	// Swaps true for false, so will produce inverse contour image.
	{
	  typedef itk::UnaryFunctorImageFilter<__RegionImageType, TBinImg, _ReplaceIntensityFunctor<__BinPixelType> > __ReplaceIntensityFilter;

		typename __ReplaceIntensityFilter::Pointer sp_filter;
				_ReplaceIntensityFunctor<__BinPixelType> functor(trueval, falseval);

		sp_filter = __ReplaceIntensityFilter::New();
		sp_filter->SetInput(sp_contour);
		sp_filter->SetFunctor(functor);
		sp_filter->Update();
		rsp_contimg = sp_filter->GetOutput();
	}

	// Randomly pick pixels until we have 3-10 seed points
	{
    const int numSeeds = rand()%8 + 3;
    const typename __RegionImageType::RegionType::SizeType imgSize = sp_regionImg->GetLargestPossibleRegion().GetSize();

    r_seeds.GetPoints()->Initialize();
    while ((int)r_seeds.GetPoints()->Size() < numSeeds) {

      typename __RegionImageType::IndexType imgIdx;
      __GSPixelType gsVal;

      imgIdx[0] = rand()%imgSize[0];
      imgIdx[1] = rand()%imgSize[1];

      if (sp_regionImg->GetPixel(imgIdx) && (gsVal = rsp_gsImg->GetPixel(imgIdx)) >= lowerThrsh && gsVal <= upperThrsh) {
        typename TPointSet::PointType pt;

        sp_regionImg->TransformIndexToPhysicalPoint(imgIdx, pt);
        r_seeds.GetPoints()->InsertElement(r_seeds.GetPoints()->Size(), pt);
      }
    }
  }

  {
    __RegionImageType::Pointer sp_connectedPatches;

    /*
     * Generate ground-truth:
     * Threshold with connected thrsh filter (use seed points)
     * AND w/ region img.
     * NEEDS TO BE DONE THIS WAY SINCE WE CAN HAVE REGIONS W/O SEEDS!
     */
    {
      // Takes a random list of seeds,
      // converts each seed to index.
      // If the seed index is foreground in region image, and within intensity range in grey scale image, adds to list.

      typedef itk::ConnectedThresholdImageFilter<TGSImg, __RegionImageType> __ThrshFilter;
      typedef itk::AndImageFilter<__RegionImageType, __RegionImageType> __AndFilter;

      typename __ThrshFilter::Pointer sp_filter;
      typename __AndFilter::Pointer sp_andFilter;
      typename TPointSet::PointsContainer::ConstIterator ic_seed;

      sp_filter = __ThrshFilter::New();
      sp_filter->SetInput(rsp_gsImg);
      sp_filter->SetLower(lowerThrsh);
      sp_filter->SetUpper(upperThrsh);

      for (ic_seed = r_seeds.GetPoints()->Begin(); ic_seed != r_seeds.GetPoints()->End(); ++ic_seed) {

        typename TGSImg::IndexType imgIdx;

        rsp_gsImg->TransformPhysicalPointToIndex(ic_seed->Value(), imgIdx);
        assert(sp_regionImg->GetPixel(imgIdx) && rsp_gsImg->GetPixel(imgIdx) >= lowerThrsh && rsp_gsImg->GetPixel(imgIdx) <= upperThrsh);
        sp_filter->AddSeed(imgIdx);
      }

      sp_andFilter = __AndFilter::New();
      sp_andFilter->SetInput1(sp_regionImg);
      sp_andFilter->SetInput2(sp_filter->GetOutput());
      sp_andFilter->Update();
      sp_connectedPatches = sp_andFilter->GetOutput();
    }

    {
      typedef itk::ConnectedThresholdImageFilter<__RegionImageType, __RegionImageType> __ThrshFilter;

      typename __ThrshFilter::Pointer sp_filter;
      typename TPointSet::PointsContainer::ConstIterator ic_seed;

      sp_filter = __ThrshFilter::New();
      sp_filter->SetInput(sp_connectedPatches);
      sp_filter->SetLower(true);
      sp_filter->SetUpper(true);
      for (ic_seed = r_seeds.GetPoints()->Begin(); ic_seed != r_seeds.GetPoints()->End(); ++ic_seed) {

        typename TGSImg::IndexType imgIdx;

        rsp_gsImg->TransformPhysicalPointToIndex(ic_seed->Value(), imgIdx);
        assert(sp_regionImg->GetPixel(imgIdx) && rsp_gsImg->GetPixel(imgIdx) >= lowerThrsh && rsp_gsImg->GetPixel(imgIdx) <= upperThrsh);
        sp_filter->AddSeed(imgIdx);
      }
      sp_filter->Update();
      sp_connectedPatches = sp_filter->GetOutput();
    }

    rsp_groundtruth = TBinImg::New();
    rsp_groundtruth->SetRegions(rsp_contimg->GetLargestPossibleRegion());
    rsp_groundtruth->Allocate();
    rsp_groundtruth->SetSpacing(sp_regionImg->GetSpacing());
    rsp_groundtruth->SetOrigin(sp_regionImg->GetOrigin());
    rsp_groundtruth->SetDirection(sp_regionImg->GetDirection());

    {
      itk::ImageRegionConstIterator<__RegionImageType> ic_regionPx(sp_regionImg, sp_regionImg->GetLargestPossibleRegion());
      itk::ImageRegionConstIterator<__RegionImageType> ic_patchesPx(sp_connectedPatches, sp_connectedPatches->GetLargestPossibleRegion());
      itk::ImageRegionIterator<TBinImg> i_groundtruthPx(rsp_groundtruth, rsp_groundtruth->GetLargestPossibleRegion());

      for (ic_regionPx.GoToBegin(), ic_patchesPx.GoToBegin(), i_groundtruthPx.GoToBegin();
          !ic_regionPx.IsAtEnd();
          ++ic_regionPx, ++ic_patchesPx, ++i_groundtruthPx)
      {
        if (ic_regionPx.Value() && ic_patchesPx.Value()) i_groundtruthPx.Set(trueval);
        else i_groundtruthPx.Set(falseval);
      }
    }
  } // end generate ground truth
}

template<class TGSImg, class TBinImg>
int _TestImage2(const string &gsImgPath, const string &regionImgPath,
		const typename TGSImg::PixelType lowerThrsh,
		const typename TGSImg::PixelType upperThrsh) {

	typedef itk::PointSet<float, 2> __PointSet;
	typedef typename TBinImg::PixelType __BinPixelType;

	typename TGSImg::Pointer sp_gsImg;
	typename TBinImg::Pointer sp_contImg, sp_groundTruth;
	typename __PointSet::Pointer sp_seeds;
	__BinPixelType trueval, falseval;

	if (typeid(__BinPixelType) != typeid(bool)) {
		do {
			trueval = rand();
			falseval = rand();
		}while (trueval == falseval);
	} else trueval = rand(), falseval = !trueval;

	sp_seeds = __PointSet::New();
	_GenerateImages<TGSImg, TBinImg, __PointSet>(sp_gsImg, sp_contImg, sp_groundTruth, *sp_seeds, gsImgPath, regionImgPath, lowerThrsh, upperThrsh, trueval, falseval);

	try {
		typedef itk::MIDASRegionGrowingImageFilter<TGSImg, TBinImg, __PointSet> __RGFilter;

		typename __RGFilter::Pointer sp_filter;

		sp_filter = __RGFilter::New();
		sp_filter->SetInput(sp_gsImg);
		sp_filter->SetForegroundValue(trueval);
		sp_filter->SetBackgroundValue(falseval);
		sp_filter->SetLowerThreshold(lowerThrsh);
		sp_filter->SetUpperThreshold(upperThrsh);
		sp_filter->SetSeedPoints(*sp_seeds);
		sp_filter->SetContourImage(sp_contImg);
		sp_filter->Update();

		if (sp_filter->GetOutput()->GetOrigin() != sp_groundTruth->GetOrigin()
		    || sp_filter->GetOutput()->GetSpacing() != sp_groundTruth->GetSpacing()
		    || sp_filter->GetOutput()->GetLargestPossibleRegion().GetSize() != sp_groundTruth->GetLargestPossibleRegion().GetSize()) {
			cerr << "Spatial mismatch\n";
			return EXIT_FAILURE;
		}

		{

			itk::ImageRegionConstIterator<TBinImg> ic_testPx(sp_filter->GetOutput(), sp_filter->GetOutput()->GetLargestPossibleRegion());
			itk::ImageRegionConstIterator<TBinImg> ic_refPx(sp_groundTruth, sp_groundTruth->GetLargestPossibleRegion());

			for (ic_testPx.GoToBegin(), ic_refPx.GoToBegin(); !ic_testPx.IsAtEnd(); ++ic_testPx, ++ic_refPx) {
				assert(ic_testPx.GetIndex() == ic_refPx.GetIndex());
				if (ic_testPx.Get() != ic_refPx.Get()) {
					cout << ic_testPx.Get() << endl;
					cout << ic_refPx.Get() << endl;
					return EXIT_FAILURE;
				}
			}

			return EXIT_SUCCESS;
		}
	} catch (itk::ExceptionObject &r_ex) {
		cerr << "Caught ITK exception:" << r_ex.what() << endl;

		return EXIT_FAILURE;
	}
}

int itkMIDASRegionGrowingImageFilterTest(int argc, char *argv[]) {
	int eVal, inputIdx;

	eVal = 0;
	for (inputIdx = 1; inputIdx < argc && !eVal; inputIdx += 4) {
		const std::string gsImgPath = argv[inputIdx];
		const std::string regImgPath = argv[inputIdx + 1];
		const int lowerThrsh = atoi(argv[inputIdx + 2]);
		const int upperThrsh = atoi(argv[inputIdx + 3]);

		eVal |= _TestImage2<itk::Image<unsigned short, 2>, itk::Image<bool, 2> >           (gsImgPath, regImgPath, lowerThrsh, upperThrsh);
		eVal |= _TestImage2<itk::Image<unsigned short, 2>, itk::Image<unsigned short, 2> > (gsImgPath, regImgPath, lowerThrsh, upperThrsh);
		eVal |= _TestImage2<itk::Image<float, 2>, itk::Image<unsigned short, 2> >          (gsImgPath, regImgPath, lowerThrsh, upperThrsh);
	}

	return eVal;
}
