#include <string>
#include <cstdlib>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include "itkMIDASMorphologicalSegmentorLargestConnectedComponentImageFilter.h"

using namespace std;

struct _Test {
	string Name;
	unsigned short BgColour;
	bool Is2D;
};

template <const unsigned int t_Dim>
bool _RunTest(const _Test &test) {
	typedef itk::Image<unsigned char, t_Dim> __InputImgType;
	typedef itk::Image<unsigned short, t_Dim> __RegionImgType;
	typedef itk::ImageFileReader<__InputImgType> __LoaderType;
	typedef itk::MIDASMorphologicalSegmentorLargestConnectedComponentImageFilter<__InputImgType, __RegionImgType> __FilterType;

	const string testCaseFilename = test.Name + (t_Dim == 2? ".png" : ".nii");
	const string refFilename = test.Name + "_largest_region" + (t_Dim == 2? ".png" : ".nii");

	typename __LoaderType::Pointer sp_loader;
	typename __FilterType::Pointer sp_filter;

	sp_loader = __LoaderType::New();
	sp_loader->SetFileName(testCaseFilename);
  
	sp_filter = __FilterType::New();
	sp_filter->SetInputBackgroundValue(test.BgColour);
	sp_filter->SetOutputBackgroundValue(rand());
	do {
		sp_filter->SetOutputForegroundValue(rand());
	} while (sp_filter->GetOutputForegroundValue() == sp_filter->GetOutputBackgroundValue());
	sp_filter->SetInput(sp_loader->GetOutput());
	sp_filter->Update();

	sp_loader = __LoaderType::New();
	sp_loader->SetFileName(refFilename);
	sp_loader->Update();

	if (sp_filter->GetOutput()->GetLargestPossibleRegion().GetSize() == sp_loader->GetOutput()->GetLargestPossibleRegion().GetSize()
			&& sp_filter->GetOutput()->GetOrigin() == sp_loader->GetOutput()->GetOrigin()
			&& sp_filter->GetOutput()->GetSpacing() == sp_loader->GetOutput()->GetSpacing()) {
		itk::ImageRegionConstIterator<__InputImgType> ic_refPx(sp_loader->GetOutput(), sp_loader->GetOutput()->GetLargestPossibleRegion());
		itk::ImageRegionConstIterator<__RegionImgType> ic_testPx(sp_filter->GetOutput(), sp_filter->GetOutput()->GetLargestPossibleRegion());

		for (ic_refPx.GoToBegin(), ic_testPx.GoToBegin(); !ic_refPx.IsAtEnd(); ++ic_refPx, ++ic_testPx) {
			if ((ic_refPx.Get() == sp_filter->GetInputBackgroundValue() && ic_testPx.Get() != sp_filter->GetOutputBackgroundValue())
					|| (ic_refPx.Get() != sp_filter->GetInputBackgroundValue() && ic_testPx.Get() == sp_filter->GetOutputBackgroundValue())) {
				return false;
			}
		}

		return true;
	} else {
		return false;
	}
} /* _RunTest */

int itkMIDASMorphologicalSegmentorLargestConnectedComponentFilterTest(int argc, char *argv[]) {
	_Test tests[10];	
	_Test const *pc_test;

	{
		int iIdx, tIdx;

		for (iIdx = 1, tIdx = 0 && tIdx < 9; iIdx < argc; tIdx++) {
			tests[tIdx].Name = argv[iIdx++];
			tests[tIdx].Is2D = (*argv[iIdx++] == '2');
		}
		tests[tIdx].Name = "";
	}
		 
	for (pc_test = tests; pc_test->Name.length() > 0; pc_test++) {
		if (pc_test->Is2D) {
			if (!_RunTest<2>(*pc_test)) return EXIT_FAILURE;
		} else {
			if (!_RunTest<3>(*pc_test)) return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}
