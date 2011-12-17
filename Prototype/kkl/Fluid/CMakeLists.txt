SET(NIFTK_APPLICATION_LINK_LIBRARIES 
  niftkcommon 
  niftkITK
  ${NIFTK_ITK_LIB}
  ${Boost_LIBRARIES} 
)

SET(NIFTK_FFTW_LINK_LIBRARIES
    ${NIFTK_APPLICATION_LINK_LIBRARIES}
    ${FFTW_LIBRARIES}
)
    
ADD_EXECUTABLE(niftkFluid niftkFluid.cxx )
TARGET_LINK_LIBRARIES(niftkFluid ${NIFTK_FFTW_LINK_LIBRARIES} )
INSTALL(TARGETS niftkFluid RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
    
#ADD_EXECUTABLE(niftkFluid2D niftkFluid2D.cxx )
#TARGET_LINK_LIBRARIES(niftkFluid2D ${NIFTK_FFTW_LINK_LIBRARIES} )
#INSTALL(TARGETS niftkFluid2D RUNTIME DESTINATION ${NIFTK_INSTALL_BIN_DIR} COMPONENT applications)
    
  
#################################################################################
# Fluid Tests
#################################################################################

#	SET(NIFTK_TEST_EXT_ITK_REGISTRATION_TOOLBOX_FLUID_LINK_LIBRARIES 
#  		niftkcommon 
#  		niftkITK 
#  		ITKCommon 
#  		ITKIO 
#  		ITKStatistics
#  		ITKAlgorithms
#  		${Boost_LIBRARIES}
#  		${FFTW_LIBRARIES}
#  	)

#	SET(REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS ${CXX_TEST_PATH}/FluidIntegrationTests)
	
#	ADD_TEST(Fluid-2D-1 ${REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS} --compare ${BASELINE}/inverseGrid.png ${TEMP_DIR}/FluidDeformableTransformTestInverse.png FluidDeformableTransformTest ${INPUT_DATA}/grid.png 256 256  ${TEMP_DIR}/FluidDeformableTransformTest.png ${TEMP_DIR}/FluidDeformableTransformTestInverse.png )
#	ADD_TEST(Fluid-PDE         ${REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS} FluidPDEFilterTest ${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png 1.50725 1.00013 0.0001)
#	ADD_TEST(Fluid-FluidVelocityToDeformationFilterTest   ${REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS} FluidVelocityToDeformationFilterTest )

#ADD_TEST(Fluid-Reg-1 ${REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS} itkFluidRegistrationTest 
#   	${INPUT_DATA}/fluid_fixed.png ${INPUT_DATA}/fluid_moving.png
#  		${TEMP_DIR}/fluid_forward_output.png ${TEMP_DIR}/fluid_forward_stretch.png 
#    	0.01 0.0 1 200 64 
#    	${TEMP_DIR}/fluid_forward_diff_after.png 
#    	${TEMP_DIR}/fluid_forward_diff_before.png 0.05 0 0)
#	ADD_TEST(Fluid-Reg-2 ${REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS} itkFluidRegistrationTest 
#    	${INPUT_DATA}/fluid_moving.png ${INPUT_DATA}/fluid_fixed.png 
#    	${TEMP_DIR}/fluid_backward_output.png ${TEMP_DIR}/fluid_backward_stretch.png 
#    	0.01 0.0 1 200 64 
#    	${TEMP_DIR}/fluid_backward_diff_after.png 
#    	${TEMP_DIR}/fluid_backward_diff_before.png 0.05 0 0)
#	ADD_TEST(Fluid-Reg-3 ${REGISTRATION_TOOLBOX_FLUID_INTEGRATION_TESTS} itkFluidRegistrationTest 
#    	${INPUT_DATA}/mr1.png ${INPUT_DATA}/mr2.png 
#    	${TEMP_DIR}/fluid_mr_forward_output.png ${TEMP_DIR}/fluid_mr_forward_stretch.png 
#    	0.01 0.0 0.4 200 64 
#    	${TEMP_DIR}/fluid_mr_forward_diff_after.png 
#    	${TEMP_DIR}/fluid_mr_forward_diff_before.png 0.05 1 1)
    
	#################################################################################
	# These tend to time out.... are they unit tests or regressions tests... ponder.
	#################################################################################

	#ADD_TEST(Fluid3D-Reg-1 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} itkFluidRegistration3DTest 
	#    ${INPUT_DATA}/fluid-3d-test-2.nii ${INPUT_DATA}/fluid-3d-test-1.nii
	#    ${TEMP_DIR}/fluid_forward_3D_output.nii ${TEMP_DIR}/fluid_3D_forward_stretch.hdr
	#    0.01 0.0 1 200 64 
	#    ${TEMP_DIR}/fluid_3D_forward_diff_after.hdr 
	#    ${TEMP_DIR}/fluid_3D_forward_diff_before.hdr 0.05)
    
	#ADD_TEST(Fluid3D-Reg-2 ${REGISTRATION_TOOLBOX_INTEGRATION_TESTS} itkFluidRegistration3DTest 
	#    ${INPUT_DATA}/ellipse-128-128-128-50-45-40.nii ${INPUT_DATA}/ellipse-128-128-128-50-50-50.nii
	#    ${TEMP_DIR}/fluid_ellipse_forward_3D_output.hdr ${TEMP_DIR}/fluid_ellipse_3D_forward_stretch.hdr
	#    0.01 0.0 2 200 64 
	#    ${TEMP_DIR}/fluid_ellipse_3D_forward_diff_after.hdr 
	#    ${TEMP_DIR}/fluid_ellipse_3D_forward_diff_before.hdr 0.05)

#	SET( FluidIntegrationTests_SRCS
#		FluidPDEFilterTest.cxx
#		FluidVelocityToDeformationFilterTest.cxx
#		itkFluidRegistrationTest.cxx
#		itkFluidDeformableTransformTest.cxx
#		itkFluidRegistration3DTest.cxx
#	)

#	ADD_EXECUTABLE(FluidIntegrationTests FluidIntegrationTests.cxx ${FluidIntegrationTests_SRCS})
#	TARGET_LINK_LIBRARIES(FluidIntegrationTests ${NIFTK_TEST_EXT_ITK_REGISTRATION_TOOLBOX_FLUID_LINK_LIBRARIES} )

