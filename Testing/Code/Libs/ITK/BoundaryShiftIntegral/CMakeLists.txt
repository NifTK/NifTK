PROJECT(itkBSITesting)

# Ouput data. 
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

# Set up the tests with the correct executables.
SET(LOCAL_TESTS ${CXX_TEST_PATH}/itkBSILocalTests)

ADD_TEST(itkMultipleDilateImageFilterTest ${LOCAL_TESTS} itkMultipleDilateErodeImageFilterTest)
ADD_TEST(itkBinaryIntersectWithPaddingImageFilterTest ${LOCAL_TESTS} itkBinaryIntersectWithPaddingImageFilterTest)
ADD_TEST(itkBinaryUnionWithPaddingImageFilterTest ${LOCAL_TESTS} itkBinaryUnionWithPaddingImageFilterTest)
ADD_TEST(itkIntensityNormalisationCalculatorTest ${LOCAL_TESTS} itkIntensityNormalisationCalculatorTest 
         ${INPUT_DATA}/134055125-623-1.hdr
         ${INPUT_DATA}/29790120-1200-1.hdr
         ${INPUT_DATA}/134055125-623-1_test_0.hdr
         ${INPUT_DATA}/134055125-623-1_test_1.hdr
         3.587049166392765e+02
         6.823141922906915e+02)
         
ADD_TEST(itkSimpleKMeansClusteringImageFilterTest ${LOCAL_TESTS} itkSimpleKMeansClusteringImageFilterTest 
         ${INPUT_DATA}/56970125-005-1.hdr
         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUT_DATA}/85680125-005-1.hdr
         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
         ${TEMPORARY_OUTPUT}/knbsi-kmeans.hdr
         280.391 678.648 953.249
         113.429 89.674 75.0122)
         
#ADD_TEST(itkSimpleKMeansClusteringImageFilterTest ${LOCAL_TESTS} itkSimpleKMeansClusteringImageFilterTest 
#         ${INPUT_DATA}/14457-002-1.hdr
#         ${INPUT_DATA}/Rhi_14457_985099287.hdr
#         ${INPUT_DATA}/12968-333-1.hdr
#         ${INPUT_DATA}/Jen_12968_1003153034.hdr
#         ${TEMPORARY_OUTPUT}/knbsi-kmeans.hdr
#         137.589 327.561 459.516
#         50.1285 42.9707 40.3896)
         
IF(NOT WIN32)         
  ADD_TEST(itkKN-BSITest1  ${itkBSITesting_SOURCE_DIR}/itkKN-BSITests.sh
         ${CXX_TEST_PATH}/niftkKMeansWindowWithLinearRegressionNormalisationBSI
         ${INPUT_DATA}/56970125-005-1.hdr
         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUT_DATA}/85680125-005-1.hdr
         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
         ${INPUT_DATA}/56970125-005-1.hdr
         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUT_DATA}/85680125-005-1.hdr
         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
         1 1 3 -1 -1 
         ${TEMPORARY_OUTPUT}/baseline-seg.hdr 
         ${TEMPORARY_OUTPUT}/repeat-seg.hdr 
         ${TEMPORARY_OUTPUT}/repeat-normalised.hdr
         ${TEMPORARY_OUTPUT}/kn-bsi-test-output.txt)

  ADD_TEST(itkKN-BSITest2 ${CMAKE_COMMAND} -E compare_files ${INPUT_DATA}/kn-bsi-test-output.txt ${TEMPORARY_OUTPUT}/kn-bsi-test-output.txt)
  ADD_TEST(itkKN-BSITest3 ${CMAKE_COMMAND} -E compare_files ${INPUT_DATA}/repeat-normalised.img ${TEMPORARY_OUTPUT}/repeat-normalised.img)
  #ADD_TEST(itkKN-BSITest4 ${CMAKE_COMMAND} -E compare_files ${INPUT_DATA}/repeat-normalised.hdr ${TEMPORARY_OUTPUT}/repeat-normalised.hdr)
ENDIF(NOT WIN32)

ADD_TEST(itkBoundaryShiftIntegralTest0 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
         ${INPUT_DATA}/56970125-005-1.hdr
         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUT_DATA}/85680125-005-1.hdr
         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
         ${INPUT_DATA}/56970125-005-1.hdr
         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUT_DATA}/85680125-005-1.hdr
         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
         1
         1
         0.25
         0.75
         ""
         5.62215631041469)

        
         
#ADD_TEST(itkBoundaryShiftIntegralTest0 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/2817_2_1211201039.hdr
#         ${INPUT_DATA}/smask04586-02817.hdr
#         ${INPUT_DATA}/04586-02817.hdr
#         ${INPUT_DATA}/repeat_mask.hdr
#         ${INPUT_DATA}/2817_2_1211201039.hdr
#         ${INPUT_DATA}/smask04586-02817.hdr
#         ${INPUT_DATA}/04586-02817.hdr
#         ${INPUT_DATA}/repeat_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         -3.40644240864726)
        
# Myriad Brain BSI 1.         
#ADD_TEST(itkBoundaryShiftIntegralTest1 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/16590-002-1.hdr
#         ${INPUT_DATA}/Jen_16590_1011090534.hdr
#         ${INPUT_DATA}/12677-333-1.hdr
#         ${INPUT_DATA}/Jen_12677_1025865661.hdr
#         ${INPUT_DATA}/16590-002-1.hdr
#         ${INPUT_DATA}/Jen_16590_1011090534.hdr
#         ${INPUT_DATA}/12677-333-1.hdr
#         ${INPUT_DATA}/Jen_12677_1025865661.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         10.863415)
         
# Myriad Brain BSI 2.         
#ADD_TEST(itkBoundaryShiftIntegralTest2 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14457-002-1.hdr
#         ${INPUT_DATA}/Rhi_14457_985099287.hdr
#         ${INPUT_DATA}/12968-333-1.hdr
#         ${INPUT_DATA}/Jen_12968_1003153034.hdr
#         ${INPUT_DATA}/14457-002-1.hdr
#         ${INPUT_DATA}/Rhi_14457_985099287.hdr
#         ${INPUT_DATA}/12968-333-1.hdr
#         ${INPUT_DATA}/Jen_12968_1003153034.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         4.570744)

# Myriad Brain BSI 3.         
#ADD_TEST(itkBoundaryShiftIntegralTest3 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14456-002-1.hdr
#         ${INPUT_DATA}/Rhi_14456_985097597.hdr
#         ${INPUT_DATA}/11955-333-1.hdr
#         ${INPUT_DATA}/Jen_11955_1003156681.hdr
#         ${INPUT_DATA}/14456-002-1.hdr
#         ${INPUT_DATA}/Rhi_14456_985097597.hdr
#         ${INPUT_DATA}/11955-333-1.hdr
#         ${INPUT_DATA}/Jen_11955_1003156681.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         1.895081)
         
# Myriad Brain BSI 4.         
#ADD_TEST(itkBoundaryShiftIntegralTest4 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14458-002-1.hdr
#         ${INPUT_DATA}/Rhi_14458_985103293.hdr
#         ${INPUT_DATA}/10876-333-1.hdr
#         ${INPUT_DATA}/Jen_10876_1003157995.hdr
#         ${INPUT_DATA}/14458-002-1.hdr
#         ${INPUT_DATA}/Rhi_14458_985103293.hdr
#         ${INPUT_DATA}/10876-333-1.hdr
#         ${INPUT_DATA}/Jen_10876_1003157995.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         4.061684)
         
# Myriad Brain BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest5 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         -0.667036)

# Myriad Brain BSI 6.         
#ADD_TEST(itkBoundaryShiftIntegralTest6 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         11.192541)

# Myriad Brain BSI 7.         
#ADD_TEST(itkBoundaryShiftIntegralTest7 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/15046-002-1.hdr
#         ${INPUT_DATA}/Rhi_15046_995632719.hdr
#         ${INPUT_DATA}/12265-333-1.hdr
#         ${INPUT_DATA}/Jen_12265_1011097315.hdr
#         ${INPUT_DATA}/15046-002-1.hdr
#         ${INPUT_DATA}/Rhi_15046_995632719.hdr
#         ${INPUT_DATA}/12265-333-1.hdr
#         ${INPUT_DATA}/Jen_12265_1011097315.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         4.793014)
         
# Myriad Brain BSI 8.         
#ADD_TEST(itkBoundaryShiftIntegralTest8 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/17191-002-1.hdr
#         ${INPUT_DATA}/Jen_17191_1017230328.hdr
#         ${INPUT_DATA}/11063-333-1.hdr
#         ${INPUT_DATA}/Jen_11063_1031664023.hdr
#         ${INPUT_DATA}/17191-002-1.hdr
#         ${INPUT_DATA}/Jen_17191_1017230328.hdr
#         ${INPUT_DATA}/11063-333-1.hdr
#         ${INPUT_DATA}/Jen_11063_1031664023.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         18.826257)
         
# Myriad Brain BSI 9.         
#ADD_TEST(itkBoundaryShiftIntegralTest9 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14491-002-1.hdr
#         ${INPUT_DATA}/Rhi_14491_986544474.hdr
#         ${INPUT_DATA}/11655-333-1.hdr
#         ${INPUT_DATA}/Jen_11655_1003159972.hdr
#         ${INPUT_DATA}/14491-002-1.hdr
#         ${INPUT_DATA}/Rhi_14491_986544474.hdr
#         ${INPUT_DATA}/11655-333-1.hdr
#         ${INPUT_DATA}/Jen_11655_1003159972.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         11.423518)
         
# Myriad Brain BSI 10.         
#ADD_TEST(itkBoundaryShiftIntegralTest10 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14613-002-1.hdr
#         ${INPUT_DATA}/Rhi_14613_987494773.hdr
#         ${INPUT_DATA}/12888-333-1.hdr
#         ${INPUT_DATA}/Jen_12888_1003218976.hdr
#         ${INPUT_DATA}/14613-002-1.hdr
#         ${INPUT_DATA}/Rhi_14613_987494773.hdr
#         ${INPUT_DATA}/12888-333-1.hdr
#         ${INPUT_DATA}/Jen_12888_1003218976.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         8.122624)
         

# Erosion/dilation test 1 - Myriad Brain BSI 6.           
#ADD_TEST(itkBoundaryShiftIntegralTest11 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         1
#         2
#         0.25
#         0.75
#         ""
#         11.204895)

# Erosion/dilation test 2 - Myriad Brain BSI 6.           
#ADD_TEST(itkBoundaryShiftIntegralTest12 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         2
#         1
#         0.25
#         0.75
#         ""
#         11.076887)
         
# Erosion/dilation test 3 - Myriad Brain BSI 6.           
#ADD_TEST(itkBoundaryShiftIntegralTest13 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         2
#         2
#         0.25
#         0.75
#         ""
#         11.089242)
         
# Erosion/dilation test 4 - Myriad Brain BSI 6.           
#ADD_TEST(itkBoundaryShiftIntegralTest14 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         3
#         3
#         0.25
#         0.75
#         ""
#         11.070470)

# Erosion/dilation test 5 - Myriad Brain BSI 6.           
#ADD_TEST(itkBoundaryShiftIntegralTest15 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         ${INPUT_DATA}/13569-002-1.hdr
#         ${INPUT_DATA}/Rhi_13569_975500628.hdr
#         ${INPUT_DATA}/13215-333-1.hdr
#         ${INPUT_DATA}/Rhi_13215_997969836.hdr
#         0
#         0
#         0.25
#         0.75
#         ""
#         2.302686)


# Intensity window test 1 - Myriad Brain BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest21 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         1
#         1
#         0.0
#         1.0
#         ""
#         -0.251706)

# Intensity window test 2 - Myriad Brain BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest22 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         1
#         1
#         0.1
#         0.9
#         ""
#         -0.363643)

# Intensity window test 3 - Myriad Brain BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest23 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         1
#         1
#         0.2
#         0.8
#         ""
#         -0.536025)

# Intensity window test 4 - Myriad Brain BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest24 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         1
#         1
#         0.3
#         0.7
#         ""
#         -0.782136)

# Intensity window test 5 - Myriad Brain BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest25 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         ${INPUT_DATA}/14650-002-1.hdr
#         ${INPUT_DATA}/Rhi_14650_995554546.hdr
#         ${INPUT_DATA}/11861-333-1.hdr
#         ${INPUT_DATA}/Jen_11861_1007460607.hdr
#         1
#         1
#         0.4
#         0.6
#         ""
#         -0.858589)
         
         
# ADNI Ventricle BSI 1.         
#ADD_TEST(itkBoundaryShiftIntegralTest31 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/71685125-005-1.hdr
#         ${INPUT_DATA}/Tra_71685125_1187349148_mask.hdr
#         ${INPUT_DATA}/139275125-005-1.hdr
#         ${INPUT_DATA}/Pro_139275125_1200660889_mask.hdr
#         ${INPUT_DATA}/71685125-005-1.hdr
#         ${INPUT_DATA}/Tra_71685125_1187610377_mask.hdr
#         ${INPUT_DATA}/139275125-005-1.hdr
#         ${INPUT_DATA}/Tes_139275125_1200913478_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         1.223436)
         
# ADNI Ventricle BSI 2.         
#ADD_TEST(itkBoundaryShiftIntegralTest32 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/70290125-005-1.hdr
#         ${INPUT_DATA}/Tra_70290125_1188819588_mask.hdr
#         ${INPUT_DATA}/91125125-005-1.hdr
#         ${INPUT_DATA}/Tra_91125125_1188819286_mask.hdr
#         ${INPUT_DATA}/70290125-005-1.hdr
#         ${INPUT_DATA}/Tra_70290125_1188835999_mask.hdr
#         ${INPUT_DATA}/91125125-005-1.hdr
#         ${INPUT_DATA}/Tes_139275125_1200913478_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         1.990514)

# ADNI Ventricle BSI 3.         
#ADD_TEST(itkBoundaryShiftIntegralTest33 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/70695125-005-1.hdr
#         ${INPUT_DATA}/Mat_70695125_1196096089_mask.hdr
#         ${INPUT_DATA}/144135125-005-1.hdr
#         ${INPUT_DATA}/Pro_144135125_1204126368_mask.hdr 
#         ${INPUT_DATA}/70695125-005-1.hdr
#         ${INPUT_DATA}/Mat_70695125_1196168043_mask.hdr
#         ${INPUT_DATA}/144135125-005-1.hdr
#         ${INPUT_DATA}/Rai_144135125_1204197336_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         2.149934)

# ADNI Ventricle BSI 4.         
#ADD_TEST(itkBoundaryShiftIntegralTest34 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/49320125-005-1.hdr
#         ${INPUT_DATA}/Liz_49320125_1187169046_mask.hdr
#         ${INPUT_DATA}/97380125-005-1.hdr
#         ${INPUT_DATA}/JoF_97380125_1189065635_mask.hdr 
#         ${INPUT_DATA}/49320125-005-1.hdr
#         ${INPUT_DATA}/Mat_70695125_1196168043_mask.hdr
#         ${INPUT_DATA}/97380125-005-1.hdr
#         ${INPUT_DATA}/JoF_97380125_1189082476_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         7.672666)

# ADNI Ventricle BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest35 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/65790125-005-1.hdr
#         ${INPUT_DATA}/Tra_65790125_1186486734_mask.hdr
#         ${INPUT_DATA}/62820125-005-1.hdr
#         ${INPUT_DATA}/Tra_62820125_1186486533_mask.hdr 
#         ${INPUT_DATA}/65790125-005-1.hdr
#         ${INPUT_DATA}/Tra_65790125_1186999047_mask.hdr
#         ${INPUT_DATA}/62820125-005-1.hdr
#         ${INPUT_DATA}/Tra_62820125_1186766503_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         5.743436)

# ADNI Ventricle BSI 6.         
#ADD_TEST(itkBoundaryShiftIntegralTest36 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/135990125-005-1.hdr
#         ${INPUT_DATA}/Mat_135990125_1197985018_mask.hdr
#         ${INPUT_DATA}/136440125-005-1.hdr
#         ${INPUT_DATA}/Pro_136440125_1202294083_mask.hdr 
#         ${INPUT_DATA}/135990125-005-1.hdr
#         ${INPUT_DATA}/Tes_135990125_1198145334_mask.hdr
#         ${INPUT_DATA}/136440125-005-1.hdr
#         ${INPUT_DATA}/Man_136440125_1202377759_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         9.721847)

# ADNI Ventricle BSI 7.         
#ADD_TEST(itkBoundaryShiftIntegralTest37 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/67365125-005-1.hdr
#         ${INPUT_DATA}/Lai_67365125_1188320963_mask.hdr
#         ${INPUT_DATA}/145350125-005-1.hdr
#         ${INPUT_DATA}/Pro_145350125_1204133905_mask.hdr
#         ${INPUT_DATA}/67365125-005-1.hdr
#         ${INPUT_DATA}/Lai_67365125_1188902379_mask.hdr
#         ${INPUT_DATA}/145350125-005-1.hdr
#         ${INPUT_DATA}/Rai_145350125_1204287686_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         16.544762)

# ADNI Ventricle BSI 8.         
#ADD_TEST(itkBoundaryShiftIntegralTest38 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/94770125-005-1.hdr
#         ${INPUT_DATA}/Tra_94770125_1188813804_mask.hdr
#         ${INPUT_DATA}/106785125-005-1.hdr
#         ${INPUT_DATA}/JoF_106785125_1189691740_mask.hdr
#         ${INPUT_DATA}/94770125-005-1.hdr
#         ${INPUT_DATA}/Tra_94770125_1188904785_mask.hdr
#         ${INPUT_DATA}/106785125-005-1.hdr
#         ${INPUT_DATA}/JoF_106785125_1189766908_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         2.350424)
         
# Ventricle BSI 9.         
#ADD_TEST(itkBoundaryShiftIntegralTest39 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/54945125-005-1.hdr
#         ${INPUT_DATA}/JoF_54945125_1186648089_mask.hdr
#         ${INPUT_DATA}/64440125-005-1.hdr
#         ${INPUT_DATA}/JoF_64440125_1186647816_mask.hdr
#         ${INPUT_DATA}/54945125-005-1.hdr
#         ${INPUT_DATA}/JoF_54945125_1186670124_mask.hdr
#         ${INPUT_DATA}/64440125-005-1.hdr
#         ${INPUT_DATA}/JoF_64440125_1186670309_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         -0.073805)
         
# Ventricle BSI 10.         
#ADD_TEST(itkBoundaryShiftIntegralTest40 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/56970125-005-1.hdr
#         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
#         ${INPUT_DATA}/85680125-005-1.hdr
#         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
#         ${INPUT_DATA}/56970125-005-1.hdr
#         ${INPUT_DATA}/Lai_56970125_1188831792_mask.hdr
#         ${INPUT_DATA}/85680125-005-1.hdr
#         ${INPUT_DATA}/Tra_85680125_1188822481_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ""
#         0.152870)


# Sub ROI BSI test 1 - ADNI Ventricle BSI 1.         
#ADD_TEST(itkBoundaryShiftIntegralTest41 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/71685125-005-1.hdr
#         ${INPUT_DATA}/Tra_71685125_1187349148_mask.hdr
#         ${INPUT_DATA}/139275125-005-1.hdr
#         ${INPUT_DATA}/Pro_139275125_1200660889_mask.hdr
#         ${INPUT_DATA}/71685125-005-1.hdr
#         ${INPUT_DATA}/Tra_71685125_1187349148_mask.hdr
#         ${INPUT_DATA}/139275125-005-1.hdr
#         ${INPUT_DATA}/Pro_139275125_1200660889_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Tra_71685125_1187610377_mask.hdr
#         1.157370)
         
# Sub ROI BSI test 2 - ADNI Ventricle BSI 2.         
#ADD_TEST(itkBoundaryShiftIntegralTest42 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/70290125-005-1.hdr
#         ${INPUT_DATA}/Tra_70290125_1188819588_mask.hdr
#         ${INPUT_DATA}/91125125-005-1.hdr
#         ${INPUT_DATA}/Tra_91125125_1188819286_mask.hdr
#         ${INPUT_DATA}/70290125-005-1.hdr
#         ${INPUT_DATA}/Tra_70290125_1188819588_mask.hdr
#         ${INPUT_DATA}/91125125-005-1.hdr
#         ${INPUT_DATA}/Tra_91125125_1188819286_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Tra_70290125_1188835999_mask.hdr
#         1.950371)

#  Sub ROI BSI test 3 - ADNI Ventricle BSI 3.         
#ADD_TEST(itkBoundaryShiftIntegralTest43 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/70695125-005-1.hdr
#         ${INPUT_DATA}/Mat_70695125_1196096089_mask.hdr
#         ${INPUT_DATA}/144135125-005-1.hdr
#         ${INPUT_DATA}/Pro_144135125_1204126368_mask.hdr 
#         ${INPUT_DATA}/70695125-005-1.hdr
#         ${INPUT_DATA}/Mat_70695125_1196096089_mask.hdr
#         ${INPUT_DATA}/144135125-005-1.hdr
#         ${INPUT_DATA}/Pro_144135125_1204126368_mask.hdr 
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Mat_70695125_1196168043_mask.hdr
#         2.066607)

#  Sub ROI BSI test 4 - ADNI Ventricle BSI 4.         
#ADD_TEST(itkBoundaryShiftIntegralTest44 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/49320125-005-1.hdr
#         ${INPUT_DATA}/Liz_49320125_1187169046_mask.hdr
#         ${INPUT_DATA}/97380125-005-1.hdr
#         ${INPUT_DATA}/JoF_97380125_1189065635_mask.hdr 
#         ${INPUT_DATA}/49320125-005-1.hdr
#         ${INPUT_DATA}/Liz_49320125_1187169046_mask.hdr
#         ${INPUT_DATA}/97380125-005-1.hdr
#         ${INPUT_DATA}/JoF_97380125_1189065635_mask.hdr 
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Mat_70695125_1196168043_mask.hdr
#         0.133362)

# Sub ROI BSI test 5 -  ADNI Ventricle BSI 5.         
#ADD_TEST(itkBoundaryShiftIntegralTest45 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/65790125-005-1.hdr
#         ${INPUT_DATA}/Tra_65790125_1186486734_mask.hdr
#         ${INPUT_DATA}/62820125-005-1.hdr
#         ${INPUT_DATA}/Tra_62820125_1186486533_mask.hdr 
#         ${INPUT_DATA}/65790125-005-1.hdr
#         ${INPUT_DATA}/Tra_65790125_1186486734_mask.hdr
#         ${INPUT_DATA}/62820125-005-1.hdr
#         ${INPUT_DATA}/Tra_62820125_1186486533_mask.hdr 
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Tra_65790125_1186999047_mask.hdr
#         5.454731)

# Sub ROI BSI test 6 -  ADNI Ventricle BSI 6.         
#ADD_TEST(itkBoundaryShiftIntegralTest46 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/135990125-005-1.hdr
#         ${INPUT_DATA}/Mat_135990125_1197985018_mask.hdr
#         ${INPUT_DATA}/136440125-005-1.hdr
#         ${INPUT_DATA}/Pro_136440125_1202294083_mask.hdr 
#         ${INPUT_DATA}/135990125-005-1.hdr
#         ${INPUT_DATA}/Mat_135990125_1197985018_mask.hdr
#         ${INPUT_DATA}/136440125-005-1.hdr
#         ${INPUT_DATA}/Pro_136440125_1202294083_mask.hdr 
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Tes_135990125_1198145334_mask.hdr
#         7.822409)

#  Sub ROI BSI test 7 - ADNI Ventricle BSI 7.         
#ADD_TEST(itkBoundaryShiftIntegralTest47 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/67365125-005-1.hdr
#         ${INPUT_DATA}/Lai_67365125_1188320963_mask.hdr
#         ${INPUT_DATA}/145350125-005-1.hdr
#         ${INPUT_DATA}/Pro_145350125_1204133905_mask.hdr
#         ${INPUT_DATA}/67365125-005-1.hdr
#         ${INPUT_DATA}/Lai_67365125_1188320963_mask.hdr
#         ${INPUT_DATA}/145350125-005-1.hdr
#         ${INPUT_DATA}/Pro_145350125_1204133905_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Lai_67365125_1188902379_mask.hdr
#         -15.800910)

#  Sub ROI BSI test 8 - ADNI Ventricle BSI 8.         
#ADD_TEST(itkBoundaryShiftIntegralTest48 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/94770125-005-1.hdr
#         ${INPUT_DATA}/Tra_94770125_1188813804_mask.hdr
#         ${INPUT_DATA}/106785125-005-1.hdr
#         ${INPUT_DATA}/JoF_106785125_1189691740_mask.hdr
#         ${INPUT_DATA}/94770125-005-1.hdr
#         ${INPUT_DATA}/Tra_94770125_1188813804_mask.hdr
#         ${INPUT_DATA}/106785125-005-1.hdr
#         ${INPUT_DATA}/JoF_106785125_1189691740_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Tra_94770125_1188904785_mask.hdr
#         2.271554)
         
#  Sub ROI BSI test 9 - Ventricle BSI 9.         
#ADD_TEST(itkBoundaryShiftIntegralTest49 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/54945125-005-1.hdr
#         ${INPUT_DATA}/JoF_54945125_1186648089_mask.hdr
#         ${INPUT_DATA}/64440125-005-1.hdr
#         ${INPUT_DATA}/JoF_64440125_1186647816_mask.hdr
#         ${INPUT_DATA}/54945125-005-1.hdr
#         ${INPUT_DATA}/JoF_54945125_1186648089_mask.hdr
#         ${INPUT_DATA}/64440125-005-1.hdr
#         ${INPUT_DATA}/JoF_64440125_1186647816_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/JoF_54945125_1186670124_mask.hdr
#         -0.084287)
         
#  Sub ROI BSI test 10 - Ventricle BSI 10.         
#ADD_TEST(itkBoundaryShiftIntegralTest50 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest 
#         ${INPUT_DATA}/56970125-005-1.hdr
#         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
#         ${INPUT_DATA}/85680125-005-1.hdr
#         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
#         ${INPUT_DATA}/56970125-005-1.hdr
#         ${INPUT_DATA}/Lai_56970125_1188319593_mask.hdr
#         ${INPUT_DATA}/85680125-005-1.hdr
#         ${INPUT_DATA}/Tra_85680125_1188322273_mask.hdr
#         1
#         1
#         0.25
#         0.75
#         ${INPUT_DATA}/Lai_56970125_1188831792_mask.hdr
#         0.140280)
         


# Build the executables. 
ADD_EXECUTABLE(itkBSILocalTests itkLocalTests.cxx 
               itkBinaryIntersectWithPaddingImageFilterTest.cxx
               itkBinaryUnionWithPaddingImageFilterTest.cxx
               itkIntensityNormalisationCalculatorTest.cxx
			         itkBoundaryShiftIntegralTest.cxx
               itkMultipleDilateImageFilterTest.cxx
               itkSimpleKMeansClusteringImageFilterTest.cxx)
               
TARGET_LINK_LIBRARIES(itkBSILocalTests ITKCommon ITKIO)

             
               



