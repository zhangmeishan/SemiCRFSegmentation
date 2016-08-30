
include_directories(
  basic
)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()

add_executable(NNCRFSegmentor NNCRFSegmentor.cpp)
add_executable(NNSemiCRFSegmentor NNSemiCRFSegmentor.cpp)
add_executable(NNHypridCRFSegmentor NNHypridCRFSegmentor.cpp)
add_executable(NNSEmbSemiCRFSegmentor NNSEmbSemiCRFSegmentor.cpp)
add_executable(NNSEmbHypridCRFSegmentor NNSEmbHypridCRFSegmentor.cpp)

