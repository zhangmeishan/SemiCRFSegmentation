
include_directories(
  model
)

add_executable(NNSemiCRFSegmentor NNSemiCRFSegmentor.cpp)
#target_link_libraries(NNSemiCRFSegmentor ${LIBS})
