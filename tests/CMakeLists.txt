include(FetchContent)

find_package(GTest REQUIRED)

file(GLOB_RECURSE TESTS *.cpp)

add_executable(tests ${TESTS})

target_include_directories(tests PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(tests GTest::GTest GTest::Main)
add_test(NAME tests COMMAND tests)
