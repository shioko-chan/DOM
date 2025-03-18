file(GLOB_RECURSE SOURCES "*.cpp")

add_executable(orthophoto STATIC ${SOURCES})

target_include_directories(orthophoto PUBLIC
    ${CMAKE_SOURCE_DIR}/thirdparty/mylib/include
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

install(TARGETS orthophoto
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY ../include DESTINATION include)
