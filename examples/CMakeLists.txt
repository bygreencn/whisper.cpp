# dependencies

find_package(Threads REQUIRED)

# third-party

if (WHISPER_SDL2)
    # SDL2
    find_package(SDL2 REQUIRED)

    string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

    message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
    message(STATUS "SDL2_LIBRARIES = ${SDL2_LIBRARIES}")
endif()

if (WHISPER_CLBLAST)
    find_package(CLBlast REQUIRED)
endif()

# common

set(TARGET common)

add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    grammar-parser.h
    grammar-parser.cpp
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper)

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

if (WHISPER_SDL2)
    # common-sdl

    set(TARGET common-sdl)

    add_library(${TARGET} STATIC
        common-sdl.h
        common-sdl.cpp
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC ${SDL2_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${SDL2_LIBRARIES})

    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# add json lib
add_library(json_cpp INTERFACE)
target_include_directories(json_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})



# Get all propreties that cmake supports
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)
endif()
    
function(print_properties)
    message("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction()
    
function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message("${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

set(CMAKE_VERBOSE_MAKEFILE ON)
if (WHISPER_portaudio)
    
    find_package(SampleRate CONFIG REQUIRED)
    if(SampleRate_FOUND)
        message(STATUS "SampleRate_VERSION = ${SampleRate_VERSION}")
    else(SampleRate_FOUND)
        message(FATAL_ERROR SampleRate library not found¡±)
    endif(SampleRate_FOUND)

    find_package(SndFile REQUIRED)
    if(SndFile_FOUND)
        message(STATUS "SndFile_VERSION = ${SndFile_VERSION}")
    else(SndFile_FOUND)
        message(FATAL_ERROR SndFile library not found¡±)
    endif(SndFile_FOUND)

    find_package(PortAudio REQUIRED)
    if(PortAudio_FOUND)
        message(STATUS "PortAudio_VERSION = ${PortAudio_VERSION}")
    else(PortAudio_FOUND)
        message(FATAL_ERROR PortAudio library not found¡±)
    endif(PortAudio_FOUND)

    find_package(PortAudioCpp REQUIRED)
    if(PortAudioCpp_FOUND)
        message(STATUS "PortAudioCpp_VERSION = ${PortAudioCpp_VERSION}")
    else(PortAudioCpp_FOUND)
        message(FATAL_ERROR PortAudioCpp library not found¡±)
    endif(PortAudioCpp_FOUND)

    find_package(onnxruntime REQUIRED)
    if(onnxruntime_FOUND)
        message(STATUS "onnxruntime_VERSION = ${onnxruntime_VERSION}")
    else(onnxruntime_FOUND)
        message(FATAL_ERROR onnxruntime library not found¡±)
    endif(onnxruntime_FOUND)

    find_package(rnnoise REQUIRED)
    if(rnnoise_FOUND)
        message(STATUS "rnnoise_VERSION = ${rnnoise_VERSION}")
    else(rnnoise_FOUND)
        message(FATAL_ERROR rnnoise library not found¡±)
    endif(rnnoise_FOUND)

    #print_target_properties(SampleRate::samplerate)
    #print_target_properties(SndFile::sndfile)
    #print_target_properties(PortAudio)
    #print_target_properties(PortAudio)
    #print_target_properties(ONNXRUNTIME)

    add_library(common-portaudio STATIC
        common-portaudio.h
        common-portaudio.cpp
        )
    

    include(DefaultTargetOptions)
    #target_include_directories(common-portaudio PUBLIC "${ONNXRUNTIME_INCLUDE_DIR}" )
    target_link_libraries(common-portaudio PUBLIC onnxruntime::onnxruntime SampleRate::samplerate SndFile::sndfile PortAudio::PortAudio PortAudio::portaudiocpp rnnoise::rnnoise winmm dsound setupapi ole32 uuid)
    
    set_target_properties(common-portaudio PROPERTIES POSITION_INDEPENDENT_CODE ON)

  
    #add_example(record_resample_file)
    #target_link_libraries(record_resample_file SampleRate::samplerate SndFile::sndfile)
    #if(SUPPORT_sndfile)
    #    add_example(record_resample_sndfile)
    #    target_link_libraries(record_resample_sndfile SampleRate::samplerate SndFile::sndfile)
        
    #    add_example(sndfile_resample)
    #    target_link_libraries(sndfile_resample SampleRate::samplerate SndFile::sndfile)
    #endif()
endif()

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
    add_subdirectory(whisper.wasm)
    set_target_properties(libmain PROPERTIES FOLDER "libs")
    add_subdirectory(stream.wasm)
    set_target_properties(libstream PROPERTIES FOLDER "libs")
    add_subdirectory(command.wasm)
    set_target_properties(libcommand PROPERTIES FOLDER "libs")
    add_subdirectory(talk.wasm)
    set_target_properties(libtalk PROPERTIES FOLDER "libs")
    add_subdirectory(bench.wasm)
    set_target_properties(libbench PROPERTIES FOLDER "libs")
elseif(CMAKE_JS_VERSION)
    add_subdirectory(addon.node)
    set_target_properties(addon.node PROPERTIES FOLDER "examples")
else()
    add_subdirectory(main)
    set_target_properties(main PROPERTIES FOLDER "examples")
if (WHISPER_SDL2)
    add_subdirectory(stream)
    set_target_properties(stream PROPERTIES FOLDER "examples")
endif (WHISPER_SDL2)
    add_subdirectory(stream-portaudio)
    set_target_properties(stream-portaudio PROPERTIES FOLDER "examples")
    add_subdirectory(server)
    set_target_properties(server PROPERTIES FOLDER "examples")
if (WHISPER_SDL2)
    add_subdirectory(command)
    set_target_properties(command PROPERTIES FOLDER "examples")
endif (WHISPER_SDL2)
    add_subdirectory(bench)
    set_target_properties(bench PROPERTIES FOLDER "examples")
    add_subdirectory(quantize)
    set_target_properties(quantize PROPERTIES FOLDER "examples")
if (WHISPER_SDL2)
    add_subdirectory(talk)
    set_target_properties(talk PROPERTIES FOLDER "examples")
    add_subdirectory(talk-llama)
    set_target_properties(talk-llama PROPERTIES FOLDER "examples")
    add_subdirectory(lsp)
    set_target_properties(lsp PROPERTIES FOLDER "examples")
    if (LLAMA_SYCL)
        add_subdirectory(sycl)
        set_target_properties(sycl PROPERTIES FOLDER "examples")
    endif()
endif (WHISPER_SDL2)
endif()

if (WHISPER_SDL2)
    add_subdirectory(wchess)
    set_target_properties(wchess PROPERTIES FOLDER "examples")
endif (WHISPER_SDL2)
