if (WHISPER_portaudio)
    # stream
    add_executable(stream-portaudio stream-portaudio.cpp)
    include(DefaultTargetOptions)
    target_link_libraries(stream-portaudio PRIVATE common common-portaudio whisper ${CMAKE_THREAD_LIBS_INIT})
endif ()
