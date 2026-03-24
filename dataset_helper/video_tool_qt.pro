QT       += widgets core
CONFIG   += c++17
TEMPLATE  = app
TARGET    = video_tool_qt

SOURCES  += dataset_helper_qt.cpp

# FFmpeg
LIBS     += $$system(pkg-config --libs libavcodec libavformat libavutil libswscale libswresample)
QMAKE_CXXFLAGS += $$system(pkg-config --cflags libavcodec libavformat libavutil libswscale libswresample)

# Disable deprecation warnings from FFmpeg headers
QMAKE_CXXFLAGS += -Wno-deprecated-declarations
