
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <memory>
#include <string>
#include <stdexcept>
#include "common-portaudio.h"
#include "portaudiocpp/PortAudioCpp.hxx"

#ifdef WIN32
#include <windows.h>
#endif


audio_async::audio_async() 
    : m_psys(NULL), 
    m_pInParamsRecord(NULL),
    m_pParamsRecord(NULL),
    m_pStreamRecord(NULL),
    m_running(false),
    m_pAudioBuffer(NULL)
{
    m_psys = &portaudio::System::instance();
#ifdef WIN32
    setlocale(LC_ALL, "");
#endif
}


audio_async::~audio_async() {
    if(m_pParamsRecord)
    {
        delete m_pParamsRecord;
        m_pParamsRecord = NULL;
    }
    if(m_pInParamsRecord)
    {
        delete m_pInParamsRecord;
        m_pInParamsRecord = NULL;
    }

    if(m_pAudioBuffer){
        m_pAudioBuffer->clear();
        delete m_pAudioBuffer;
        m_pAudioBuffer = NULL;
    }
    if(m_psys){
        m_psys->terminate();
        m_psys = NULL;
    }
}


// Convert a UTF-8 string to UTF-16 (wchar_t)
std::wstring UTF8StringToWString(const std::string& utf8Str) {
    int sizeNeeded = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), -1, nullptr, 0);
    std::wstring utf16Str(sizeNeeded, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), -1, &utf16Str[0], sizeNeeded);
    return utf16Str;
}

// Convert a UTF-16 (wchar_t) string to UTF-8
std::string utf16ToUtf8(const std::wstring& utf16Str) {
    int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, utf16Str.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string utf8Str(sizeNeeded, 0);
    WideCharToMultiByte(CP_UTF8, 0, utf16Str.c_str(), -1, &utf8Str[0], sizeNeeded, nullptr, nullptr);
    return utf8Str;
}

void audio_async::print_device_info()
{
    std::cout << std::string(80, '*') << std::endl;
    for (portaudio::System::DeviceIterator i = m_psys->devicesBegin(); i != m_psys->devicesEnd(); ++i)
    {
        std::string strDetails = "";
        if ((*i).isSystemDefaultInputDevice())
            strDetails += "default input";
        if ((*i).isSystemDefaultOutputDevice())
            strDetails += "default output";

        char device_info[512];
        sprintf_s(device_info, 512,
            "%d: %s, in=%d, out=%d, %s, %s",
            (*i).index(),
            (*i).name(),
            (*i).maxInputChannels(),
            (*i).maxOutputChannels(),
            (*i).hostApi().name(),
            strDetails.c_str());

#ifdef WIN32
        std::wcout << UTF8StringToWString(device_info) << std::endl;
#else
        std::cout << device_info << std::endl;
#endif
    }
    std::cout << std::string(80, '*') << std::endl << std::endl;
}

bool audio_async::init(int iInputDevice, uint8_t save_audio, bool enable_rnnoise){

    try
	{
        // List out all the devices we have   
        print_device_info();

        int 	iNumDevices 		= m_psys->deviceCount();
        std::cout << "Number of devices = " << iNumDevices << std::endl;		
        if ((iInputDevice >= 0) && (iInputDevice >= iNumDevices))
        {
            std::cout << "Input device index out of range!" << std::endl;
            return false;
        }

		int input_channel = m_psys->deviceByIndex(iInputDevice).maxInputChannels();
		double input_sample_rate = m_psys->deviceByIndex(iInputDevice).defaultSampleRate();

        m_pAudioBuffer = new AudioBuffer<float>((size_t)(OUTPOUT_SAMPLE_RATE * 300), enable_rnnoise, input_channel, input_sample_rate);
        
        if (NULL == m_pAudioBuffer)
            return false;
   
        m_pAudioBuffer->setSaveAudioFlag(save_audio);


#ifdef WIN32
        std::wcout << L"Opening recording input stream on " << UTF8StringToWString(m_psys->deviceByIndex(iInputDevice).name()) << std::endl;
#else
        std::cout << "Opening recording input stream on " << m_psys->deviceByIndex(iInputDevice).name() << std::endl;
#endif // WIN32

        
        m_pInParamsRecord = new portaudio::DirectionSpecificStreamParameters(
			m_psys->deviceByIndex(iInputDevice), 
            input_channel,
			portaudio::FLOAT32, 
			false, 
			m_psys->deviceByIndex(iInputDevice).defaultLowInputLatency(), 
			NULL
			);
        m_pParamsRecord = new portaudio::StreamParameters(
			*m_pInParamsRecord, 
			portaudio::DirectionSpecificStreamParameters::null(), 
            input_sample_rate,
			FRAMES_PER_BUFFER, 
			paClipOff
			);		
        m_pStreamRecord = new portaudio::MemFunCallbackStream<AudioBuffer<float>>(
			*m_pParamsRecord, 
			*m_pAudioBuffer, 
			&AudioBuffer<float>::RecordCallback
			);

    }
	catch (const portaudio::PaException &e)
	{
        std::cout << "A PortAudio error occured: " << e.paErrorText() << std::endl;
	}
	catch (const portaudio::PaCppException &e)
	{
        std::cout << "A PortAudioCpp error occured: " << e.what() << std::endl;
	}
	catch (const exception &e)
	{
        std::cout << "A generic exception occured: " << e.what() << std::endl;
	}
	catch (...)
	{
        std::cout << "An unknown exception occured." << std::endl;
	}

    return true;
}

bool audio_async::resume() 
{
    cout << "resume" << endl;
    if (NULL == m_pStreamRecord) {
        std::cout << __func__ << ": no audio device to resume!" << std::endl;
        return false;
    }

    if (m_running) {
        std::cout << __func__ << ": already running!" << std::endl;
        return false;
    }

    m_pStreamRecord->start();

    m_running = true;

    return true;
}

bool audio_async::pause() 
{
    if (NULL == m_pStreamRecord) {
        std::cout << __func__ << ": no audio device to pause!" << std::endl;
        return false;
    }

    if (!m_running) {
        std::cout << __func__ << ": already paused!" << std::endl;
        return false;
    }

    m_pStreamRecord->stop();

    m_running = false;

    return true;
}

void audio_async::close()
{
    m_running = true;
    if (m_pStreamRecord) {
        m_pStreamRecord->close();
        delete m_pStreamRecord;
        m_pStreamRecord = NULL;
    }
}

bool audio_async::clear() 
{
    this->m_pAudioBuffer->clear();
    return true;
}


bool audio_async::get(int fames, std::vector<float> & result) 
{
    if (NULL == m_pStreamRecord) {
        std::cout << __func__ << ": no audio device to get audio from!" << std::endl;
        return false;
    }

    if (!m_running) {
        std::cout << __func__ << ": not running!" << std::endl;
        return false;
    }
    
    size_t osize = (size_t)(fames);
    result.resize(osize);
    bool status = m_pAudioBuffer->dequeue(&result[0],osize);
    if (false == status)
    {
        result.clear();
        return false;
    }
    return true;
}