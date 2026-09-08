
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
#include "pa_win_wasapi.h"
#include "portaudiocpp/PortAudioCpp.hxx"


namespace string_encoding_utility {

    namespace convert {
        inline std::wstring mb_to_wide(const std::string& str, UINT code_page) {
            if (str.empty()) return std::wstring();

            int len = MultiByteToWideChar(code_page, 0, str.c_str(), -1, nullptr, 0);
            if (len <= 0) throw std::runtime_error("MultiByteToWideChar failed");

            std::wstring wstr(len - 1, L'\0');
            MultiByteToWideChar(code_page, 0, str.c_str(), -1, (LPWSTR)wstr.data(), len);
            return wstr;
        }


        inline std::string wide_to_mb(const std::wstring& wstr, UINT code_page, const char* default_char = nullptr) {
            if (wstr.empty()) return std::string();

            const char* actual_default = (code_page == CP_UTF8) ? nullptr : default_char;

            int len = WideCharToMultiByte(code_page, 0, wstr.c_str(), -1, nullptr, 0, actual_default, nullptr);
            if (len <= 0) throw std::runtime_error("WideCharToMultiByte failed");

            std::string str(len - 1, '\0');
            WideCharToMultiByte(code_page, 0, wstr.c_str(), -1, (LPSTR)str.data(), len, actual_default, nullptr);
            return str;
        }
    } // namespace detail

    namespace filter{
        inline std::string remove_repeated_char(const std::string& str, size_t threshold = 3, size_t keep = 2) {
            if (str.empty() || threshold < 2) return str;

            std::string pattern = "(.)\\1{" + std::to_string(threshold - 1) + ",}";
            std::regex re(pattern);

            std::string replacement;
            replacement.reserve(keep * 2);
            for (size_t i = 0; i < keep; ++i) replacement += "$1";

            return std::regex_replace(str, re, replacement);
        }

        inline std::wstring remove_repeated_char(const std::wstring& wstr, size_t threshold = 3, size_t keep = 2) {
            if (wstr.empty() || threshold < 2) return wstr;

            std::wstring pattern = L"(.)\\1{" + std::to_wstring(threshold - 1) + L",}";
            std::wregex re(pattern);

            std::wstring replacement;
            replacement.reserve(keep * 2);
            for (size_t i = 0; i < keep; ++i) replacement += L"$1";

            return std::regex_replace(wstr, re, replacement);
        }

        inline std::string remove_repeated_substr(const std::string& str, size_t threshold = 3, size_t keep = 2) {
            if (str.empty() || threshold < 2) return str;

            std::string pattern = "(.{2,}?)\\1{" + std::to_string(threshold - 1) + ",}";
            std::regex re(pattern);

            std::string replacement;
            replacement.reserve(keep * 2);
            for (size_t i = 0; i < keep; ++i) replacement += "$1";

            return std::regex_replace(str, re, replacement);
        }

        inline std::wstring remove_repeated_substr(const std::wstring& wstr, size_t threshold = 3, size_t keep = 2) {
            if (wstr.empty() || threshold < 2) return wstr;

            std::wstring pattern = L"(.{2,}?)\\1{" + std::to_wstring(threshold - 1) + L",}";
            std::wregex re(pattern);

            std::wstring replacement;
            replacement.reserve(keep * 2);
            for (size_t i = 0; i < keep; ++i) replacement += L"$1";

            return std::regex_replace(wstr, re, replacement);
        }
	} //namespace filter



    // 1. utf8 to wstring
    std::wstring utf8_to_wstring(const std::string& utf8_str) {
        return convert::mb_to_wide(utf8_str, CP_UTF8);
    }

    // 2. gbk to wstring
    std::wstring gbk_to_wstring(const std::string& gbk_str) {
        return convert::mb_to_wide(gbk_str, CP_GBK);
    }

    // 3. wstring to utf8
    std::string wstring_to_utf8(const std::wstring& wstr) {
        return convert::wide_to_mb(wstr, CP_UTF8);
    }

    // 4. wstring to gbk
    std::string wstring_to_gbk(const std::wstring& wstr) {
        return convert::wide_to_mb(wstr, CP_GBK, "?");
    }

    // 5. utf8 to gbk
    std::string utf8_to_gbk(const std::string& utf8_str) {
        return convert::wide_to_mb(convert::mb_to_wide(utf8_str, CP_UTF8), CP_GBK, "?");
    }

    // 6. gbk to utf8
    std::string gbk_to_utf8(const std::string& gbk_str) {
        return convert::wide_to_mb(convert::mb_to_wide(gbk_str, CP_GBK), CP_UTF8);
    }

	std::wstring remove_repeated_substr_utf8_to_wstring(const std::string& str, size_t threshold /*= 3*/, size_t keep /*= 2*/) {
		std::wstring wstr = utf8_to_wstring(str);
        return filter::remove_repeated_substr(wstr, threshold, keep);
	}
} // namespace Encoding


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
        std::wcout << string_encoding_utility::utf8_to_wstring(device_info) << std::endl;
#else
        std::cout << device_info << std::endl;
#endif
    }
    std::cout << std::string(80, '*') << std::endl << std::endl;
}

void audio_async::print_working_microphones()
{
    
    std::cout << std::string(80, '*') << std::endl;
    for (portaudio::System::DeviceIterator i = m_psys->devicesBegin(); i != m_psys->devicesEnd(); ++i)
    {
        std::string strDetails = "";
        if ((*i).maxInputChannels() > 0)
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
            std::wcout << string_encoding_utility::utf8_to_wstring(device_info) << std::endl;
#else
            std::cout << device_info << std::endl;
#endif
        }
    }
    std::cout << std::string(80, '*') << std::endl << std::endl;
}

bool audio_async::checkIfLoopback(const portaudio::Device& device) {
    // Get the underlying C device index
    PaDeviceIndex index = device.index();

    // Use the C API function
    return PaWasapi_IsLoopback(index) != 0;
}

int audio_async::print_loopback_devices()
{
	int loopback_device_index=-1;
	int loopback_device_count = 0;
    std::cout << std::string(80, '*') << std::endl;
    for (portaudio::System::DeviceIterator i = m_psys->devicesBegin(); i != m_psys->devicesEnd(); ++i)
    {
        std::string strDetails = "";
        if ((*i).maxInputChannels() > 0)
        {
            std::string strDetails = "";
			if ((*i).maxInputChannels() > 0 && (*i).hostApi().typeId() == paWASAPI && checkIfLoopback(*i))
            {
                strDetails += "loopback device";
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
                std::wcout << string_encoding_utility::utf8_to_wstring(device_info) << std::endl;
#else
                std::cout << device_info << std::endl;
#endif
				loopback_device_index = (*i).index();
                loopback_device_count++;
            }
        }
    }
    std::cout << std::string(80, '*') << std::endl << std::endl;

    if(loopback_device_count != 1)
    {
        loopback_device_index = -1;
        std::cout << "Please select the device index for audio input:" << std::endl;
        int result = scanf_s("%d", &loopback_device_index);

        if (result == 1) {
            std::printf("You entered: loopback_device_index = %d\n", loopback_device_index);
        }
	}

    return loopback_device_index;
}
bool audio_async::init(int iInputDevice, uint8_t save_audio, bool enable_rnnoise)
{
    try
	{
        // List out all the devices we have   
        print_working_microphones();

        if (iInputDevice < 0)
            iInputDevice = print_loopback_devices();

        int iNumDevices = m_psys->deviceCount();	
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
        std::wcout << L"Opening recording input stream on " << string_encoding_utility::utf8_to_wstring(m_psys->deviceByIndex(iInputDevice).name()) << std::endl;
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