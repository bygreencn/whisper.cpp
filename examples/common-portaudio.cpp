
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


audio_async::audio_async() 
    : m_psys(NULL), 
    m_pInParamsRecord(NULL),
    m_pParamsRecord(NULL),
    m_pStreamRecord(NULL),
    m_running(false){
    m_pAudioBuffer = new AudioBuffer<float>((size_t)(OUTPOUT_SAMPLE_RATE * 300));
    m_psys = &portaudio::System::instance();
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

bool audio_async::init(int iInputDevice, uint8_t save_audio){

    try
	{
        // List out all the devices we have    
        int 	iNumDevices 		= m_psys->deviceCount();


        std::cout << "Number of devices = " << iNumDevices << std::endl;		
        if ((iInputDevice >= 0) && (iInputDevice >= iNumDevices))
        {
            std::cout << "Input device index out of range!" << std::endl;
            return false;
        }

#ifdef __DEBUG__  
        string	strDetails			= "";

        for (portaudio::System::DeviceIterator i = m_psys->devicesBegin(); i != m_psys->devicesEnd(); ++i)
        {
            strDetails = "";
            if ((*i).isSystemDefaultInputDevice())
                strDetails += ", default input";
            if ((*i).isSystemDefaultOutputDevice())
                strDetails += ", default output";

            std::cout << (*i).index() << ": " << (*i).name() << ", ";
            std::cout << "in=" << (*i).maxInputChannels() << " ";
            std::cout << "out=" << (*i).maxOutputChannels() << ", ";
            std::cout << (*i).hostApi().name();

            std::cout << strDetails.c_str() << std::endl;
        }
#endif
   
        m_pAudioBuffer->setSaveAudioFlag(save_audio);

        cout << "Opening recording input stream on " << m_psys->deviceByIndex(iInputDevice).name() << endl;
        m_pInParamsRecord = new portaudio::DirectionSpecificStreamParameters(
			m_psys->deviceByIndex(iInputDevice), 
			INPUT_CHANNEL, 
			portaudio::FLOAT32, 
			false, 
			m_psys->deviceByIndex(iInputDevice).defaultLowInputLatency(), 
			NULL
			);
        m_pParamsRecord = new portaudio::StreamParameters(
			*m_pInParamsRecord, 
			portaudio::DirectionSpecificStreamParameters::null(), 
			INPUT_SAMPLE_RATE, 
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