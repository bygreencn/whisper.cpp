
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
    m_running = true;
    if(m_pStreamRecord){
        m_pStreamRecord->stop();
        delete m_pStreamRecord;
        m_pStreamRecord = NULL;
    }
    
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

bool audio_async::init(int iInputDevice, bool save_audio){

    try
	{
        // List out all the devices we have    
        int 	iNumDevices 		= m_psys->deviceCount();


        std::cout << "Number of devices = " << iNumDevices << std::endl;		
        if ((iInputDevice >= 0) && (iInputDevice >= iNumDevices))
        {
            cout << "Input device index out of range!" << endl;
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

            cout << (*i).index() << ": " << (*i).name() << ", ";
            cout << "in=" << (*i).maxInputChannels() << " ";
            cout << "out=" << (*i).maxOutputChannels() << ", ";
            cout << (*i).hostApi().name();

            cout << strDetails.c_str() << endl;
        }
#endif
   
        m_pAudioBuffer->Save_Audio(save_audio);

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
		cout << "A PortAudio error occured: " << e.paErrorText() << endl;
	}
	catch (const portaudio::PaCppException &e)
	{
		cout << "A PortAudioCpp error occured: " << e.what() << endl;
	}
	catch (const exception &e)
	{
		cout << "A generic exception occured: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "An unknown exception occured." << endl;
	}

    return true;
}

bool audio_async::resume() 
{
    cout << "resume" << endl;
    if (!m_pStreamRecord) {
        fprintf(stderr, "%s: no audio device to resume!\n", __func__);
        return false;
    }

    if (m_running) {
        fprintf(stderr, "%s: already running!\n", __func__);
        return false;
    }

    m_pStreamRecord->start();

    m_running = true;

    return true;
}

bool audio_async::pause() 
{
    if (!m_pStreamRecord) {
        fprintf(stderr, "%s: no audio device to pause!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: already paused!\n", __func__);
        return false;
    }

    m_pStreamRecord->stop();

    m_running = false;

    return true;
}

bool audio_async::clear() 
{
    this->m_pAudioBuffer->clear();
    return true;
}


void audio_async::get(int fames, std::vector<float> & result) 
{
    if (!m_pStreamRecord) {
        fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
        return;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }
    
    size_t osize = (size_t)(fames);
    result.resize(osize);
    m_pAudioBuffer->dequeue(&result[0],osize);
}