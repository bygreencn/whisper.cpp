#pragma once

#include <portaudiocpp/PortAudioCpp.hxx>
#include <samplerate.h>
#include <sndfile.hh>
#include <rnnoise.h>
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <string>


#define SAVE_AUDIO_RAW          (0x08)
#define SAVE_AUDIO_RNNOISED     (0x04)
#define SAVE_AUDIO_RESAMPLED    (0x02)
#define SAVE_AUDIO_VAD          (0x01)


//#include <format>
// ---------------------------------------------------------------------
// Some constants:
const int       OUTPUT_CHANNEL             = 1;
const double	OUTPOUT_SAMPLE_RATE			= 16000;
const int		FRAMES_PER_BUFFER	= 4800;

const int       RNNOISE_BUFFER_SIZE = 480;

using namespace std;


#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstring>  

template <typename T>
class CircularQueue {
public:
    explicit CircularQueue(size_t capacity) : capacity_(capacity), queue_(capacity){}

public:
    bool enqueue(const void* pdata, size_t blockSize) {
        int retry = 2;
        std::unique_lock<std::mutex> lock(mutex_);

        //std::cout << "Before enqueue" << std::endl;
        //std::cout << readIndex_ << " " << writeIndex_ << std::endl;
        //std::cout << size_ << std::endl;

        // use lambda function detect capacity
        auto isCapacityEnough = [&]() {
            return ((capacity_-size_) >= blockSize);
        };

        volatile bool status = conditionVariable_.wait_for(lock, std::chrono::microseconds(100), isCapacityEnough);

        if (status == false)
        {
            lock.unlock();
            std::cout << "[enqueue] timeout" << std::endl;
            return false;
        }

        size_t remainingSpace = capacity_ - writeIndex_;
        size_t dataSize = blockSize * sizeof(T);

        if (dataSize <= remainingSpace * sizeof(T)) {
            std::memcpy(&queue_[writeIndex_], pdata, dataSize);
        } else {
            size_t firstPartSize = remainingSpace * sizeof(T);
            std::memcpy(&queue_[writeIndex_], (char *)pdata, firstPartSize);

            size_t secondPartSize = dataSize - firstPartSize;
            std::memcpy(&queue_[0], (char *)((T *)pdata + remainingSpace), secondPartSize);
        }

        writeIndex_ = (writeIndex_ + blockSize) % capacity_;
        size_ += blockSize;

        lock.unlock();
        //std::cout << "After enqueue" << std::endl;
        //std::cout << readIndex_ << " " << writeIndex_ << std::endl;
        //std::cout << size_ << std::endl;
        conditionVariable_.notify_one();  

        return true;
    }


    bool dequeue(void* pdata, size_t blockSize) {
        std::unique_lock<std::mutex> lock(mutex_);

        //std::cout << "Before dequeue" << std::endl;
        //std::cout << readIndex_ << " " << writeIndex_ << std::endl;
        //std::cout << size_ << std::endl;

        // use lambda function detect capacity
        auto isCapacityEnough = [&]() {
            return size_ >= blockSize;
        };
        volatile bool status = conditionVariable_.wait_for(lock, std::chrono::microseconds(100),isCapacityEnough);

        if (status == false)
        {
            lock.unlock();
            //std::cout << "[dequeue] timeout" << std::endl;
            return false;
        }

        size_t remainingData = capacity_ - readIndex_;
        size_t dataSize = blockSize * sizeof(T);

        if (dataSize <= remainingData * sizeof(T)) {
            std::memcpy((char *)pdata, &queue_[readIndex_], dataSize);
        } else {
            size_t firstPartSize = remainingData * sizeof(T);
            std::memcpy((char *)pdata, &queue_[readIndex_], firstPartSize);

            size_t secondPartSize = dataSize - firstPartSize;
            std::memcpy((char *)pdata + remainingData, &queue_[0], secondPartSize);
        }

        readIndex_ = (readIndex_ + blockSize) % capacity_;
        size_ -= blockSize;

        lock.unlock();
        //std::cout << "After dequeue" << std::endl;
        //std::cout << readIndex_ << " " << writeIndex_ << std::endl;
        //std::cout << size_ << std::endl;
        conditionVariable_.notify_one();  

        return true;
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    bool full() {
        std::unique_lock<std::mutex> lock(mutex_);
        return size_ == capacity_;
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return size_;
    }
    
    size_t capacity()
    {
        return capacity_;
    }

    void clear()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        size_ = 0;
        readIndex_ = 0;
        writeIndex_ = 0;
        queue_.clear();
    }

private:
    size_t capacity_;
    std::vector<T> queue_;
    volatile size_t size_ = 0;
    volatile size_t readIndex_ = 0;
    volatile size_t writeIndex_ = 0;
    std::mutex mutex_;
    std::condition_variable conditionVariable_;
};



class wav_writer {
public:
    wav_writer() : pfile_(NULL) {};
    ~wav_writer() {
        close();
    };

    bool open(const std::string & filename,
              const    uint32_t   sample_rate,
              const    uint16_t   bits_per_sample,
              const    uint16_t   channels) {

        close();

        filename_ = filename;

        if (bits_per_sample == 32)
            pfile_ = new SndfileHandle (filename.c_str(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_FLOAT, channels, sample_rate) ;
        else if (bits_per_sample == 16)
            pfile_ = new SndfileHandle (filename.c_str(), SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, channels, sample_rate) ;

        return true;
    }


    void write(const float * data, size_t length) {
       pfile_->write(data,length);
    }

    void close()
    {
        if(pfile_){
            delete pfile_;
            pfile_=NULL;
        }
    }

    const std::string get_filename() const {
        return filename_;
    }
private:
    SndfileHandle * pfile_; 
    std::string filename_;
};



//#define __DEBUG_SPEECH_PROB___
//#define __DEBUG__

#include <cstdio>
#include <cstdarg>

#if __cplusplus < 201703L
#include <memory>
#endif


//#define __DEBUG_SPEECH_PROB___

class timestamp_t
{
public:
    int64_t start;
    int64_t end;

    // default + parameterized constructor
    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end)
    {
    };

    // assignment operator modifies object, therefore non-const
    timestamp_t& operator=(const timestamp_t& a)
    {
        start = a.start;
        end = a.end;
        return *this;
    };

    // equality comparison. doesn't modify object. therefore const.
    bool operator==(const timestamp_t& a) const
    {
        return (start == a.start && end == a.end);
    };

    std::string c_str()
    {
        //return std::format("timestamp {:08d}, {:08d}", start, end);
        return format("{start:%08d,end:%08d}", start, end);
    };

private:

    std::string format(const char* fmt, ...)
    {
        char buf[256];

        va_list args;
        va_start(args, fmt);
        const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);

        if (r < 0)
            // conversion failed
            return {};

        const size_t len = r;
        if (len < sizeof buf)
            // we fit in the buffer
            return { buf, len };

#if __cplusplus >= 201703L
        // C++17: Create a string and write to its underlying array
        std::string s(len, '\0');
        va_start(args, fmt);
        std::vsnprintf(s.data(), len + 1, fmt, args);
        va_end(args);

        return s;
#else
        // C++11 or C++14: We need to allocate scratch memory
        auto vbuf = std::unique_ptr<char[]>(new char[len + 1]);
        va_start(args, fmt);
        std::vsnprintf(vbuf.get(), len + 1, fmt, args);
        va_end(args);

        return { vbuf.get(), len };
#endif
    };
};


class VadIterator
{
private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

private:
    void init_engine_threads(int inter_threads, int intra_threads)
    {
        // The method should be called in each thread/proc in multi-thread/proc work
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    };

    void init_onnx_model(const std::wstring& model_path)
    {
        // Init threads = 1 for 
        init_engine_threads(1, 1);
        // Load model
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    };

    void reset_states()
    {
        // Call reset before each audio start
        std::memset(_state.data(), 0, _state.size() * sizeof(float));
        triggered = false;
        temp_end = 0;
        current_sample = 0;

        prev_end = next_start = 0;

        speeches.clear();
        current_speech = timestamp_t();
    };

    void predict(const std::vector<float> &data)
    {
        // Infer
        // Create ort tensors
        input.assign(data.begin(), data.end());
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);

        // Clear and add inputs
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        // Infer
        ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // Output probability & update h,c recursively
        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        float *stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));

        // Push forward sample index
        current_sample += window_size_samples;

        // Reset temp_end when > threshold 
        if ((speech_prob >= threshold))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
            printf("{    start: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (temp_end != 0)
            {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (triggered == false)
            {
                triggered = true;

                current_speech.start = current_sample - window_size_samples;
            }
            return;
        }

        if (
            (triggered == true)
            && ((current_sample - current_speech.start) > max_speech_samples)
            ) {
            if (prev_end > 0) {
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                
                // previously reached silence(< neg_thres) and is still not speech(< thres)
                if (next_start < prev_end)
                    triggered = false;
                else{
                    current_speech.start = next_start;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;

            }
            else{ 
                current_speech.end = current_sample;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
            }
            return;

        }
        if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold))
        {
            if (triggered) {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{ speeking: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            else {
#ifdef __DEBUG_SPEECH_PROB___
                float speech = current_sample - window_size_samples; // minus window_size_samples to get precise start time point.
                printf("{  silence: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            }
            return;
        }


        // 4) End 
        if ((speech_prob < (threshold - 0.15)))
        {
#ifdef __DEBUG_SPEECH_PROB___
            float speech = current_sample - window_size_samples - speech_pad_samples; // minus window_size_samples to get precise start time point.
            printf("{      end: %.3f s (%.3f) %08d}\n", 1.0 * speech / sample_rate, speech_prob, current_sample - window_size_samples);
#endif //__DEBUG_SPEECH_PROB___
            if (triggered == true)
            {
                if (temp_end == 0)
                {
                    temp_end = current_sample;
                }
                if (current_sample - temp_end > min_silence_samples_at_max_speech)
                    prev_end = temp_end;
                // a. silence < min_slience_samples, continue speaking 
                if ((current_sample - temp_end) < min_silence_samples)
                {

                }
                // b. silence >= min_slience_samples, end speaking
                else
                {
                    current_speech.end = temp_end;
                    if (current_speech.end - current_speech.start > min_speech_samples)
                    {
                        speeches.push_back(current_speech);
                        current_speech = timestamp_t();
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        triggered = false;
                    }
                }
            }
            else {
                // may first windows see end state.
            }
            return;
        }
    };
public:
    void process(const std::vector<float>& input_wav)
    {
        reset_states();

        audio_length_samples = input_wav.size();

        for (size_t j = 0; j < audio_length_samples; j += window_size_samples)
        {
            if (j + window_size_samples > audio_length_samples)
                break;
            std::vector<float> r{ &input_wav[0] + j, &input_wav[0] + j + window_size_samples };
            predict(r);
        }

        if (current_speech.start >= 0) {
            current_speech.end = audio_length_samples;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
        }
    };

    void process(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        process(input_wav);
        collect_chunks(input_wav, output_wav);
    }

    void collect_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        output_wav.clear();
        for (int i = 0; i < speeches.size(); i++) {
#ifdef __DEBUG_SPEECH_PROB___
            std::cout << speeches[i].c_str() << std::endl;
#endif //#ifdef __DEBUG_SPEECH_PROB___
            std::vector<float> slice(&input_wav[speeches[i].start], &input_wav[speeches[i].end]);
            output_wav.insert(output_wav.end(),slice.begin(),slice.end());
        }
    };

    const std::vector<timestamp_t> get_speech_timestamps() const
    {
        return speeches;
    }

    void drop_chunks(const std::vector<float>& input_wav, std::vector<float>& output_wav)
    {
        output_wav.clear();
        int64_t current_start = 0;
        for (size_t i = 0; i < speeches.size(); i++) {

            std::vector<float> slice(&input_wav[current_start], &input_wav[speeches[i].start]);
            output_wav.insert(output_wav.end(), slice.begin(), slice.end());
            current_start = speeches[i].end;
        }

        std::vector<float> slice(&input_wav[current_start], &input_wav[input_wav.size()]);
        output_wav.insert(output_wav.end(), slice.begin(), slice.end());
    };

private:
    // model config
    int64_t window_size_samples;  // Assign when init, support 256 512 768 for 8k; 512 1024 1536 for 16k.
    int sample_rate;  //Assign when init support 16000 or 8000      
    int sr_per_ms;   // Assign when init, support 8 or 16
    float threshold;
    int min_silence_samples; // sr_per_ms * #ms
    int min_silence_samples_at_max_speech; // sr_per_ms * #98
    int min_speech_samples; // sr_per_ms * #ms
    float max_speech_samples;
    int speech_pad_samples; // usually a 
    size_t audio_length_samples;

    // model states
    bool triggered = false;
    int64_t temp_end = 0;
    int64_t current_sample = 0;
    // MAX 4294967295 samples / 8sample per ms / 1000 / 60 = 8947 minutes  
    int64_t prev_end;
    int64_t next_start = 0;

    //Output timestamp
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;


    // Onnx model
    // Inputs
    std::vector<Ort::Value> ort_inputs;

    std::vector<const char *> input_node_names = {"input", "state", "sr"};
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128; // It's FIXED.
    std::vector<float> _state;
    std::vector<int64_t> sr;

    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = {2, 1, 128}; 
    const int64_t sr_node_dims[1] = {1};

    // Outputs
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> output_node_names = {"output", "stateN"};

public:
    // Construction
    VadIterator(const std::wstring ModelPath,
        int Sample_rate = 16000, int windows_frame_size = 32,
        float Threshold = 0.5, int min_silence_duration_ms = 0,
        int speech_pad_ms = 32, int min_speech_duration_ms = 32,
        float max_speech_duration_s = std::numeric_limits<float>::infinity())
    {
        init_onnx_model(ModelPath);
        threshold = Threshold;
        sample_rate = Sample_rate;
        sr_per_ms = sample_rate / 1000;

        window_size_samples = windows_frame_size * sr_per_ms;

        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        speech_pad_samples = sr_per_ms * speech_pad_ms;

        max_speech_samples = (
            sample_rate * max_speech_duration_s
            - window_size_samples
            - 2 * speech_pad_samples
            );

        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;

        input.resize(window_size_samples);
        input_node_dims[0] = 1;
        input_node_dims[1] = window_size_samples;

        _state.resize(size_state);
        sr.resize(1);
        sr[0] = sample_rate;
    };
};


#define drwav_min(a, b)                    (((a) < (b)) ? (a) : (b))
#define drwav_max(a, b)                    (((a) > (b)) ? (a) : (b))
#define drwav_clamp(x, lo, hi)             (drwav_max((lo), drwav_min((hi), (x)))) 
class RNNoiseIterator
{
private:
    DenoiseState* pRnnoise = NULL;
public:
    RNNoiseIterator() :pRnnoise(NULL) {
        pRnnoise = rnnoise_create(NULL);
    }
    ~RNNoiseIterator()
    {
        if (!pRnnoise) {
            return;
        }

        rnnoise_destroy(pRnnoise);
    }
    float process_frame(float* pFrameOut, float* pFrameIn) {
        unsigned int n;
        float vadProb;
        float buffer[RNNOISE_BUFFER_SIZE];

        // Note: Be careful for the format of the input data.
        for (n = 0; n < RNNOISE_BUFFER_SIZE; ++n) {
            buffer[n] = pFrameIn[n] * 32768.0f;
        }

        vadProb = rnnoise_process_frame(this->pRnnoise, buffer, buffer);
        for (n = 0; n < RNNOISE_BUFFER_SIZE; ++n) {
            pFrameOut[n] = drwav_clamp(buffer[n], -32768, 32767) * (1.0f / 32768.0f);
        }
        return vadProb;
    }
    void process(float* buffer, size_t size) {
        for (size_t i = 0; i < size / RNNOISE_BUFFER_SIZE; i++)
        {
            process_frame(buffer + i * RNNOISE_BUFFER_SIZE, buffer + i * RNNOISE_BUFFER_SIZE);
        }
    }
    inline bool available()
    {
        return (pRnnoise != NULL);
    }

};

template<typename T> 
class AudioBuffer : public CircularQueue<T>
{
	public:
    	AudioBuffer(size_t capacity, bool enable_rnnoise = true, int input_channel=2, double input_sample_rate=48000.0) : CircularQueue<T>(capacity), 
            _input_channel_(input_channel),
            _input_sample_rate_(input_sample_rate),
            srcState(NULL), 
            resample_outputBuffer(NULL), 
            m_un8SaveAudioFlag(0)
        {
            _input_channel_ = input_channel;
            _input_sample_rate_ = input_sample_rate;
			srcState = src_new(SRC_SINC_BEST_QUALITY, 1, NULL);
			if (!srcState) {
				cout << "AudioBuffer::srcState was NULL!" << endl;
			}

			resample_outputBuffer = (float *)std::malloc((size_t)_input_sample_rate_ *2*sizeof(float));
			if( resample_outputBuffer == NULL )
			{
				cout << "AudioBuffer::resample_outputBuffer was NULL!" << endl;
			}

			src_data.data_in = NULL;
			src_data.data_out = resample_outputBuffer;
			src_data.input_frames = 0;
			src_data.output_frames = 0;
			src_data.end_of_input = 0;
			src_data.src_ratio = (double)OUTPOUT_SAMPLE_RATE / _input_sample_rate_;


            m_bEnableRnnoise = enable_rnnoise;
            if (m_bEnableRnnoise)
            { 
                if (!rnnoise.available())
                {
                    cout << "AudioBuffer::RNNoise was NULL!" << endl;
                }
            }
		};
		~AudioBuffer() {
			if(resample_outputBuffer) {
				free(resample_outputBuffer);
				resample_outputBuffer = NULL;
			}
			if(srcState) {
    			src_delete(srcState);
    			srcState = NULL;
			}
            if (m_un8SaveAudioFlag & SAVE_AUDIO_RAW)
            {
                wavWriterRaw.close();
            }
            if (this->m_bEnableRnnoise && m_un8SaveAudioFlag & SAVE_AUDIO_RNNOISED)
            {
                wavWriterRnnoised.close();
            }
            if (m_un8SaveAudioFlag & SAVE_AUDIO_RESAMPLED)
            {
                wavWriterResampled.close();
            }
            if (m_un8SaveAudioFlag & SAVE_AUDIO_VAD)
            {
                wavWriterVad.close();
            }

		 };
		
		int RecordCallback(const void* pInputBuffer, 
							void* pOutputBuffer, 
							unsigned long iFramesPerBuffer, 
							const PaStreamCallbackTimeInfo* timeInfo, 
							PaStreamCallbackFlags statusFlags)
		{
			int error = 0;
			if (pInputBuffer == NULL)
			{
				cout << "AudioBuffer::RecordCallback, input buffer was NULL!" << endl;
				return paContinue;
			}

			T** pData = (T**) pInputBuffer;

            if (m_un8SaveAudioFlag & SAVE_AUDIO_RAW)
            {
                wavWriterRaw.write(pData[0], iFramesPerBuffer);
            }

            if (m_bEnableRnnoise && rnnoise.available())
            {
                rnnoise.process(pData[0], iFramesPerBuffer);
            }


            if (m_bEnableRnnoise && m_un8SaveAudioFlag & SAVE_AUDIO_RNNOISED)
            {
                this->wavWriterRnnoised.write(pData[0], iFramesPerBuffer);// Copy all the frames over to our internal vector of samples
            }
            src_data.data_in = pData[0];
            src_data.input_frames = iFramesPerBuffer; 
            src_data.output_frames = iFramesPerBuffer;  
            if ((error = src_process(srcState, &(src_data)))) {
                std::cout << "Error: " << src_strerror(error) << std::endl;
            }

            if (m_un8SaveAudioFlag & SAVE_AUDIO_RESAMPLED)
            {
                this->wavWriterResampled.write(src_data.data_out, src_data.output_frames_gen);
            }

			this->enqueue(src_data.data_out, src_data.output_frames_gen);

			return paContinue;
		};

		int PlaybackCallback(const void* pInputBuffer, 
							void* pOutputBuffer, 
							unsigned long iFramesPerBuffer, 
							const PaStreamCallbackTimeInfo* timeInfo, 
							PaStreamCallbackFlags statusFlags)
		{
			if (pOutputBuffer == NULL)
			{
				cout << "AudioBuffer::PlaybackCallback was NULL!" << endl;
				return paComplete;
			}

			T** pData = (T**) pOutputBuffer;
			//std::cout << " dequeue checking" << std::endl;
			//std::cout << *(pData+0) << " " << *(pData+1) << " ";
			//std::cout << *(pData+2) << " " << *(pData+3) << std::endl;
			int osize = this->size();
			if(osize < iFramesPerBuffer){
				this->dequeue(pData[0],osize);
				return paComplete;
			} else {
				this->dequeue(pData[0],iFramesPerBuffer);
			}


			//std::cout << *(pData+0) << " " << *(pData+1) << " ";
			//std::cout << *(pData+2) << " " << *(pData+3) << std::endl;

			return paContinue;
		};

        inline std::tm localtime_xp(std::time_t timer)
        {
            std::tm bt{};
#if defined(__unix__)
            localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
            localtime_s(&bt, &timer);
#else
            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);
            bt = *std::localtime(&timer);
#endif
            return bt;
        }

        void setSaveAudioFlag(uint8_t save_audio)
        {
            auto now = localtime_xp(std::time(0));
            char buffer[80];
  
            strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", &now);

            m_un8SaveAudioFlag = save_audio;
            if(m_un8SaveAudioFlag & SAVE_AUDIO_RAW)
            {
                std::string filename = std::string(buffer) + "_raw.wav";
                wavWriterRaw.open(filename, (uint32_t)(this->_input_sample_rate_), 32, 1);
            }
            if (this->m_bEnableRnnoise && m_un8SaveAudioFlag & SAVE_AUDIO_RNNOISED)
            {
                std::string filename = std::string(buffer) + "_rnnoise.wav";
                wavWriterRnnoised.open(filename, (uint32_t)(this->_input_sample_rate_), 32, 1);
            }
            if (m_un8SaveAudioFlag & SAVE_AUDIO_RESAMPLED)
            {
                std::string filename = std::string(buffer) + "_resample.wav";
                wavWriterResampled.open(filename, (uint32_t)OUTPOUT_SAMPLE_RATE, 32, 1);
            }
            if (m_un8SaveAudioFlag & SAVE_AUDIO_VAD)
            {
                std::string filename = std::string(buffer) + "_vad.wav";
                wavWriterVad.open(filename, (uint32_t)OUTPOUT_SAMPLE_RATE, 32, 1);
            }
        }

	private:
		int _input_channel_;
		double _input_sample_rate_;
        // Sample Rate conveter
		SRC_STATE          *srcState;
    	float              *resample_outputBuffer;
    	SRC_DATA            src_data;
        // RNN-based noise suppression
        bool m_bEnableRnnoise;
        RNNoiseIterator rnnoise;

public:
        uint8_t m_un8SaveAudioFlag;

        wav_writer wavWriterRaw;
        wav_writer wavWriterRnnoised;
        wav_writer wavWriterResampled;
        wav_writer wavWriterVad;
};
//
// PortAudio Audio capture
//


class audio_async {
    
public:
    audio_async(); 
    ~audio_async();

    bool init(int iInputDevice, uint8_t save_audio = 0, bool enable_rnnoise = true);

    // start capturing audio via the provided SDL callback
    // keep last len_ms seconds of audio in a circular buffer
    bool resume();
    bool pause();
    void close();
    bool clear();
    double memory_usage_info() {
        return (double)m_pAudioBuffer->size()/m_pAudioBuffer->capacity();
    }


    // get audio data from the circular buffer
    bool get(int frames, std::vector<float> & audio);

    void write_vad_audio(const float * const data, size_t length)
    {
        if (m_pAudioBuffer->m_un8SaveAudioFlag & SAVE_AUDIO_VAD)
        {
            m_pAudioBuffer->wavWriterVad.write(data, length);
        }
    }

private:

	void print_device_info();
    portaudio::AutoSystem mautoSys;
    portaudio::DirectionSpecificStreamParameters * m_pInParamsRecord;
    portaudio::StreamParameters * m_pParamsRecord;
    portaudio::MemFunCallbackStream<AudioBuffer<float>> * m_pStreamRecord;
    volatile std::atomic_bool m_running;
    AudioBuffer<float> * m_pAudioBuffer;
    portaudio::System *m_psys;
};