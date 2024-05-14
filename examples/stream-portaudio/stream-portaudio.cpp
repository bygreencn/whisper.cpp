// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-portaudio.h"
#include "common.h"
#include "whisper.h"
#include <signal.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <sndfile.h>


static bool is_running = true;
void exit_handler(int signo) {
    if (signo == SIGINT)
        printf("received SIGINT, while exit soon\n");
    signal(signo, SIG_IGN);
    is_running = false;
}


//  500 -> 00:05.000
// 6000 -> 01:00.000
/*
std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);

    return std::string(buf);
}
*/
// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(2, (int32_t) std::thread::hardware_concurrency());
    double step_s    = 15;
    double length_s  = 30;
    double keep_s    = 0.0;
    int32_t capture_id = 10;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 768;

    float vad_thold    = 0.5f;
    float freq_thold   = 200.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = true;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = true;
    bool tinydiarize   = false;
    uint8_t save_audio    = 0; // save audio to wav file
    bool use_gpu       = false;

    std::string language  = "zh";
    std::string model     = "ggml-base-q5_1.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_s       = std::stod(argv[++i]); }
        else if (                  arg == "--length")        { params.length_s     = std::stod(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_s       = std::stod(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"   || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = std::stoi(argv[++i]); }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7.1f] audio step size in seconds\n",                   params.step_s);
    fprintf(stderr, "            --length N      [%-7.1f] audio length in seconds\n",                      params.length_s);
    fprintf(stderr, "            --keep N        [%-7.1f] audio to keep from previous step in seconds\n",  params.keep_s);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n",        params.speed_up ? "true" : "false");
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7x] save the recorded audio to a file\n",              params.save_audio);
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {

    signal(SIGINT, exit_handler);

    whisper_params params;
    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }



    params.keep_s   = std::min(params.keep_s,   params.step_s);
    params.length_s = std::max(params.length_s, params.step_s);

    const int n_samples_step = (int)((params.step_s  )*WHISPER_SAMPLE_RATE);
    const int n_samples_len  = (int)((params.length_s)*WHISPER_SAMPLE_RATE);
    const int n_samples_keep = (int)((params.keep_s  )*WHISPER_SAMPLE_RATE);
    const int n_samples_30s  = (int)((30.0           )*WHISPER_SAMPLE_RATE);
    const int vadleast_n_samples_len = n_samples_len;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    //const int n_new_line = !use_vad ? std::max(1, int(params.length_s / params.step_s - 1)) : 1; // number of steps to print new line

    //params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio

    audio_async audio;
    if (!audio.init(params.capture_id, params.save_audio)) {
        printf("%s: audio.init() failed!\n", __func__);
        return 1;
    }

    // init silero vad
    VadIterator silero_vad(L"silero_vad.onnx");

    audio.resume();

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        printf("error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams;
    cparams.use_gpu = params.use_gpu;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        printf("\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                printf("%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        printf("%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                params.step_s,
                params.length_s ,
                params.keep_s,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            //printf("%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
            printf("%s: no_context = %d\n", __func__, params.no_context);
        } else {
            printf("%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        printf("\n");
    }

    int n_iter = 0;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            printf("%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    printf("[Start speaking]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // main audio loop
    while (is_running) {

        // process new audio
        if (!use_vad) {
            while (is_running) {
                if (false == audio.get(n_samples_step, pcmf32_new))
                    continue;
                
                size_t size_pcmf32_new = pcmf32_new.size();
                if (size_pcmf32_new > 2*n_samples_step) {
                    //printf("\n\n1.%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    //audio.clear();
                    //continue;
                    break;
                }

                if ( size_pcmf32_new >= n_samples_step) {
                    //printf("\n\n2.%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    //audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = (int)pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.clear();
            pcmf32.insert(pcmf32.end(), pcmf32_old.end() - n_samples_take, pcmf32_old.end());
            pcmf32.insert(pcmf32.end(), pcmf32_new.begin(), pcmf32_new.end());
            pcmf32_old = pcmf32;
        } else {
            //const auto t_now  = std::chrono::high_resolution_clock::now();
            //const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();
            //if (t_diff < 2000) {
            //    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            //    continue;
            //}
            pcmf32.clear();
            
            while(is_running)
            {
                if (false == audio.get((int)(n_samples_len), pcmf32_new))
                    continue;
                silero_vad.process(pcmf32_new, pcmf32_old);
                pcmf32.insert(pcmf32.end(), pcmf32_old.begin(), pcmf32_old.end());
                if (pcmf32.size() >= vadleast_n_samples_len)
                    break;
            }

            //t_last = t_now;
        }

        if (!is_running) {
            break;
        }

        printf("\n***memory usage : %7.6f\n",audio.memory_usage_info());

        if (params.save_audio && SAVE_AUDIO_VAD)
        {
            audio.write_vad_audio(pcmf32.data(), pcmf32.size());
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;

            wparams.audio_ctx        = params.audio_ctx;
            wparams.speed_up         = params.speed_up;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            wparams.temperature_inc  = -1.0f;
            //wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : (int)prompt_tokens.size();


            wparams.n_max_text_ctx = 0;


            {
                wparams.abort_callback = [](void* user_data) {
                    bool is_aborted = ((*(bool*)user_data) == false);
                    return is_aborted;
                };
                wparams.abort_callback_user_data = &is_running;
            }

            if (whisper_full(ctx, wparams, pcmf32.data(), (int)pcmf32.size()) != 0) {
                std::cout << "aborted to process audio\n" << std::endl;
                break;
            }


            // print result;
            {
                /*
                if (use_vad) {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                }
                */
                printf("\n");

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        std::cout << "**" << text << std::endl;
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0) + " --> " + to_timestamp(t1) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        std::cout << output;
                        fflush(stdout);
                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                /*
                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
                */
            }

            ++n_iter;

            //if (!use_vad && (n_iter % n_new_line) == 0) {
            if (!use_vad) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();
                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
        }
    }
	
	

    audio.pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    audio.close();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    

    return 0;
}
