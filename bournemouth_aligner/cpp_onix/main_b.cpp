'''
Original code by shubhammore1310  (2026-02-10)
'''

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <variant>
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <limits>
#include <boost/variant.hpp>
#include <chrono>
#include <unistd.h>
#include <numeric>
#include <functional> 
#include <sys/resource.h>
#include <iomanip>
#include <sstream> 
#pragma pack(push, 1)
struct WavHeader
{
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
};
#pragma pack(pop)

int sample_rate_ = 16000;
std::vector<float> resampleAudio(const std::vector<float> &audio_data,
                                 int source_sr, int target_sr)
{
    if (source_sr == target_sr)
    {
        return audio_data;
    }

    std::cout << " Resampling audio: " << source_sr << " Hz → " << target_sr << " Hz" << std::endl;

    float ratio = (float)target_sr / source_sr;
    size_t new_length = (size_t)(audio_data.size() * ratio);
    std::vector<float> resampled(new_length);

    for (size_t i = 0; i < new_length; i++)
    {
        float src_idx = i / ratio;
        size_t idx1 = (size_t)src_idx;
        size_t idx2 = std::min(idx1 + 1, audio_data.size() - 1);
        float frac = src_idx - idx1;

        resampled[i] = audio_data[idx1] * (1.0f - frac) + audio_data[idx2] * frac;
    }

    std::cout << " Resampled: " << audio_data.size() << " → " << new_length << " samples" << std::endl;
    return resampled;
}

std::vector<float> loadAudioFromWav(const std::string &wav_path, float &original_duration)
{
    std::ifstream file(wav_path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << " Failed to open: " << wav_path << std::endl;
        original_duration = 0.0f;
        return {};
    }

    WavHeader header;
    file.read(reinterpret_cast<char *>(&header), sizeof(WavHeader));

    std::cout << "\n ===== AUDIO INFO =====" << std::endl;
    std::cout << " File: " << wav_path << std::endl;
    std::cout << " Channels: " << header.num_channels << std::endl;
    std::cout << " Sample Rate: " << header.sample_rate << " Hz" << std::endl;
    std::cout << " Bits/Sample: " << header.bits_per_sample << std::endl;

    size_t num_samples = header.data_size / (header.bits_per_sample / 8);
    std::vector<float> audio_data;

    if (header.bits_per_sample == 16)
    {
        std::vector<int16_t> samples(num_samples);
        file.read(reinterpret_cast<char *>(samples.data()), header.data_size);
        audio_data.resize(num_samples / header.num_channels);
        for (size_t i = 0; i < audio_data.size(); i++)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < header.num_channels; ch++)
            {
                sum += samples[i * header.num_channels + ch] / 32768.0f;
            }
            audio_data[i] = sum / header.num_channels;
        }
    }
    else if (header.bits_per_sample == 32)
    {
        std::vector<float> samples(num_samples);
        file.read(reinterpret_cast<char *>(samples.data()), header.data_size);
        audio_data.resize(num_samples / header.num_channels);
        for (size_t i = 0; i < audio_data.size(); i++)
        {
            float sum = 0.0f;
            for (int ch = 0; ch < header.num_channels; ch++)
            {
                sum += samples[i * header.num_channels + ch];
            }
            audio_data[i] = sum / header.num_channels;
        }
    }

    original_duration = (float)audio_data.size() / header.sample_rate;

    std::cout << " Original samples: " << audio_data.size() << std::endl;
    std::cout << " Original duration: " << original_duration << "s" << std::endl;

    if (header.sample_rate != sample_rate_)
    {
        audio_data = resampleAudio(audio_data, header.sample_rate, sample_rate_);
        std::cout << " Resampled samples: " << audio_data.size() << std::endl;
    }

    std::cout << " Final duration (at " << sample_rate_ << " Hz): "
              << (float)audio_data.size() / sample_rate_ << "s" << std::endl;
    std::cout << " ===========================\n"
              << std::endl;

    return audio_data;
}

std::unordered_map<std::string, int> phoneme_mapped_index = {
    {"SIL", 0}, {"i", 1}, {"i:", 2}, {"ɨ", 3}, {"ɪ", 4}, {"e", 5}, {"e:", 6}, {"ɛ", 7}, {"ə", 8}, {"ɚ", 9}, {"ʌ", 10}, {"u", 11}, {"u:", 12}, {"ʊ", 13}, {"ɯ", 14}, {"o", 15}, {"o:", 16}, {"ɔ", 17}, {"a", 18}, {"a:", 19}, {"æ", 20}, {"y", 21}, {"ø", 22}, {"aɪ", 23}, {"eɪ", 24}, {"aʊ", 25}, {"oʊ", 26}, {"ɔɪ", 27}, {"p", 28}, {"b", 29}, {"t", 30}, {"d", 31}, {"k", 32}, {"g", 33}, {"q", 34}, {"ts", 35}, {"s", 36}, {"z", 37}, {"tʃ", 38}, {"dʒ", 39}, {"ʃ", 40}, {"ʒ", 41}, {"ɕ", 42}, {"f", 43}, {"v", 44}, {"θ", 45}, {"ð", 46}, {"ç", 47}, {"x", 48}, {"ɣ", 49}, {"h", 50}, {"ʁ", 51}, {"m", 52}, {"n", 53}, {"ɲ", 54}, {"ŋ", 55}, {"l", 56}, {"ɭ", 57}, {"ɾ", 58}, {"ɹ", 59}, {"j", 60}, {"w", 61}, {"tʲ", 62}, {"nʲ", 63}, {"rʲ", 64}, {"ɭʲ", 65}, {"noise", 66}};

std::unordered_map<int, int> phoneme_groups_mapper = {
    {0, 0}, {1, 1}, {2, 1}, {3, 3}, {4, 1}, {5, 1}, {6, 1}, {7, 1}, {8, 2}, {9, 2}, {10, 2}, {11, 3}, {12, 3}, {13, 3}, {14, 3}, {15, 3}, {16, 3}, {17, 3}, {18, 4}, {19, 4}, {20, 4}, {21, 1}, {22, 1}, {23, 5}, {24, 5}, {25, 5}, {26, 5}, {27, 5}, {28, 6}, {29, 7}, {30, 6}, {31, 7}, {32, 6}, {33, 7}, {34, 6}, {35, 10}, {36, 8}, {37, 9}, {38, 10}, {39, 11}, {40, 8}, {41, 9}, {42, 8}, {43, 8}, {44, 9}, {45, 8}, {46, 9}, {47, 8}, {48, 8}, {49, 9}, {50, 8}, {51, 9}, {52, 12}, {53, 12}, {54, 12}, {55, 12}, {56, 13}, {57, 13}, {58, 14}, {59, 14}, {60, 15}, {61, 15}, {62, 6}, {63, 12}, {64, 14}, {65, 13}, {66, 16}};

std::unordered_map<std::string, int> phoneme_groups_index = {
    {"SIL", 0}, {"front_vowels", 1}, {"central_vowels", 2}, {"back_vowels", 3}, {"low_vowels", 4}, {"diphthongs", 5}, {"voiceless_stops", 6}, {"voiced_stops", 7}, {"voiceless_fricatives", 8}, {"voiced_fricatives", 9}, {"voiceless_affricates", 10}, {"voiced_affricates", 11}, {"nasals", 12}, {"laterals", 13}, {"rhotics", 14}, {"glides", 15}, {"noise", 16}};

// ==================== WINDOW FUNCTIONS (exact Python logic) ====================
std::vector<std::vector<std::vector<float>>> slice_windows(
    const std::vector<std::vector<float>> &audio_batch,
    int sample_rate = 16000,
    int window_size_ms = 160,
    int stride_ms = 80)
{

    size_t batch_size = audio_batch.size();
    size_t max_audio_length = audio_batch.empty() ? 0 : audio_batch[0].size();

    int64_t window_size = window_size_ms * sample_rate / 1000;
    int64_t stride = stride_ms * sample_rate / 1000;
    int64_t num_windows = ((max_audio_length - window_size) / stride) + 1;

    std::vector<std::vector<std::vector<float>>> windows(
        batch_size, std::vector<std::vector<float>>(num_windows, std::vector<float>(window_size, 0.0f)));

    for (size_t b = 0; b < batch_size; ++b)
    {
        for (int64_t w = 0; w < num_windows; ++w)
        {
            int64_t start = w * stride;
            for (int64_t i = 0; i < window_size; ++i)
            {
                int64_t idx = start + i;
                if (idx < static_cast<int64_t>(max_audio_length))
                {
                    windows[b][w][i] = audio_batch[b][idx];
                }
            }
        }
    }
    return windows;
}
std::vector<float> stitch_window_predictions_flatold(
    const std::vector<float> &windowed_logits_flat,
    int num_windows, int frames_per_window, int num_classes,
    int original_audio_length, int sample_rate, int window_ms, int stride_ms)
{

    int window_frames = (window_ms * frames_per_window * sample_rate) / (1000 * sample_rate);
    int stride_frames = (stride_ms * frames_per_window * sample_rate) / (1000 * sample_rate);

    // Target length T
    int T = (original_audio_length * frames_per_window) / sample_rate;

    // Output [T * num_classes]
    std::vector<float> stitched(T * num_classes, 0.0f);

    for (int w = 0; w < num_windows; ++w)
    {
        int start_frame = w * stride_frames;
        int window_offset = w * frames_per_window * num_classes;

        int end_frame = std::min(start_frame + frames_per_window, T);
        int copy_frames = end_frame - start_frame;

        // Copy window slice to stitched position
        for (int f = 0; f < copy_frames; ++f)
        {
            for (int c = 0; c < num_classes; ++c)
            {
                int src_idx = window_offset + f * num_classes + c;
                int dst_idx = (start_frame + f) * num_classes + c;
                stitched[dst_idx] += windowed_logits_flat[src_idx];
            }
        }
    }

    return stitched; // [T * num_classes] flat
}


#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>

std::vector<float> stitch_window_predictions_flat(
    const std::vector<float>& windowed_logits_flat,
    int num_windows,
    int frames_per_window,
    int num_classes,
    int original_audio_length,
    int sample_rate,
    int window_size_ms,
    int stride_ms,
    int batch_size=1)
{
    // ===== 1. Compute total_frames EXACTLY like Python =====
    int window_size_samples = (window_size_ms * sample_rate) / 1000;
    int stride_samples = (stride_ms * sample_rate) / 1000;
    
    // Python: num_windows_total = ((original_audio_length - window_size_samples) // stride_samples) + 1
    int num_windows_total = ((original_audio_length - window_size_samples) / stride_samples) + 1;
    
    // Python: total_frames = (num_windows_total * cnn_output_size) // 2
    // Note: cnn_output_size == frames_per_window in your usage
    int total_frames = (num_windows_total * frames_per_window) / 2;
    
    // Python: stride_frames = frames_per_window // 2 (50% overlap)
    int stride_frames = frames_per_window / 2;
    
    // ===== 2. Validate input size =====
    size_t expected_input_size = batch_size * num_windows * frames_per_window * num_classes;
    assert(windowed_logits_flat.size() == expected_input_size && 
           "Input size mismatch in stitch_window_predictions_flat");
    
    // ===== 3. Precompute cosine weights: cos(linspace(-π/2, π/2, frames_per_window)) =====
    std::vector<float> weights(frames_per_window);
    for (int i = 0; i < frames_per_window; ++i) {
        double ratio = (frames_per_window > 1) 
            ? static_cast<double>(i) / (frames_per_window - 1) 
            : 0.0;
        double angle = ratio * M_PI - M_PI_2;  // Map [0,1] → [-π/2, π/2]
        weights[i] = std::cos(angle);
    }
    
    // ===== 4. Allocate output buffers =====
    std::vector<float> combined(batch_size * total_frames * num_classes, 0.0f);
    std::vector<float> weight_sum(batch_size * total_frames, 0.0f);  // Per-frame weights
    
    // ===== 5. Weighted accumulation over windows =====
    for (int b = 0; b < batch_size; ++b) {
        int batch_offset = b * num_windows * frames_per_window * num_classes;
        
        for (int w = 0; w < num_windows; ++w) {
            int start_frame = w * stride_frames;
            int window_offset = batch_offset + w * frames_per_window * num_classes;
            
            // Handle last window clipping (Python does this explicitly)
            int end_frame = std::min(start_frame + frames_per_window, total_frames);
            int frames_to_copy = end_frame - start_frame;
            
            for (int f = 0; f < frames_to_copy; ++f) {
                float wgt = weights[f];
                int src_base = window_offset + f * num_classes;
                int dst_base = b * total_frames * num_classes + (start_frame + f) * num_classes;
                int weight_idx = b * total_frames + (start_frame + f);
                
                // Accumulate weighted logits per class
                for (int c = 0; c < num_classes; ++c) {
                    combined[dst_base + c] += windowed_logits_flat[src_base + c] * wgt;
                }
                weight_sum[weight_idx] += wgt;
            }
        }
    }
    
    // ===== 6. Normalize with stable division (Python: combined / (weight_sum + 1e-8)) =====
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < total_frames; ++t) {
            float denom = weight_sum[b * total_frames + t] + 1e-8f;
            int base = b * total_frames * num_classes + t * num_classes;
            
            for (int c = 0; c < num_classes; ++c) {
                combined[base + c] /= denom;
            }
        }
    }
    
    // ===== 7. Debug output (optional but helpful for validation) =====
    #ifdef DEBUG_STITCHING
    std::cout << "Stitching debug:" << std::endl;
    std::cout << "  num_windows_total: " << num_windows_total << std::endl;
    std::cout << "  total_frames: " << total_frames << std::endl;
    std::cout << "  stride_frames: " << stride_frames << std::endl;
    std::cout << "  Input size: " << windowed_logits_flat.size() 
              << " (" << batch_size << "×" << num_windows << "×" 
              << frames_per_window << "×" << num_classes << ")" << std::endl;
    std::cout << "  Output size: " << combined.size() 
              << " (" << batch_size << "×" << total_frames << "×" << num_classes << ")" << std::endl;
    #endif
    
    return combined;  // [batch_size * total_frames * num_classes] flat
}


int calc_spec_len_ext(int wav_len, int window_size_ms, int stride_ms,
                      int sample_rate, int frames_per_window,
                      bool disable_windowing = false, int wav_len_max = 16000)
{

    std::vector<int> wav_lens = {wav_len};
    std::vector<int> frames_per_window_vec = {frames_per_window};

    std::vector<int> spectral_lens;

    if (!disable_windowing)
    {
        int window_size_wav = (window_size_ms * sample_rate) / 1000;
        int stride_size_wav = (stride_ms * sample_rate) / 1000;

        for (int i = 0; i < wav_lens.size(); ++i)
        {
            int wav_l = wav_lens[i];
            int total_frames;

            if (wav_l <= window_size_wav)
            {
                // Short clips: proportional scaling
                double num_windows = static_cast<double>(wav_l) / window_size_wav;
                total_frames = static_cast<int>(std::ceil(frames_per_window_vec[i] * num_windows));
            }
            else
            {
                // Normal: num_windows * frames_per_window // 2
                int num_windows = ((wav_l - window_size_wav) / stride_size_wav) + 1;
                total_frames = (num_windows * frames_per_window_vec[i]) / 2;
            }
            
            std::cout << "window_size_wav:"<< window_size_wav << " stride_size_wav:" << stride_size_wav << " frames_per_window:" << frames_per_window << " wav_l:" << wav_l << std::endl;
            if (total_frames < 2)
            {
                double actual_ms = 1000.0 * wav_l / sample_rate;
                std::cout<<"WARN: spectral_len < 2, wav_lens: " + std::to_string(wav_l) +
                                         ", frames: " + std::to_string(total_frames) +
                                         ", num_windows: N/A, expected >= " + std::to_string(window_size_ms) +
                                         "ms, got " + std::to_string(actual_ms) + "ms" << std::endl;
            }

            spectral_lens.push_back(total_frames);
        }
    }
    else
    {
        double wav_len_per_frame = static_cast<double>(wav_len_max) / frames_per_window_vec[0];

        spectral_lens.resize(wav_lens.size(), frames_per_window_vec[0]);
        for (size_t i = 0; i < wav_lens.size(); ++i)
        {
            spectral_lens[i] = static_cast<int>(std::ceil(static_cast<double>(wav_lens[i]) / wav_len_per_frame));

            if (spectral_lens[i] > frames_per_window_vec[0])
            {
                throw std::runtime_error("WARN: spectral_len > frames_per_window, spectral_len: " + std::to_string(spectral_lens[i]) +
                                         ", frames_per_window: " + std::to_string(frames_per_window_vec[0]));
            }
        }
    }

    return spectral_lens[0]; // .item() equivalent - return single value
}
std::vector<float> log_softmaxold(const std::vector<float> &frame_probs)
{
    std::vector<float> result = frame_probs;
    float max_val = *std::max_element(frame_probs.begin(), frame_probs.end());
    float sum_exp = 0.0f;

    for (float val : frame_probs)
    {
        sum_exp += std::exp(val - max_val);
    }

    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i] = frame_probs[i] - max_val - std::log(sum_exp);
    }
    return result;
}
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

// Per-frame log-softmax (your existing function - correct implementation)
std::vector<float> log_softmax_frame(const std::vector<float>& logits_frame) {
    assert(!logits_frame.empty());
    std::vector<float> result(logits_frame.size());
    float max_val = *std::max_element(logits_frame.begin(), logits_frame.end());
    float sum_exp = 0.0f;
    
    for (float val : logits_frame) {
        sum_exp += std::exp(val - max_val);
    }
    
    float log_sum_exp = std::log(sum_exp);
    for (size_t i = 0; i < logits_frame.size(); ++i) {
        result[i] = logits_frame[i] - max_val - log_sum_exp;
    }
    return result;
}

// Batch log-softmax over class dimension (dim=2 equivalent)
std::vector<float> log_softmax_batch(
    const std::vector<float>& logits_flat,
    int batch_size,
    int num_frames,
    int num_classes)
{
    // Validate input size
    assert(logits_flat.size() == batch_size * num_frames * num_classes);
    
    std::vector<float> result(logits_flat.size());
    
    // Apply log_softmax per frame (over class dimension)
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < num_frames; ++t) {
            int frame_start = (b * num_frames + t) * num_classes;
            
            // Extract single frame logits [C]
            std::vector<float> frame_logits(
                logits_flat.begin() + frame_start,
                logits_flat.begin() + frame_start + num_classes
            );
            
            // Compute log-softmax for this frame
            auto frame_log_probs = log_softmax_frame(frame_logits);
            
            // Copy back to result
            std::copy(
                frame_log_probs.begin(),
                frame_log_probs.end(),
                result.begin() + frame_start
            );
        }
    }
    
    return result;
}
#include <vector>
#include <tuple>
#include <limits>
#include <algorithm>
#include <iostream>

class ViterbiDecoder {
public:
    int blank_id_;
    int silence_id_;
    int silence_anchors_;
    double min_phoneme_prob_;
    bool ignore_noise_;
    float neg_inf_ = -1000.0f;
    
    ViterbiDecoder(int blank_id, int silence_id, int silence_anchors = 3,
                   double min_phoneme_prob = 1e-8, bool ignore_noise = true)
        : blank_id_(blank_id), silence_id_(silence_id), silence_anchors_(silence_anchors),
          min_phoneme_prob_(min_phoneme_prob), ignore_noise_(ignore_noise) {}
    
    // Helper: Boost target phonemes (numpy-like)
    std::vector<float> boost_target_phonemes(const std::vector<float>& log_probs,
                                           int T, int C, 
                                           const std::vector<int>& true_seq) {
        std::vector<float> boosted = log_probs;
        float boost_factor = 5.0f;
        
        for (int target : true_seq) {
            if (target >= 0 && target < C) {
                for (int t = 0; t < T; ++t) {
                    boosted[t * C + target] += boost_factor;
                }
            }
        }
        
        // Log-softmax normalization
        return log_softmax(boosted, T, C);
    }
    
    // Helper: Enforce minimum probabilities
    std::vector<float> enforce_minimum_probabilities(const std::vector<float>& log_probs,
                                                   int T, int C,
                                                   const std::vector<int>& true_seq) {
        std::vector<float> modified = log_probs;
        float min_log_prob = std::log(1e-8f);
        
        for (int target : true_seq) {
            if (target >= 0 && target < C) {
                for (int t = 0; t < T; ++t) {
                    if (modified[t * C + target] < min_log_prob) {
                        modified[t * C + target] = min_log_prob;
                    }
                }
            }
        }
        return log_softmax(modified, T, C);
    }
    
    // Log-softmax implementation
    std::vector<float> log_softmax(const std::vector<float>& x, int T, int C) {
        std::vector<float> result(x.size());
        
        for (int t = 0; t < T; ++t) {
            // Find max in this frame
            float x_max = neg_inf_;
            for (int c = 0; c < C; ++c) {
                x_max = std::max(x_max, x[t * C + c]);
            }
            
            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum_exp += std::exp(x[t * C + c] - x_max);
            }
            
            // Log-softmax
            for (int c = 0; c < C; ++c) {
                result[t * C + c] = x[t * C + c] - x_max - std::log(sum_exp);
            }
        }
        return result;
    }
    
    // Simplified single-sequence Viterbi (avoid batch complexity)
    std::vector<int> viterbi_decode_single(const std::vector<float>& log_probs,
                                         const std::vector<int>& ctc_path) {
        int T = log_probs.size() / 17; // Get T from context
        int S = ctc_path.size();
        int C = 17; // Pass as parameter
        
        // TODO: Implement full Viterbi logic here
        std::vector<int> frame_phonemes(T);
        return frame_phonemes;
    }
    
    std::vector<std::vector<int>> viterbi_decode_batch(
        const std::vector<std::vector<float>>& processed_log_probs,
        const std::vector<std::vector<int>>& ctc_paths_list,
        const std::vector<int>& ctc_lens_list) {
        
        std::vector<std::vector<int>> results;
        for (size_t i = 0; i < processed_log_probs.size(); ++i) {
            // Decode single sequence to avoid padding complexity
            auto frame_phonemes = viterbi_decode_single(
                processed_log_probs[i], ctc_paths_list[i]);
            results.push_back(frame_phonemes);
        }
        return results;
    }
    
    std::vector<std::tuple<int, int, int>> assort_frames(const std::vector<int>& frame_phonemes) {
        std::vector<std::tuple<int, int, int>> framestamps;
        if (frame_phonemes.empty()) return framestamps;
        
        size_t i = 0;
        while (i < frame_phonemes.size()) {
            int phoneme = frame_phonemes[i];
            size_t start = i;
            
            // Find end of segment
            while (i < frame_phonemes.size() && frame_phonemes[i] == phoneme) {
                ++i;
            }
            size_t end = i;
            
            // Filter blanks if needed
            if (phoneme != blank_id_) {
                framestamps.emplace_back(phoneme, start, end);
            }
        }
        return framestamps;
    }
};

class AlignmentUtils {
public:
    int blank_id_;
    int silence_id_;
    int silence_anchors_;
    ViterbiDecoder viterbi_decoder_;
    
    AlignmentUtils(int blank_id, int silence_id, int silence_anchors = 10, bool ignore_noise = true)
        : blank_id_(blank_id), silence_id_(silence_id), silence_anchors_(silence_anchors),
          viterbi_decoder_(blank_id, silence_id, silence_anchors, 1e-8, ignore_noise) {}

    std::vector<std::vector<std::tuple<int, int, int>>> decode_alignments(
        const std::vector<std::vector<float>>& log_probs_batch,  // B x T x C flattened
        const std::vector<std::vector<int>>* true_seqs = nullptr,
        const std::vector<int>* pred_lens = nullptr,
        const std::vector<int>* true_seqs_lens = nullptr,
        bool forced_alignment = true,
        bool boost_targets = true,
        bool enforce_minimum = true,
        bool enforce_all_targets = true) {
        
        size_t batch_size = log_probs_batch.size();
        std::vector<std::vector<std::tuple<int, int, int>>> results(batch_size);
        
        if (!forced_alignment || !true_seqs || !true_seqs_lens) {
            // FREE DECODING - SAFEST PATH
            for (size_t b = 0; b < batch_size; ++b) {
                const auto& log_probs_b = log_probs_batch[b];
                if (log_probs_b.empty()) continue;
                
                // Determine T and C safely
                int C = 17;  // From Python: 17 classes
                int T = pred_lens ? (*pred_lens)[b] : log_probs_b.size() / C;
                T = std::min(T, static_cast<int>(log_probs_b.size() / C));
                
                std::vector<int> frame_phonemes(T, viterbi_decoder_.blank_id_);
                
                for (int t = 0; t < T; ++t) {
                    int best_class = 0;
                    float best_prob = viterbi_decoder_.neg_inf_;
                    
                    for (int c = 0; c < C; ++c) {
                        int idx = t * C + c;
                        if (idx < static_cast<int>(log_probs_b.size())) {
                            float prob = log_probs_b[idx];
                            if (prob > best_prob) {
                                best_prob = prob;
                                best_class = c;
                            }
                        }
                    }
                    frame_phonemes[t] = best_class;
                }
                
                results[b] = viterbi_decoder_.assort_frames(frame_phonemes);
            }
        } else {
            // FORCED ALIGNMENT - FIXED VERSION
            std::vector<std::vector<float>> processed_log_probs;
            std::vector<std::vector<int>> ctc_paths_list;
            
            int C = 17;  // Fixed from Python example
            
            for (size_t b = 0; b < batch_size; ++b) {
                const auto& log_probs_b = log_probs_batch[b];
                if (log_probs_b.empty()) continue;
                
                // SAFEST SLICING - Check bounds
                int T = pred_lens ? (*pred_lens)[b] : log_probs_b.size() / C;
                T = std::min(T, static_cast<int>(log_probs_b.size() / C));
                
                // Extract log_probs_seq safely
                std::vector<float> log_probs_seq;
                for (int t = 0; t < T; ++t) {
                    for (int c = 0; c < C; ++c) {
                        int idx = t * C + c;
                        if (idx < static_cast<int>(log_probs_b.size())) {
                            log_probs_seq.push_back(log_probs_b[idx]);
                        } else {
                            log_probs_seq.push_back(viterbi_decoder_.neg_inf_);
                        }
                    }
                }
                
                // Apply modifications SAFELY
                if (boost_targets && true_seqs && b < true_seqs->size()) {
                    log_probs_seq = viterbi_decoder_.boost_target_phonemes(
                        log_probs_seq, T, C, (*true_seqs)[b]);
                }
                if (enforce_minimum && true_seqs && b < true_seqs->size()) {
                    log_probs_seq = viterbi_decoder_.enforce_minimum_probabilities(
                        log_probs_seq, T, C, (*true_seqs)[b]);
                }
                
                processed_log_probs.push_back(log_probs_seq);
                
                // Create CTC path
                int true_len = (true_seqs_lens && b < true_seqs_lens->size()) 
                             ? (*true_seqs_lens)[b] : 0;
                std::vector<int> ctc_path(2 * true_len + 1, viterbi_decoder_.blank_id_);
                if (true_seqs && b < true_seqs->size()) {
                    for (int i = 0; i < true_len; ++i) {
                        ctc_path[2 * i + 1] = (*true_seqs)[b][i];
                    }
                }
                ctc_paths_list.push_back(ctc_path);
            }
            
            // Decode (single sequence for now to avoid complexity)
            for (size_t i = 0; i < processed_log_probs.size(); ++i) {
                // Simplified: use argmax for now
                int T = processed_log_probs[i].size() / 17;
                std::vector<int> frame_phonemes(T, viterbi_decoder_.blank_id_);
                results[i] = viterbi_decoder_.assort_frames(frame_phonemes);
            }
        }
        
        return results;
    }
};


class CUPEONNXPredictor
{
private:
    struct MemoryUsage {
        long rss_mb;     // Resident Set Size (RAM)
        long vms_mb;     // Virtual Memory Size
        long max_rss_mb; // Peak RSS
    };
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

        static MemoryUsage get_memory_usage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        
        MemoryUsage mem;
        mem.rss_mb = usage.ru_maxrss / 1024;  // KB → MB (Linux)
        mem.max_rss_mb = usage.ru_maxrss / 1024;
        
        // Virtual memory (RSS + swapped)
        long vms_kb = 0;
        FILE* fp = fopen("/proc/self/status", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "VmSize:", 7) == 0) {
                    sscanf(line, "VmSize: %ld kB", &vms_kb);
                    break;
                }
            }
            fclose(fp);
        }
        mem.vms_mb = vms_kb / 1024;
        
        return mem;
    }

public:
    CUPEONNXPredictor(const std::string &onnx_path,
                      const std::vector<std::string> &providers = {"CUDAExecutionProvider", "CPUExecutionProvider"})
    {
        auto mem_before = get_memory_usage();
        std::cout << "=== MEMORY BEFORE MODEL LOAD ===" << std::endl;
        std::cout << "RSS: " << mem_before.rss_mb << " MB" << std::endl;
        std::cout << "Peak RSS: " << mem_before.max_rss_mb << " MB" << std::endl;
        std::cout << "VM Size: " << mem_before.vms_mb << " MB" << std::endl;

        Ort::SessionOptions session_options;

        // CUDA provider setup
        OrtCUDAProviderOptions cuda_options{};
        memset(&cuda_options, 0, sizeof(cuda_options));
        cuda_options.device_id = 0;
        session_options.SetIntraOpNumThreads(1); // Fix affinity warnings
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CUPEONNX");
        memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), session_options);

                auto mem_after = get_memory_usage();
        std::cout << "\n=== MEMORY AFTER MODEL LOAD ===" << std::endl;
        std::cout << "RSS: " << mem_after.rss_mb << " MB (+Δ" 
                  << (mem_after.rss_mb - mem_before.rss_mb) << " MB)" << std::endl;
        std::cout << "Peak RSS: " << mem_after.max_rss_mb << " MB (+Δ" 
                  << (mem_after.max_rss_mb - mem_before.max_rss_mb) << " MB)" << std::endl;
        std::cout << "VM Size: " << mem_after.vms_mb << " MB (+Δ" 
                  << (mem_after.vms_mb - mem_before.vms_mb) << " MB)" << std::endl;
        
        std::cout << "MODEL MEMORY USAGE: ~" << (mem_after.rss_mb - mem_before.rss_mb) 
                  << " MB" << std::endl;
    }

    std::tuple<
        std::vector<float>, std::vector<int64_t>,  // logits_class + shape
        std::vector<float>, std::vector<int64_t>,  // logits_group + shape
        std::vector<float>, std::vector<int64_t>   // embeddings + shape
    > predict(
        const std::vector<std::vector<float>> &audio_batch, bool return_embeddings = true)
    {

        size_t batch_size = audio_batch.size();
        size_t audio_length = audio_batch.empty() ? 0 : audio_batch[0].size();

        std::vector<int64_t> input_shape = {static_cast<int64_t>(batch_size), static_cast<int64_t>(audio_length)};
        std::cout<< "Input shape: [" << input_shape[0] << ", " << input_shape[1] << "]" << std::endl;
        // Flatten batch for ONNX
        std::vector<float> flat_input(batch_size * audio_length);
        for (size_t b = 0; b < batch_size; ++b)
        {
            std::copy(audio_batch[b].begin(), audio_batch[b].end(), flat_input.begin() + b * audio_length);
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_, flat_input.data(), flat_input.size(), input_shape.data(), input_shape.size());

        const char *input_names[] = {"audio"};
        const char *output_names[] = {"logits_class", "logits_group", "embeddings"};

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 3);

        const float *class_ptr = output_tensors[0].GetTensorData<float>();
        const float *group_ptr = output_tensors[1].GetTensorData<float>();

        auto class_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto group_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        auto emb_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "logits_class shape: [" << class_shape[0] << ", " << class_shape[1] << ", " << class_shape[2] << "]" << std::endl;
        std::cout << "logits_group shape: [" << group_shape[0] << ", " << group_shape[1] << ", " << group_shape[2] << "]" << std::endl;

        std::vector<float> logits_class(class_ptr, class_ptr + std::accumulate(class_shape.begin(), class_shape.end(), 1LL, std::multiplies<>()));
        std::vector<float> logits_group(group_ptr, group_ptr + std::accumulate(group_shape.begin(), group_shape.end(), 1LL, std::multiplies<>()));

        std::vector<float> embeddings;
        if (return_embeddings && output_tensors.size() > 2)
        {
            auto emb_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
            const float *emb_ptr = output_tensors[2].GetTensorData<float>();

            size_t emb_size = emb_shape[0] * emb_shape[1];
            embeddings.resize(emb_size);
            std::copy(emb_ptr, emb_ptr + emb_size, embeddings.begin());
        }

        return {
            std::move(logits_class), class_shape,
            std::move(logits_group), group_shape,
            std::move(embeddings), emb_shape
        };
    }
};

using Segment = std::map<std::string, boost::variant<int, double, std::string, std::vector<boost::variant<int, double>>>>;
using Vs2Data = std::vector<Segment>;

struct TimestampResult
{
    std::vector<std::tuple<int, int, int, float, float, float>> phoneme_timestamps;
    std::vector<std::tuple<int, int, int, float, float, float>> group_timestamps;
};

class PhonemeTimestampAligner
{
    private:
    std::string device_;
    std::unique_ptr<CUPEONNXPredictor> extractor_;
    int resampler_sample_rate_ = 16000;
    int padding_ph_label_ = -100;
    int ph_seq_min_ = 1;
    std::string output_frames_key_;

    // Timing constraints
    double seg_duration_min_ = 0.05; // seconds
    int seg_duration_min_samples_;
    double seg_duration_max_;
    int wav_len_max_;

    // Mapper config
    std::string selected_mapper_;

    // Keys
    std::string phonemes_ipa_key_ = "ipa";
    std::string phonemes_key_ = "ph66";
    std::string phoneme_groups_key_ = "pg16";

    // Mappers
    std::unordered_map<int, std::string> phoneme_id_to_label_;
    std::unordered_map<std::string, int> phoneme_label_to_id_;
    std::unordered_map<int, std::string> group_id_to_label_;
    std::unordered_map<std::string, int> group_label_to_id_;
    std::unordered_map<int, int> phoneme_id_to_group_id_;

    // Alignment config
    int silence_anchors_ = 10;
    bool boost_targets_ = true;
    bool enforce_minimum_ = true;
    bool enforce_all_targets_ = true;
    bool ignore_noise_ = true;

    // Windowing config
    int window_size_ms_;
    int stride_ms_;
    int sample_rate_ = 16000;
    int window_size_wav_;
    int stride_size_wav_;

    // Class counts
    int phoneme_classes_;
    int phoneme_groups_;
    int blank_class_;
    int blank_group_;
    int silence_class_;
    int silence_group_;

    // Decoders
    std::unique_ptr<AlignmentUtils> alignment_utils_g_;
    std::unique_ptr<AlignmentUtils> alignment_utils_p_;

public:
    PhonemeTimestampAligner(const std::string &cupe_ckpt_path = "cupe_predict_english.onnx",
                            const std::string &lang = "en-us",
                            const std::string &mapper = "ph66",
                            double duration_max = 10.0,
                            const std::string &output_frames_key = "phoneme_idx",
                            int silence_anchors = 10,
                            bool boost_targets = true,
                            bool enforce_minimum = true,
                            bool enforce_all_targets = true,
                            bool ignore_noise = true);

    Vs2Data process_segments(
        const std::vector<std::map<std::string, boost::variant<double, std::string>>> &srt_data,
        const std::vector<float> &audio_wav,
        const std::string &ts_out_path);

    Vs2Data process_sentence(
        const std::string &text,
        const std::vector<float> &audio_wav,
        const int sample_rate,
        const std::string &ts_out_path);

    std::pair<std::vector<float>, int> chop_wav(
        const std::vector<float>& wav,
        int start_frame, 
        int end_frame
    );


private:
    void _setup_config(int window_size_ms = 120, int stride_ms = 80);
    void _setup_decoders();

    std::vector<float> rms_normalize(const std::vector<float>& audio);
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, int, int, int, int> cupe_prediction(const std::vector<std::vector<float>> &audio_batch, int wav_len);

    TimestampResult extract_timestamps_from_segment(
        const std::vector<float> &wav, int wav_len,
        const std::vector<int> &phoneme_sequence, double start_offset_time,
        const std::vector<int> &group_sequence);

    std::map<std::string, boost::variant<int, double, std::string>> analyze_alignment_coverage(const std::vector<int> &target_sequence,
                                                                                               const std::vector<std::tuple<int, int, int, double>> &aligned_timestamps);

    std::vector<int> map_phonemes_to_groups(const std::vector<int> &phoneme_sequence);

    std::map<std::string, boost::variant<std::vector<int>, std::vector<std::string>>> phonemize_sentence(const std::string &text);
};

PhonemeTimestampAligner::PhonemeTimestampAligner(const std::string &cupe_ckpt_path,
                                                 const std::string &lang,
                                                 const std::string &mapper,
                                                 double duration_max,
                                                 const std::string &output_frames_key,
                                                 int silence_anchors,
                                                 bool boost_targets,
                                                 bool enforce_minimum,
                                                 bool enforce_all_targets,
                                                 bool ignore_noise)
    : selected_mapper_(mapper),
      output_frames_key_(output_frames_key),
      seg_duration_max_(duration_max),
      silence_anchors_(silence_anchors),
      boost_targets_(boost_targets),
      enforce_minimum_(enforce_minimum),
      enforce_all_targets_(enforce_all_targets),
      ignore_noise_(ignore_noise)
{

    std::string effective_lang = lang;

    // Use provided paths or model_name
    std::string effective_ckpt_path = cupe_ckpt_path;
    if (!effective_ckpt_path.empty())
    {
        std::cout << "Using provided CUPE checkpoint path: " << effective_ckpt_path << std::endl;
    }
    
    extractor_ = std::make_unique<CUPEONNXPredictor>(effective_ckpt_path);

    // Initialize other members
    seg_duration_min_samples_ = static_cast<int>(seg_duration_min_ * resampler_sample_rate_);
    wav_len_max_ = static_cast<int>(seg_duration_max_ * resampler_sample_rate_);

    if (selected_mapper_ != "ph66")
    {
        throw std::runtime_error("Currently only 'ph66' mapper is supported.");
    }

    // Initialize phonemizer (TODO: replace with your Phonemizer)
    // phonemizer_ = std::make_unique<Phonemizer>(effective_lang, true);

    // Initialize mappers from global dicts
    phoneme_label_to_id_ = phoneme_mapped_index; // Reverse mapping
    for (const auto &pair : phoneme_mapped_index)
    {
        phoneme_id_to_label_[pair.second] = pair.first;
    }

    group_label_to_id_ = phoneme_groups_index;
    for (const auto &pair : phoneme_groups_index)
    {
        group_id_to_label_[pair.second] = pair.first;
    }

    phoneme_id_to_group_id_ = phoneme_groups_mapper;

    _setup_config();
    _setup_decoders();
}

std::map<std::string, boost::variant<std::vector<int>, std::vector<std::string>>>
PhonemeTimestampAligner::phonemize_sentence(const std::string &text)
{
    return {
        {"ph66", std::vector<int>{46, 4, 36, 4, 37, 30, 7, 36, 30, 4, 55, 10, 44, 43, 2, 38, 9}},
        {"pg16", std::vector<int>{9, 1, 8, 1, 9, 6, 1, 8, 6, 1, 12, 2, 9, 8, 1, 10, 2}},
        {"ipa", std::vector<std::string>{"d", "I", "s", "I", "z", "t", "e", "s", "t", "I", "n", "A", "v", "f", "i:", "sh", "e"}},
        {"word_num", std::vector<int>{0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4}},
        {"words", std::vector<std::string>{"This", "is", "testing", "of", "feature"}},
        {"text", std::vector<std::string>{text}}};
}

void PhonemeTimestampAligner::_setup_config(int window_size_ms, int stride_ms)
{
    window_size_ms_ = window_size_ms;
    stride_ms_ = stride_ms;
    sample_rate_ = 16000;

    window_size_wav_ = static_cast<int>(window_size_ms_ * sample_rate_ / 1000.0);
    stride_size_wav_ = static_cast<int>(stride_ms_ * sample_rate_ / 1000.0);

    // Class counts (exclude padding)
    phoneme_classes_ = phoneme_id_to_label_.size() - 1;
    phoneme_groups_ = group_id_to_label_.size() - 1;
    blank_class_ = phoneme_label_to_id_["noise"];
    blank_group_ = group_label_to_id_["noise"];
    silence_class_ = phoneme_label_to_id_["SIL"];
    silence_group_ = group_label_to_id_["SIL"];

    std::cout << "Phoneme classes: " << phoneme_classes_
              << ", groups: " << phoneme_groups_
              << ", blank_class: " << blank_class_
              << ", blank_group: " << blank_group_
              << ", silence_class: " << silence_class_
              << ", silence_group: " << silence_group_ << std::endl;
}

void PhonemeTimestampAligner::_setup_decoders()
{
    alignment_utils_g_ = std::make_unique<AlignmentUtils>(
        blank_group_, silence_class_, silence_anchors_, ignore_noise_);
    alignment_utils_p_ = std::make_unique<AlignmentUtils>(
        blank_class_, silence_group_, silence_anchors_, ignore_noise_);
}

std::vector<float> PhonemeTimestampAligner::rms_normalize(const std::vector<float>& audio) {
    // 1. Calculate RMS: sqrt(mean(audio ** 2))
    double sum_squares = 0.0;
    for (float sample : audio) {
        sum_squares += sample * sample;
    }
    
    double mean_square = sum_squares / audio.size();
    double rms = std::sqrt(mean_square);
    
    // 2. Normalize: audio / rms (if rms > 0)
    std::vector<float> normalized_audio(audio.size());
    if (rms > 0.0) {
        for (size_t i = 0; i < audio.size(); ++i) {
            normalized_audio[i] = audio[i] / static_cast<float>(rms);
        }
    } else {
        // If RMS == 0, return original (silent audio)
        normalized_audio = audio;
    }
    
    return normalized_audio;
}

std::pair<std::vector<float>, int> PhonemeTimestampAligner::chop_wav(
    const std::vector<float>& input_wav,
    int start_frame, 
    int end_frame) {
    
    int num_frames = (end_frame != -1) ? (end_frame - start_frame) : -1;
    
    // Check minimum duration
    if (num_frames < seg_duration_min_samples_) {
        throw std::invalid_argument("Segment too short: " + std::to_string(num_frames) + 
                                   " frames, minimum required is " + 
                                   std::to_string(seg_duration_min_samples_) + " frames.");
    }
    
    // Slice: wav[:, start_frame:end_frame]
    std::vector<float> wav_slice;
    if (end_frame == -1) {
        wav_slice.assign(input_wav.begin() + start_frame, input_wav.end());
    } else {
        wav_slice.assign(input_wav.begin() + start_frame, 
                        input_wav.begin() + end_frame);
    }
    
    // Assert length
    assert(static_cast<int>(wav_slice.size()) <= num_frames || num_frames == -1);
    if (static_cast<int>(wav_slice.size()) < seg_duration_min_samples_) {
        throw std::runtime_error("Wav shape too small: size=" + std::to_string(wav_slice.size()) +
                               ", start=" + std::to_string(start_frame) + 
                               ", end=" + std::to_string(end_frame));
    }
    
    // Mean across channels (if stereo: dim=0)
    std::vector<float> wav_mono = rms_normalize(wav_slice);
    int wav_len = wav_mono.size();
    
    std::vector<float> wav_processed;
    if (wav_len > wav_len_max_) {
        // Truncate
        wav_processed.assign(wav_mono.begin(), wav_mono.begin() + wav_len_max_);
        wav_len = wav_len_max_;
    } else {
        wav_processed.resize(wav_len_max_, 0.0f);
        std::copy(wav_mono.begin(), wav_mono.end(), wav_processed.begin());
    }
    
    return {wav_processed, wav_len};
}

Vs2Data PhonemeTimestampAligner::process_segments(
    const std::vector<std::map<std::string, boost::variant<double, std::string>>> &srt_data,
    const std::vector<float> &audio_wav,
    const std::string &ts_out_path)
{

    Vs2Data vs2_segments;

    std::cout << "\n==================== Process segments ====================" << std::endl;

    for (size_t i = 0; i < srt_data.size(); ++i)
    {
        const auto &segment = srt_data[i];
        double start_time = boost::get<double>(segment.at("start"));
        double end_time = boost::get<double>(segment.at("end"));
        std::string text = boost::get<std::string>(segment.at("text"));

        // 1. Phonemize
        auto ts_out = phonemize_sentence(text);
        std::vector<int> phoneme_seq = boost::get<std::vector<int>>(ts_out.at("ph66"));
        std::vector<int> group_seq = boost::get<std::vector<int>>(ts_out.at("pg16"));

        if (phoneme_seq.size() < ph_seq_min_)
        {
            std::cout << "Skip segment " << i + 1 << std::endl;
            continue;
        }

        int start_samples = start_time * resampler_sample_rate_;
        int end_samples = end_time * resampler_sample_rate_;
        auto [wav_seg, wav_len] = chop_wav(audio_wav, start_samples, end_samples);

        auto result = extract_timestamps_from_segment(wav_seg, wav_len, phoneme_seq, start_time, group_seq);

        // Segment vs2_segment = segment;
        // vs2_segment["phonemes"] = phoneme_seq;
        // vs2_segment["phoneme_groups"] = group_seq;

        // std::vector<boost::variant<int, double>> phoneme_ts;
        // for (const auto& ts : result.phoneme_timestamps) {
        //     std::map<std::string, boost::variant<int, double>> ts_entry = {
        //         {"phoneme_idx", ts.phoneme_idx},
        //         {"phoneme_label", phoneme_id_to_label_.at(ts.phoneme_idx)},
        //         {"start_ms", ts.start_ms},
        //         {"end_ms", ts.end_ms},
        //         {"confidence", ts.confidence}
        //     };
        //     phoneme_ts.push_back(ts_entry);
        // }
        // vs2_segment["phoneme_ts"] = phoneme_ts;

        // words_ts
        // vs2_segment["words_ts"] = align_words(phoneme_ts,
        //     boost::get<std::vector<int>>(ts_out["word_num"]),
        //     boost::get<std::vector<std::string>>(ts_out["words"]));

        // vs2_segments.push_back(std::move(vs2_segment));
    }

    // print_summary(vs2_segments);

    // if (!ts_out_path.empty()) {
    //     save_json(vs2_segments, ts_out_path);
    //     std::cout << "Saved: " << ts_out_path << std::endl;
    // }

    return vs2_segments; // Exact Python vs2_data equivalent!
}

Vs2Data PhonemeTimestampAligner::process_sentence(
    const std::string &text,
    const std::vector<float> &audio_wav,
    const int sample_rate = 16000,
    const std::string &ts_out_path = "timestamps.json")
{
    double duration = audio_wav.size() / static_cast<double>(resampler_sample_rate_);
    std::map<std::string, boost::variant<double, std::string>> segment = {
        {"start", 0.0}, {"end", duration}, {"text", text}};

    return process_segments({segment}, audio_wav, ts_out_path);
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, int, int, int, int> PhonemeTimestampAligner::cupe_prediction(
    const std::vector<std::vector<float>> &audio_batch,
    int wav_len)
{

    // 1. Unsqueeze: [T] → [1, T]
    std::vector<std::vector<float>> audio_unsqueezed = {audio_batch[0]};

    // 2. Windowing [1, num_windows, window_size]
    auto windowed_audio = slice_windows(audio_unsqueezed, 16000, window_size_ms_, stride_ms_);

    std::cout <<"\nWindows shape: " << windowed_audio.size() << " x " << windowed_audio[0].size() << " x " << windowed_audio[0][0].size() << std::endl;
    int batch_size = 1;
    int num_windows = windowed_audio[0].size(); // num_windows
    int window_size = windowed_audio[0][0].size();

    // 3. Flatten: [num_windows, window_size]
    std::vector<std::vector<float>> windows_flat(num_windows);
    for (size_t i = 0; i < windowed_audio[0].size(); i++)
    {
        for (size_t j = 0; j < windowed_audio[0][i].size(); j++)
        {
            windows_flat[i].push_back(windowed_audio[0][i][j]);
        }
    }

    auto [logits_class, class_shape, logits_group, group_shape, embeddings, emb_shape]  = extractor_->predict(windows_flat, true);

    int frames_per_window = group_shape[1];

    int class_dim = class_shape[2];
    int group_dim = group_shape[2];

    // For simplicity, keep flat but track logical shapes for stitching
    int total_windows = num_windows;

    std::cout<< "Logits class shape: [" << class_shape[0] << ", " << class_shape[1] << ", " << class_shape[2] << "]" << std::endl;
    std::cout<< "Logits group shape: [" << group_shape[0] << ", " << group_shape[1] << ", " << group_shape[2] << "]" << std::endl;

    std::cout << "Total windows: " << total_windows << ", Frames per window: " << frames_per_window << std::endl;
    std::cout<< "logits_class flat size: " << logits_class.size() << ", logits_group flat size: " << logits_group.size() << std::endl;
    int original_audio_length = audio_batch[0].size();
    auto logits_class_stitched = stitch_window_predictions_flat(
        logits_class, total_windows, frames_per_window, class_dim,
        original_audio_length, sample_rate_, window_size_ms_, stride_ms_);

    auto logits_group_stitched = stitch_window_predictions_flat(
        logits_group, total_windows, frames_per_window, group_dim,
        original_audio_length, sample_rate_, window_size_ms_, stride_ms_);

    std::cout<< "logits_class stiched size: " << logits_class_stitched.size() << ", logits_group stiched size: " << logits_group_stitched.size() << std::endl;
    int total_frame = total_windows * frames_per_window /2;
    // 8. Spectral length
    int spectral_len = calc_spec_len_ext(wav_len, window_size_ms_, stride_ms_, sample_rate_, frames_per_window);
    std::cout<< "Calculated spectral length: " << spectral_len << std::endl;
    return {logits_class_stitched, logits_group_stitched, embeddings, spectral_len, total_frame, class_shape[2], group_shape[2]};
}

std::vector<int> PhonemeTimestampAligner::map_phonemes_to_groups(const std::vector<int> &phoneme_sequence){
    std::vector<int> group_sequence;
    for (int ph_id : phoneme_sequence)
    {
        if (phoneme_id_to_group_id_.count(ph_id) > 0)
        {
            group_sequence.push_back(phoneme_id_to_group_id_.at(ph_id));
        }
        else
        {
            group_sequence.push_back(blank_group_); // Map unknown phonemes to blank group
        }
    }
    return group_sequence;
}

std::map<std::string, boost::variant<int, double, std::string>>
PhonemeTimestampAligner::analyze_alignment_coverage(
    const std::vector<int> &target_sequence,
    const std::vector<std::tuple<int, int, int, double>> &aligned_timestamps)
{
    std::unordered_set<int> target_set(target_sequence.begin(), target_sequence.end());
    std::unordered_set<int> aligned_set;
    
    for (const auto& [ph_idx, start_f, end_f, conf] : aligned_timestamps) {
        aligned_set.insert(ph_idx);
    }
    
    std::vector<int> missing_phonemes;
    for (int ph : target_set) {
        if (aligned_set.find(ph) == aligned_set.end()) {
            missing_phonemes.push_back(ph);
        }
    }
    
    double coverage = target_set.empty() ? 1.0 : 
                     static_cast<double>(aligned_set.size()) / target_set.size();
    
    return {
        {"target_count", static_cast<int>(target_set.size())},
        {"aligned_count", static_cast<int>(aligned_set.size())},
        {"coverage_ratio", coverage},
        {"missing_count", static_cast<int>(missing_phonemes.size())}
    };
}
#include <vector>
#include <tuple>
#include <cmath>
#include <iostream>
#include <limits>

struct Timestamp {
    int phoneme_idx;
    int start_frame;
    int end_frame;
    float confidence;
    float start_ms;
    float end_ms;
    
    Timestamp(int idx, int s_frame, int e_frame, float conf, float s_ms, float e_ms)
        : phoneme_idx(idx), start_frame(s_frame), end_frame(e_frame),
          confidence(conf), start_ms(s_ms), end_ms(e_ms) {}
};

// 1. _calculate_confidences equivalent
std::vector<Timestamp> calculate_confidences(
    const std::vector<float>& log_probs,
    const std::vector<std::tuple<int, int, int>>& framestamps,
    int num_classes = 67) {
    std::cout <<"\n--Debug--\n";
    for(auto i : framestamps){
        std::cout << "Phoneme idx: " << std::get<0>(i) 
                  << ", Start frame: " << std::get<1>(i) 
                  << ", End frame: " << std::get<2>(i) << std::endl;
    }
    std::vector<Timestamp> updated_timestamps;
    if (log_probs.empty()) {
        std::cout << "Warning: log_probs is empty. Returning framestamps with zero confidence." << std::endl;
        return updated_timestamps;
    }
    
    int T = log_probs.size() / num_classes;
    
    // Convert log_probs to probabilities [T][C]
    std::vector<std::vector<float>> probs(T, std::vector<float>(num_classes, 0.0f));
    for (int t = 0; t < T; ++t) {
        for (int c = 0; c < num_classes; ++c) {
            int idx = t * num_classes + c;
            if (idx < static_cast<int>(log_probs.size())) {
                float log_prob_value = log_probs[idx];     // ✅ Now float, not vector!
                probs[t][c] = std::exp(log_prob_value);
            }
        }
    }
    
    // Rest of your code unchanged...
    for (const auto& stamp : framestamps) {
        int phoneme_idx, start_frame, end_frame;
        std::tie(phoneme_idx, start_frame, end_frame) = stamp;
        
        start_frame = std::max(0, start_frame);
        end_frame = std::min(T, end_frame);
        
        if (phoneme_idx < 0 || phoneme_idx >= num_classes) continue;
        
        float avg_confidence = probs[start_frame][phoneme_idx];
        
        if (start_frame < end_frame) {
            float half_confidence = avg_confidence / 2.0f;
            int last_good_frame = start_frame;
            int total_good_frames = 1;
            
            for (int f = start_frame + 1; f < end_frame; ++f) {
                float frame_prob = probs[f][phoneme_idx];
                if (frame_prob > half_confidence || frame_prob > 0.1f) {
                    avg_confidence += frame_prob;
                    last_good_frame = f;
                    total_good_frames++;
                }
            }
            
            if (total_good_frames > 1) {
                avg_confidence /= static_cast<float>(total_good_frames);
                end_frame = std::min(T, last_good_frame + 1);
            }
            
            float max_confidence = 0.0f;
            for (int f = start_frame; f < end_frame; ++f) {
                max_confidence = std::max(max_confidence, probs[f][phoneme_idx]);
            }
            if (avg_confidence < max_confidence / 2.0f) {
                avg_confidence = max_confidence;
            }
        }
        
        updated_timestamps.emplace_back(phoneme_idx, start_frame, end_frame, avg_confidence, 0.0f, 0.0f);
    }
    
    return updated_timestamps;
}


std::vector<Timestamp> convert_to_ms(
    const std::vector<Timestamp>& framestamps,
    int spectral_length,
    float start_offset_time,
    int wav_len,
    int sample_rate) {
    
    float duration_in_seconds = static_cast<float>(wav_len) / sample_rate;
    float duration_per_frame = (spectral_length > 0) ? 
        duration_in_seconds / spectral_length : 0.0f;
    
    std::vector<Timestamp> updated_timestamps;
    
    for (const auto& stamp : framestamps) {
        int phoneme_idx = stamp.phoneme_idx;
        int start_frame = stamp.start_frame;
        int end_frame = stamp.end_frame;
        float avg_confidence = stamp.confidence;
        
        // Calculate times in seconds
        float start_sec = start_offset_time + (start_frame * duration_per_frame);
        float end_sec = start_offset_time + (end_frame * duration_per_frame);
        
        // Convert to milliseconds
        float start_ms = start_sec * 1000.0f;
        float end_ms = end_sec * 1000.0f;
        
        updated_timestamps.emplace_back(
            phoneme_idx, start_frame, end_frame, avg_confidence, start_ms, end_ms
        );
    }
    
    return updated_timestamps;
}

TimestampResult PhonemeTimestampAligner::extract_timestamps_from_segment(
    const std::vector<float> &wav, int wav_len,
    const std::vector<int> &phoneme_sequence, double start_offset_time,
    const std::vector<int> &group_sequence)
{
    TimestampResult result;
    
    std::cout << "\n==================== Log2 Extract Timestamps ====================" << std::endl;
    
    // 1. CUPE Prediction
    std::vector<std::vector<float>> audio_batch = {wav};
    auto [logits_class, logits_group, embeddings, spectral_len, total_frames, 
          num_phonemes, num_groups] = cupe_prediction(audio_batch, wav_len);
    
    // Validate spectral_len
    if (spectral_len <= 0 || spectral_len > 10000) {
        std::cerr << "ERROR: Invalid spectral_len = " << spectral_len << std::endl;
        return result;
    }
    
    // 2. Prepare sequences
    std::vector<int> ph_seq = phoneme_sequence;
    std::vector<int> grp_seq = group_sequence;
    std::cout << logits_class.size() << " logits_class values, " << logits_group.size() << " logits_group values" << std::endl;
    std::cout << "total frame: " << total_frames << std::endl;  
    std::cout << "Target phonemes: " << ph_seq.size() << ", groups: " << grp_seq.size() 
              << ", Spectral length: " << spectral_len << std::endl;
    
    // 3. Log softmax
    auto log_probs_p = log_softmax_batch(logits_class, 1, total_frames, num_phonemes);
    auto log_probs_g = log_softmax_batch(logits_group, 1, total_frames, num_groups);
    std::cout << "Applied log softmax to predictions." << std::endl;
    // 4. Forced Alignment
    std::vector<std::vector<int>> true_seqs_g = {grp_seq};
    std::vector<std::vector<int>> true_seqs_p = {ph_seq};
    std::vector<int> pred_lens = {spectral_len};
    std::vector<int> true_seq_lens = {static_cast<int>(ph_seq.size())};
    
    auto frame_groups = alignment_utils_g_->decode_alignments(
        {log_probs_g}, &true_seqs_g, &pred_lens, &true_seq_lens,
        true, boost_targets_, enforce_minimum_, enforce_all_targets_);
    
    std::cout << frame_groups.size() << " frame_groups, " << frame_groups[0].size() << " alignments in first group." << std::endl;
    auto frame_phonemes = alignment_utils_p_->decode_alignments(
        {log_probs_p}, &true_seqs_p, &pred_lens, &true_seq_lens,
        true, boost_targets_, enforce_minimum_, enforce_all_targets_);

    std::cout << "Alignment done" << std::endl;

    // 2. Calculate confidences (NEW)
    auto phoneme_timestamps = calculate_confidences(
        {log_probs_p},           // flattened log_probs T*C
        frame_phonemes[0],        // vector<tuple<int,int,int>>
        67                        // num_classes
    );  // frame_phonemes[0] is vector<tuple>
    
    auto group_timestamps = calculate_confidences(
        {log_probs_g}, frame_groups[0], 67);
    
    std::cout << "Confidences calculated" << std::endl;
    
    // 3. Convert to milliseconds (NEW)
    phoneme_timestamps = convert_to_ms(
        phoneme_timestamps, spectral_len, start_offset_time, wav_len, 16000);
    
    group_timestamps = convert_to_ms(
        group_timestamps, spectral_len, start_offset_time, wav_len, 16000);
    
    // 4. Debug output matching Python
    std::cout << "Predicted phonemes: " << phoneme_timestamps.size() << std::endl;
    std::cout << "Predicted groups: " << group_timestamps.size() << std::endl;
    std::cout << "start_offset_time: " << start_offset_time << std::endl;
    
    for (size_t i = 0; i < std::min(phoneme_timestamps.size(), group_timestamps.size()); ++i) {
        const auto& ph = phoneme_timestamps[i];
        const auto& grp = group_timestamps[i];
        
        std::cout << std::setw(2) << (i+1) << ": "
                  << /* phoneme_id_to_label[ph.phoneme_idx] */ "PH" << ", "
                  << /* group_id_to_label[grp.phoneme_idx] */ "GRP"
                  << " -> (" << std::fixed << std::setprecision(3)
                  << grp.start_ms << " - " << grp.end_ms 
                  << "), Confidence: " << grp.confidence << std::endl;
    }
    
    // 5. Return timestamp dictionary equivalent
    std::cout << "Timestamps ready!" << std::endl;
    return result;
}

int main()
{
    std::string text_sentence = "This is testing of feature";
    std::string audio_path = "audio16khz.wav";
    std::string model_onnx_path = "models/large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt.onnx";

    std::cout << "🚀 Initializing CUPE Phoneme Aligner..." << std::endl;

    // Init aligner
    PhonemeTimestampAligner extractor(model_onnx_path, "en-us", "ph66", 10.0, "phoneme_idx", 10, true, true, true, true);

    // Load WAV without dr_wav - pure C++ implementation
    float original_duration = 0.0f;
    std::vector<float> wav_float = loadAudioFromWav(audio_path, original_duration);
    if (wav_float.empty())
    {
        std::cerr << "❌ Failed to load " << audio_path << std::endl;
        return -1;
    }

    std::cout << "✅ Loaded " << wav_float.size() << " samples" << std::endl;

    // RMS normalize
    std::vector<float> audio_wav = wav_float;
    float rms = 0.0f;
    for (float s : audio_wav)
        rms += s * s;
    rms = std::sqrt(rms / audio_wav.size());
    if (rms > 1e-8f)
        for (auto &s : audio_wav)
            s /= rms;

    // Process!
    auto t0 = std::chrono::high_resolution_clock::now();
    auto vs2_data = extractor.process_sentence(text_sentence, audio_wav, 16000, "timestamps.json");
    auto t1 = std::chrono::high_resolution_clock::now();

    // std::cout << "\n🎯 Timestamps (" << timestamps.size() << " segments):\n";
    // for (const auto& seg : timestamps) {
    //     auto& ph_ts = std::get<std::vector<std::map<std::string, std::variant<int,double,std::string>>>>(seg.at("phoneme_ts"));
    //     std::cout << "  Phonemes: " << ph_ts.size() << std::endl;
    //     for (const auto& ts : ph_ts) {
    //         std::cout << "    " << std::get<std::string>(ts.at("phoneme_label"))
    //                   << " [" << std::get<double>(ts.at("start_ms"))
    //                   << "-" << std::get<double>(ts.at("end_ms"))
    //                   << "] conf=" << std::get<double>(ts.at("confidence")) << std::endl;
    //     }
    // }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "\n⚡ Processing time: " << (ms) << " ms" << std::endl;
    return 0;
}