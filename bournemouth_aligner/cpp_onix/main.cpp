/*
 * 2026-02-16 - Work in progress: C++ implementation of BFA pipeline using ONNX Runtime.
 * Self-contained BFA pipeline for C++ — matches bfaonnx.py simplified pipeline.
 * Phonemization is handled externally (espeak-ng or other).
 * Only phoneme-level alignment (no group timestamps, no boosting, no silence anchoring).
 * Straight CUPE -> log_softmax -> Viterbi -> timestamps.

 cd bournemouth_aligner/cpp_onix

# compile it with:
 cd bournemouth_aligner/cpp_onix
 g++ -std=c++17 main.cpp   -Ionnxruntime-linux-x64-gpu-1.24.1/include   -Lonnxruntime-linux-x64-gpu-1.24.1/lib   -lonnxruntime -lpthread   -o compiled_binary_x64_linux_gpu

 g++ -std=c++17 main.cpp  -Ionnxruntime-linux-x64-1.24.1/include   -Lonnxruntime-linux-x64-1.24.1/lib   -lonnxruntime -lpthread  -o compiled_binary_x64_linux
# run it :
 ./compiled_binary_x64_linux

 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <limits>
#include <numeric>
#include <functional>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#include <onnxruntime_cxx_api.h>

//static Ort::Env global_env(ORT_LOGGING_LEVEL_WARNING, "global");

std::vector<float> resampleAudio(const std::vector<float> &audio_data,
                                 int source_sr, int target_sr)
{
    if (source_sr == target_sr)
        return audio_data;

    std::cout << " Resampling audio: " << source_sr << " Hz -> " << target_sr << " Hz" << std::endl;

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

    std::cout << " Resampled: " << audio_data.size() << " -> " << new_length << " samples" << std::endl;
    return resampled;
}



#pragma pack(push, 1)
struct FmtChunk {
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};
#pragma pack(pop)

std::vector<float> loadAudioFromWav(const std::string &wav_path, float &original_duration)
{
    std::ifstream file(wav_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << wav_path << std::endl;
        original_duration = 0.0f;
        return {};
    }

    // --- Read RIFF header ---
    char riff_id[4];
    uint32_t riff_size;
    char wave_id[4];

    file.read(riff_id, 4);
    file.read(reinterpret_cast<char*>(&riff_size), 4);
    file.read(wave_id, 4);

    if (std::strncmp(riff_id, "RIFF", 4) != 0 ||
        std::strncmp(wave_id, "WAVE", 4) != 0) {
        std::cerr << "Not a valid WAV file\n";
        return {};
    }

    FmtChunk fmt{};
    uint32_t data_size = 0;
    uint32_t data_offset = 0;

    // --- Scan chunks until we find fmt and data ---
    while (file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;

        file.read(chunk_id, 4);
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (!file.good()) break;

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&fmt), sizeof(FmtChunk));

            // Skip any extra fmt bytes
            if (chunk_size > sizeof(FmtChunk))
                file.seekg(chunk_size - sizeof(FmtChunk), std::ios::cur);
        }
        else if (std::strncmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            data_offset = static_cast<uint32_t>(file.tellg());
            file.seekg(chunk_size, std::ios::cur);
        }
        else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }

        if (data_size > 0 && fmt.sample_rate > 0)
            break;
    }

    if (data_size == 0) {
        std::cerr << "No data chunk found\n";
        return {};
    }

    // --- Load audio samples ---
    file.seekg(data_offset, std::ios::beg);

    size_t num_samples = data_size / (fmt.bits_per_sample / 8);
    std::vector<float> audio_data(num_samples / fmt.num_channels);

    if (fmt.bits_per_sample == 16) {
        std::vector<int16_t> samples(num_samples);
        file.read(reinterpret_cast<char*>(samples.data()), data_size);

        for (size_t i = 0; i < audio_data.size(); i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < fmt.num_channels; ch++)
                sum += samples[i * fmt.num_channels + ch] / 32768.0f;
            audio_data[i] = sum / fmt.num_channels;
        }
    }
    else if (fmt.bits_per_sample == 32) {
        std::vector<float> samples(num_samples);
        file.read(reinterpret_cast<char*>(samples.data()), data_size);

        for (size_t i = 0; i < audio_data.size(); i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < fmt.num_channels; ch++)
                sum += samples[i * fmt.num_channels + ch];
            audio_data[i] = sum / fmt.num_channels;
        }
    }

    original_duration = float(audio_data.size()) / fmt.sample_rate;

    std::cout << "\n===== AUDIO INFO =====\n";
    std::cout << "File: " << wav_path << "\n";
    std::cout << "Channels: " << fmt.num_channels << "\n";
    std::cout << "Sample Rate: " << fmt.sample_rate << " Hz\n";
    std::cout << "Bits/Sample: " << fmt.bits_per_sample << "\n";
    std::cout << "Original samples: " << audio_data.size() << "\n";
    std::cout << "Original duration: " << original_duration << "s\n";

    return audio_data;
}

// ============================================================
// Phoneme & group index tables (from ph66_mapper.py)
// ============================================================

std::unordered_map<std::string, int> phoneme_mapped_index = {
    {"SIL", 0}, {"i", 1}, {"i:", 2}, {"\xc9\xa8", 3}, {"\xc9\xaa", 4}, {"e", 5}, {"e:", 6}, {"\xc9\x9b", 7}, 
    {"\xc9\x99", 8}, {"\xc9\x9a", 9}, {"\xca\x8c", 10}, {"u", 11}, {"u:", 12}, {"\xca\x8a", 13}, {"\xc9\xaf", 14}, 
    {"o", 15}, {"o:", 16}, {"\xc9\x94", 17}, {"a", 18}, {"a:", 19}, {"\xc3\xa6", 20}, {"y", 21}, {"\xc3\xb8", 22}, 
    {"a\xc9\xaa", 23}, {"e\xc9\xaa", 24}, {"a\xca\x8a", 25}, {"o\xca\x8a", 26}, {"\xc9\x94\xc9\xaa", 27}, 
    {"p", 28}, {"b", 29}, {"t", 30}, {"d", 31}, {"k", 32}, {"g", 33}, {"q", 34}, {"ts", 35}, {"s", 36}, {"z", 37}, 
    {"t\xca\x83", 38}, {"d\xca\x92", 39}, {"\xca\x83", 40}, {"\xca\x92", 41}, {"\xc9\x95", 42}, 
    {"f", 43}, {"v", 44}, {"\xce\xb8", 45}, {"\xc3\xb0", 46}, {"\xc3\xa7", 47}, {"x", 48}, {"\xc9\xa3", 49}, 
    {"h", 50}, {"\xca\x81", 51}, {"m", 52}, {"n", 53}, {"\xc9\xb2", 54}, {"\xc5\x8b", 55}, {"l", 56}, 
    {"\xc9\xad", 57}, {"\xc9\xbe", 58}, {"\xc9\xb9", 59}, {"j", 60}, {"w", 61}, 
    {"t\xca\xb2", 62}, {"n\xca\xb2", 63}, {"r\xca\xb2", 64}, {"\xc9\xad\xca\xb2", 65}, {"noise", 66}
};

std::unordered_map<int, int> phoneme_groups_mapper = {
    {0, 0}, {1, 1}, {2, 1}, {3, 3}, {4, 1}, {5, 1}, {6, 1}, {7, 1}, {8, 2}, {9, 2}, {10, 2}, {11, 3}, {12, 3}, {13, 3}, 
    {14, 3}, {15, 3}, {16, 3}, {17, 3}, {18, 4}, {19, 4}, {20, 4}, {21, 1}, {22, 1}, {23, 5}, {24, 5}, {25, 5}, {26, 5}, 
    {27, 5}, {28, 6}, {29, 7}, {30, 6}, {31, 7}, {32, 6}, {33, 7}, {34, 6}, {35, 10}, {36, 8}, {37, 9}, {38, 10}, {39, 11}, 
    {40, 8}, {41, 9}, {42, 8}, {43, 8}, {44, 9}, {45, 8}, {46, 9}, {47, 8}, {48, 8}, {49, 9}, {50, 8}, {51, 9}, {52, 12}, 
    {53, 12}, {54, 12}, {55, 12}, {56, 13}, {57, 13}, {58, 14}, {59, 14}, {60, 15}, {61, 15}, {62, 6}, {63, 12}, {64, 14}, 
    {65, 13}, {66, 16}
};

std::unordered_map<std::string, int> phoneme_groups_index = {
    {"SIL", 0}, {"front_vowels", 1}, {"central_vowels", 2}, {"back_vowels", 3}, {"low_vowels", 4}, {"diphthongs", 5}, 
    {"voiceless_stops", 6}, {"voiced_stops", 7}, {"voiceless_fricatives", 8}, {"voiced_fricatives", 9}, 
    {"voiceless_affricates", 10}, {"voiced_affricates", 11}, {"nasals", 12}, {"laterals", 13}, {"rhotics", 14}, 
    {"glides", 15}, {"noise", 16}
};

// Reverse lookup: index -> label
std::unordered_map<int, std::string> index_to_plabel;
std::unordered_map<int, std::string> index_to_glabel;

void init_reverse_lookups() {
    for (const auto& [label, idx] : phoneme_mapped_index)
        index_to_plabel[idx] = label;          // copy the string itself

    for (const auto& [label, idx] : phoneme_groups_index)
        index_to_glabel[idx] = label;          // same here
}

// ============================================================
// Window slicing (exact Python logic)
// ============================================================

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
                    windows[b][w][i] = audio_batch[b][idx];
            }
        }
    }
    return windows;
}

// ============================================================
// Window stitching with cosine weighting (matches Python exactly)
// ============================================================

std::vector<float> stitch_window_predictions_flat(
    const std::vector<float>& windowed_logits_flat,
    int num_windows,
    int frames_per_window,
    int num_classes,
    int original_audio_length,
    int sample_rate,
    int window_size_ms,
    int stride_ms,
    int batch_size = 1)
{
    // Compute total_frames exactly like Python
    int window_size_samples = (window_size_ms * sample_rate) / 1000;
    int stride_samples = (stride_ms * sample_rate) / 1000;
    int num_windows_total = ((original_audio_length - window_size_samples) / stride_samples) + 1;
    int total_frames = (num_windows_total * frames_per_window) / 2;
    int stride_frames = frames_per_window / 2;

    // Validate input size
    size_t expected_input_size = batch_size * num_windows * frames_per_window * num_classes;
    assert(windowed_logits_flat.size() == expected_input_size &&
           "Input size mismatch in stitch_window_predictions_flat");

    // Precompute cosine weights: cos(linspace(-pi/2, pi/2, frames_per_window))
    std::vector<float> weights(frames_per_window);
    for (int i = 0; i < frames_per_window; ++i) {
        double ratio = (frames_per_window > 1)
            ? static_cast<double>(i) / (frames_per_window - 1)
            : 0.0;
        double angle = ratio * M_PI - M_PI_2;
        weights[i] = std::cos(angle);
    }

    // Allocate output buffers
    std::vector<float> combined(batch_size * total_frames * num_classes, 0.0f);
    std::vector<float> weight_sum(batch_size * total_frames, 0.0f);

    // Weighted accumulation over windows
    for (int b = 0; b < batch_size; ++b) {
        int batch_offset = b * num_windows * frames_per_window * num_classes;
        for (int w = 0; w < num_windows; ++w) {
            int start_frame = w * stride_frames;
            int window_offset = batch_offset + w * frames_per_window * num_classes;
            int end_frame = std::min(start_frame + frames_per_window, total_frames);
            int frames_to_copy = end_frame - start_frame;

            for (int f = 0; f < frames_to_copy; ++f) {
                float wgt = weights[f];
                int src_base = window_offset + f * num_classes;
                int dst_base = b * total_frames * num_classes + (start_frame + f) * num_classes;
                int weight_idx = b * total_frames + (start_frame + f);

                for (int c = 0; c < num_classes; ++c) {
                    combined[dst_base + c] += windowed_logits_flat[src_base + c] * wgt;
                }
                weight_sum[weight_idx] += wgt;
            }
        }
    }

    // Normalize: combined / (weight_sum + 1e-8)
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < total_frames; ++t) {
            float denom = weight_sum[b * total_frames + t] + 1e-8f;
            int base = b * total_frames * num_classes + t * num_classes;
            for (int c = 0; c < num_classes; ++c) {
                combined[base + c] /= denom;
            }
        }
    }

    return combined;  // [batch_size * total_frames * num_classes] flat
}

// ============================================================
// Spectral length calculation (matches Python calc_spec_len_ext)
// ============================================================

int calc_spec_len_ext(int wav_len, int window_size_ms, int stride_ms,
                      int sample_rate, int frames_per_window,
                      bool disable_windowing = false, int wav_len_max = 16000)
{
    if (!disable_windowing)
    {
        int window_size_wav = (window_size_ms * sample_rate) / 1000;
        int stride_size_wav = (stride_ms * sample_rate) / 1000;
        int total_frames;

        if (wav_len <= window_size_wav)
        {
            double num_windows = static_cast<double>(wav_len) / window_size_wav;
            total_frames = static_cast<int>(std::ceil(frames_per_window * num_windows));
        }
        else
        {
            int num_windows = ((wav_len - window_size_wav) / stride_size_wav) + 1;
            total_frames = (num_windows * frames_per_window) / 2;
        }

        if (total_frames < 2)
        {
            double actual_ms = 1000.0 * wav_len / sample_rate;
            std::cerr << "WARN: spectral_len < 2, wav_len: " << wav_len
                      << ", frames: " << total_frames
                      << ", expected >= " << window_size_ms
                      << "ms, got " << actual_ms << "ms" << std::endl;
        }

        return total_frames;
    }
    else
    {
        double wav_len_per_frame = static_cast<double>(wav_len_max) / frames_per_window;
        int spectral_len = static_cast<int>(std::ceil(static_cast<double>(wav_len) / wav_len_per_frame));
        if (spectral_len > frames_per_window)
        {
            throw std::runtime_error("spectral_len > frames_per_window: " +
                                     std::to_string(spectral_len) + " > " +
                                     std::to_string(frames_per_window));
        }
        return spectral_len;
    }
}

// ============================================================
// Log softmax
// ============================================================

std::vector<float> log_softmax_frame(const std::vector<float>& logits_frame) {
    assert(!logits_frame.empty());
    std::vector<float> result(logits_frame.size());
    float max_val = *std::max_element(logits_frame.begin(), logits_frame.end());
    float sum_exp = 0.0f;
    for (float val : logits_frame)
        sum_exp += std::exp(val - max_val);
    float log_sum_exp = std::log(sum_exp);
    for (size_t i = 0; i < logits_frame.size(); ++i)
        result[i] = logits_frame[i] - max_val - log_sum_exp;
    return result;
}

// Batch log-softmax over class dimension (dim=2 equivalent)
std::vector<float> log_softmax_batch(
    const std::vector<float>& logits_flat,
    int batch_size,
    int num_frames,
    int num_classes)
{
    assert(logits_flat.size() == (size_t)(batch_size * num_frames * num_classes));
    std::vector<float> result(logits_flat.size());

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < num_frames; ++t) {
            int frame_start = (b * num_frames + t) * num_classes;
            std::vector<float> frame_logits(
                logits_flat.begin() + frame_start,
                logits_flat.begin() + frame_start + num_classes);
            auto frame_log_probs = log_softmax_frame(frame_logits);
            std::copy(frame_log_probs.begin(), frame_log_probs.end(),
                      result.begin() + frame_start);
        }
    }
    return result;
}

// ============================================================
// Alignment data structures
// ============================================================

struct FrameStamp {
    int phoneme_id;
    int start_frame;
    int end_frame;
    int target_seq_idx;
    float confidence = 0.0f;
};

// ============================================================
// ViterbiDecoder — matches Python ViterbiDecoder in bfaonnx.py
// ============================================================

class ViterbiDecoder {
public:
    int blank_id_;
    int silence_id_;
    int silence_anchors_;
    float min_phoneme_prob_;
    bool ignore_noise_;
    static constexpr float NEG_INF = -1000.0f;

    ViterbiDecoder(int blank_id, int silence_id, int silence_anchors = 3,
                   float min_phoneme_prob = 1e-8f, bool ignore_noise = true)
        : blank_id_(blank_id), silence_id_(silence_id),
          silence_anchors_(silence_anchors),
          min_phoneme_prob_(min_phoneme_prob), ignore_noise_(ignore_noise) {}

    /**
     * Standard CTC Viterbi decoding.
     *
     * Args:
     *   log_probs:  flat [T * C] log-probability array
     *   T:          number of frames
     *   C:          number of classes
     *   ctc_path:   CTC state sequence [ctc_len] (blank-phoneme-blank-...)
     *   ctc_len:    length of CTC path
     *   ctc_path_true_idx: mapping from CTC states to target phoneme indices [ctc_len]
     *   band_width: Sakoe-Chiba band width (0 = disabled)
     *
     * Returns:
     *   (frame_phonemes[T], frame_phonemes_idx[T])
     */
    std::pair<std::vector<int>, std::vector<int>> viterbi_decode(
        const std::vector<float>& log_probs,
        int T, int C,
        const std::vector<int>& ctc_path,
        int ctc_len,
        const std::vector<int>& ctc_path_true_idx,
        int band_width = 0)
    {
        // DP tables
        std::vector<std::vector<float>> dp(T, std::vector<float>(ctc_len, NEG_INF));
        std::vector<std::vector<int>> backpointers(T, std::vector<int>(ctc_len, 0));

        // Band constraint setup
        bool use_band = (band_width > 0 && T > 1 && ctc_len > 1);
        float pace = use_band ? (float)(ctc_len - 1) / (T - 1) : 0.0f;

        // Initialize first frame
        dp[0][0] = log_probs[0 * C + blank_id_];
        if (ctc_len > 1)
            dp[0][1] = log_probs[0 * C + ctc_path[1]];

        // Precompute skip mask: can skip state s if ctc_path[s] != ctc_path[s-2]
        std::vector<bool> can_skip(ctc_len, false);
        for (int s = 2; s < ctc_len; s++) {
            if (ctc_path[s] != ctc_path[s - 2])
                can_skip[s] = true;
        }

        // Forward pass
        for (int t = 1; t < T; t++) {
            for (int s = 0; s < ctc_len; s++) {
                float emit = log_probs[t * C + ctc_path[s]];

                // Stay transition
                float best_score = dp[t - 1][s] + emit;
                int best_prev = s;

                // Advance transition (s >= 1)
                if (s >= 1) {
                    float advance = dp[t - 1][s - 1] + emit;
                    if (advance > best_score) {
                        best_score = advance;
                        best_prev = s - 1;
                    }
                }

                // Skip transition (s >= 2, different phoneme)
                if (s >= 2 && can_skip[s]) {
                    float skip = dp[t - 1][s - 2] + emit;
                    if (skip > best_score) {
                        best_score = skip;
                        best_prev = s - 2;
                    }
                }

                dp[t][s] = best_score;
                backpointers[t][s] = best_prev;
            }

            // Apply diagonal band constraint after computing all states
            if (use_band) {
                float center = t * pace;
                for (int s = 0; s < ctc_len; s++) {
                    if ((float)s < center - band_width || (float)s > center + band_width)
                        dp[t][s] = NEG_INF;
                }
            }
        }

        // Find best valid final state
        int final_state = 0;
        float best_final = NEG_INF;
        bool found_valid = false;
        for (int s = 0; s < ctc_len; s++) {
            if (dp[T - 1][s] > NEG_INF) {
                if (!found_valid || dp[T - 1][s] > best_final) {
                    best_final = dp[T - 1][s];
                    final_state = s;
                    found_valid = true;
                }
            }
        }
        if (!found_valid) {
            // Fallback: argmax over all states
            for (int s = 0; s < ctc_len; s++) {
                if (dp[T - 1][s] > best_final) {
                    best_final = dp[T - 1][s];
                    final_state = s;
                }
            }
        }

        // Backtrack
        std::vector<int> path_states(T);
        path_states[T - 1] = final_state;
        for (int t = T - 2; t >= 0; t--)
            path_states[t] = backpointers[t + 1][path_states[t + 1]];

        // Map CTC states to phoneme IDs and target indices
        std::vector<int> frame_phonemes(T);
        std::vector<int> frame_phonemes_idx(T);
        for (int t = 0; t < T; t++) {
            frame_phonemes[t] = ctc_path[path_states[t]];
            frame_phonemes_idx[t] = ctc_path_true_idx[path_states[t]];
        }

        return {frame_phonemes, frame_phonemes_idx};
    }

    /**
     * Group consecutive frames into phoneme segments.
     * Splits when phoneme ID or target index changes.
     * Matches Python assort_frames exactly.
     */
    std::vector<FrameStamp> assort_frames(
        const std::vector<int>& frame_phonemes,
        const std::vector<int>& frame_phonemes_idx,
        int max_blanks = 10)
    {
        if (frame_phonemes.empty()) return {};

        // Find transition points
        std::vector<int> transition_indices;
        transition_indices.push_back(0);
        for (size_t i = 1; i < frame_phonemes.size(); i++) {
            if (frame_phonemes[i] != frame_phonemes[i - 1] ||
                frame_phonemes_idx[i] != frame_phonemes_idx[i - 1]) {
                transition_indices.push_back(i);
            }
        }

        std::vector<FrameStamp> framestamps;

        for (size_t i = 0; i < transition_indices.size(); i++) {
            int start_idx = transition_indices[i];
            int end_idx = (i + 1 < transition_indices.size())
                              ? transition_indices[i + 1]
                              : (int)frame_phonemes.size();

            int seg_phoneme = frame_phonemes[start_idx];
            int seg_idx = frame_phonemes_idx[start_idx];

            // If seg_idx is -1 (blank), search forward for a valid target index
            if (seg_idx == -1) {
                for (int j = start_idx; j < end_idx; j++) {
                    if (frame_phonemes_idx[j] != -1) {
                        seg_idx = frame_phonemes_idx[j];
                        break;
                    }
                }
            }

            // Handle blank segments
            if (seg_phoneme == blank_id_) {
                int seg_length = end_idx - start_idx;
                if (ignore_noise_) {
                    if (seg_length > max_blanks) continue;  // skip long blanks
                } else {
                    if (seg_length > max_blanks) {
                        framestamps.push_back({seg_phoneme, start_idx, end_idx, seg_idx});
                    }
                }
            }

            // Non-blank segments always included
            if (seg_phoneme != blank_id_) {
                framestamps.push_back({seg_phoneme, start_idx, end_idx, seg_idx});
            }
        }

        return framestamps;
    }

    // ---- Advanced alignment methods (match bfaonnx.py) ----

    struct ViterbiResult {
        std::vector<int> frame_phonemes;
        std::vector<int> frame_phonemes_idx;
        float alignment_score;
    };

    /**
     * Boost target phoneme probabilities and re-normalize with log_softmax.
     * Matches Python _boost_target_phonemes (bfaonnx.py line 496).
     */
    std::vector<float> boost_target_phonemes(
        const std::vector<float>& log_probs_flat, int T, int C,
        const std::vector<int>& target_phonemes, float boost_factor = 5.0f)
    {
        std::vector<float> boosted = log_probs_flat;

        std::unordered_set<int> unique_targets(target_phonemes.begin(), target_phonemes.end());
        unique_targets.erase(blank_id_);
        unique_targets.erase(-100);

        for (int phoneme_idx : unique_targets) {
            if (phoneme_idx >= 0 && phoneme_idx < C) {
                for (int t = 0; t < T; t++)
                    boosted[t * C + phoneme_idx] += boost_factor;
            }
        }

        for (int t = 0; t < T; t++) {
            std::vector<float> frame(boosted.begin() + t * C, boosted.begin() + (t + 1) * C);
            auto normalized = log_softmax_frame(frame);
            std::copy(normalized.begin(), normalized.end(), boosted.begin() + t * C);
        }

        return boosted;
    }

    /**
     * Ensure target phonemes have minimum probability at each frame.
     * Matches Python _enforce_minimum_probabilities (bfaonnx.py line 525).
     */
    std::vector<float> enforce_minimum_probabilities(
        const std::vector<float>& log_probs_flat, int T, int C,
        const std::vector<int>& target_phonemes)
    {
        std::vector<float> modified = log_probs_flat;
        float min_log_prob = std::log(min_phoneme_prob_);

        std::unordered_set<int> unique_targets(target_phonemes.begin(), target_phonemes.end());
        unique_targets.erase(blank_id_);
        unique_targets.erase(-100);

        for (int phoneme_idx : unique_targets) {
            if (phoneme_idx >= 0 && phoneme_idx < C) {
                for (int t = 0; t < T; t++) {
                    if (modified[t * C + phoneme_idx] < min_log_prob)
                        modified[t * C + phoneme_idx] = min_log_prob;
                }
            }
        }

        return modified;
    }

    /**
     * Detect segments of silence using softmax P(SIL) with sliding window.
     * Matches Python _detect_silence_segments (bfaonnx.py line 552).
     */
    std::vector<std::pair<int,int>> detect_silence_segments(
        const std::vector<float>& log_probs_flat, int T, int C,
        float sil_prob_threshold = 0.8f, int min_silence_frames = -1)
    {
        if (min_silence_frames < 0) min_silence_frames = silence_anchors_;
        int sliding_frames = min_silence_frames;

        if (silence_id_ >= C) return {};
        if (T < sliding_frames) return {};

        // Full softmax P(SIL) across all classes
        std::vector<float> sil_prob(T);
        for (int t = 0; t < T; t++)
            sil_prob[t] = std::exp(log_probs_flat[t * C + silence_id_]);

        // Sliding-window average
        std::vector<float> window_avg;
        if (sliding_frames > 1) {
            std::vector<float> cumsum(T + 1, 0.0f);
            for (int t = 0; t < T; t++)
                cumsum[t + 1] = cumsum[t] + sil_prob[t];
            int n_windows = T - sliding_frames + 1;
            window_avg.resize(n_windows);
            for (int i = 0; i < n_windows; i++)
                window_avg[i] = (cumsum[i + sliding_frames] - cumsum[i]) / sliding_frames;
        } else {
            window_avg = sil_prob;
        }

        // Find contiguous silence regions
        std::vector<std::pair<int,int>> silence_segments;
        bool in_silence = false;
        int silence_start = 0;

        for (int i = 0; i < (int)window_avg.size(); i++) {
            if (window_avg[i] >= sil_prob_threshold && !in_silence) {
                in_silence = true;
                silence_start = i;
            } else if (window_avg[i] < sil_prob_threshold && in_silence) {
                in_silence = false;
                int seg_end = std::min(i + sliding_frames - 1, T);
                if (seg_end - silence_start >= min_silence_frames)
                    silence_segments.push_back({silence_start, seg_end});
            }
        }

        if (in_silence) {
            int seg_end = T;
            if (seg_end - silence_start >= min_silence_frames)
                silence_segments.push_back({silence_start, seg_end});
        }

        return silence_segments;
    }

    /**
     * Find groups of consecutive SIL tokens in the target sequence.
     * Matches Python _find_target_sil_groups (bfaonnx.py line 624).
     */
    std::vector<std::pair<int,int>> find_target_sil_groups(
        const std::vector<int>& true_sequence)
    {
        std::vector<std::pair<int,int>> groups;
        int i = 0, n = (int)true_sequence.size();
        while (i < n) {
            if (true_sequence[i] == silence_id_) {
                int start = i;
                while (i < n && true_sequence[i] == silence_id_) i++;
                groups.push_back({start, i});
            } else {
                i++;
            }
        }
        return groups;
    }

    /**
     * Match target SIL groups to audio silence segments by proportional position.
     * Matches Python _match_silences (bfaonnx.py line 647).
     */
    std::vector<std::pair<int,int>> match_silences(
        const std::vector<std::pair<int,int>>& sil_groups,
        const std::vector<std::pair<int,int>>& audio_silences,
        int target_len, int num_frames, float best_dist_threshold = 0.3f)
    {
        if (sil_groups.empty() || audio_silences.empty() || target_len == 0 || num_frames == 0)
            return {};

        std::vector<std::pair<int,int>> matches;
        int audio_idx = 0;

        for (int tg_idx = 0; tg_idx < (int)sil_groups.size(); tg_idx++) {
            float target_pos = (sil_groups[tg_idx].first + sil_groups[tg_idx].second) / 2.0f / target_len;

            int best_audio = -1;
            float best_dist = std::numeric_limits<float>::infinity();

            for (int ai = audio_idx; ai < (int)audio_silences.size(); ai++) {
                float audio_pos = (audio_silences[ai].first + audio_silences[ai].second) / 2.0f / num_frames;
                float dist = std::abs(target_pos - audio_pos);

                if (dist < best_dist) {
                    best_dist = dist;
                    best_audio = ai;
                } else if (dist > best_dist) {
                    break;
                }
            }

            if (best_audio >= 0 && best_dist < best_dist_threshold) {
                matches.push_back({tg_idx, best_audio});
                audio_idx = best_audio + 1;
            }
        }

        return matches;
    }

    /**
     * Boost blank probability at detected silence frames and re-normalize.
     * Matches Python _anchor_silence_in_log_probs (bfaonnx.py line 689).
     */
    std::vector<float> anchor_silence_in_log_probs(
        const std::vector<float>& log_probs_flat, int T, int C,
        const std::vector<std::pair<int,int>>& silence_segments, float boost = 10.0f)
    {
        std::vector<float> modified = log_probs_flat;
        for (const auto& seg : silence_segments) {
            for (int t = seg.first; t < seg.second && t < T; t++)
                modified[t * C + blank_id_] += boost;
            for (int t = seg.first; t < seg.second && t < T; t++) {
                std::vector<float> frame(modified.begin() + t * C, modified.begin() + (t + 1) * C);
                auto normalized = log_softmax_frame(frame);
                std::copy(normalized.begin(), normalized.end(), modified.begin() + t * C);
            }
        }
        return modified;
    }

    /**
     * Segmented Viterbi: split at matched silence points, run Viterbi per segment.
     * Matches Python _segmented_viterbi_decode (bfaonnx.py line 709).
     * Returns empty vectors if segmentation is not possible (caller falls back).
     */
    std::pair<std::vector<int>, std::vector<int>> segmented_viterbi_decode(
        const std::vector<float>& log_probs_flat, int T, int C,
        const std::vector<int>& true_sequence,
        const std::vector<int>& true_sequence_idx,
        int boundary_pad = 3, int min_speech_frames = 20, bool debug = false)
    {
        int num_frames = T;
        int target_len = (int)true_sequence.size();

        // Step 1: find anchor points in both sequences
        auto sil_groups = find_target_sil_groups(true_sequence);
        if (sil_groups.empty()) {
            if (debug) std::cout << "WARN: No target SIL groups detected." << std::endl;
            return {{}, {}};
        }

        int min_sil_frames = silence_anchors_;
        auto audio_silences = detect_silence_segments(log_probs_flat, T, C, 0.9f, min_sil_frames);

        if (audio_silences.empty() && target_len > 200) {
            float new_thresh = 1.0f - (0.09f * min_sil_frames);
            new_thresh = std::max(0.05f, new_thresh);
            if (debug) std::cout << "WARN: Retrying with threshold " << new_thresh << std::endl;
            audio_silences = detect_silence_segments(log_probs_flat, T, C, new_thresh, min_sil_frames);
        }

        if (audio_silences.empty() && target_len > 200 && min_sil_frames > 3) {
            if (debug) std::cout << "WARN: Retrying with min_silence_frames=3" << std::endl;
            min_sil_frames = 3;
            audio_silences = detect_silence_segments(log_probs_flat, T, C, 0.9f, min_sil_frames);
        }

        if (debug) {
            std::cout << "Target SIL groups: " << sil_groups.size() << std::endl;
            std::cout << "Audio silences: " << audio_silences.size() << std::endl;
        }

        if (audio_silences.empty()) {
            if (debug) std::cout << "WARN: No audio silences detected." << std::endl;
            return {{}, {}};
        }

        // Step 2: match target SIL groups to audio silences
        auto matches = match_silences(sil_groups, audio_silences, target_len, num_frames);
        if (matches.empty()) return {{}, {}};

        // Step 3: build segments (audio_start, audio_end, tgt_start, tgt_end, is_silence)
        struct Segment {
            int audio_start, audio_end, tgt_start, tgt_end;
            bool is_silence;
        };
        std::vector<Segment> segments;
        int prev_audio = 0, prev_target = 0;

        for (const auto& m : matches) {
            int tg_start = sil_groups[m.first].first;
            int tg_end   = sil_groups[m.first].second;
            int as_start = audio_silences[m.second].first;
            int as_end   = audio_silences[m.second].second;

            if (prev_audio < as_start && prev_target < tg_start)
                segments.push_back({prev_audio, as_start, prev_target, tg_start, false});
            else if (prev_audio < as_start)
                segments.push_back({prev_audio, as_start, prev_target, prev_target, false});

            segments.push_back({as_start, as_end, tg_start, tg_end, true});
            prev_audio = as_end;
            prev_target = tg_end;
        }

        if (prev_audio < num_frames && prev_target < target_len)
            segments.push_back({prev_audio, num_frames, prev_target, target_len, false});
        else if (prev_audio < num_frames)
            segments.push_back({prev_audio, num_frames, prev_target, prev_target, false});

        // Step 3b: merge short speech segments
        std::vector<Segment> merged;
        for (const auto& seg : segments) {
            int n_frames_seg = seg.audio_end - seg.audio_start;
            int n_phonemes = seg.tgt_end - seg.tgt_start;

            if (!seg.is_silence && n_phonemes > 0 && n_frames_seg < min_speech_frames && !merged.empty()) {
                auto& prev = merged.back();
                prev.audio_end = seg.audio_end;
                prev.tgt_end = seg.tgt_end;
                prev.is_silence = false;
            } else {
                merged.push_back(seg);
            }
        }
        segments = merged;

        if (debug) std::cout << "Segments after silence anchoring: " << segments.size() << std::endl;

        // Step 4: process each segment
        std::vector<int> all_frame_phonemes;
        std::vector<int> all_frame_phonemes_idx;

        for (const auto& seg : segments) {
            int n_frames_seg = seg.audio_end - seg.audio_start;
            if (n_frames_seg <= 0) continue;

            if (seg.is_silence) {
                int n_sils = seg.tgt_end - seg.tgt_start;
                std::vector<int> sil_phonemes(n_frames_seg, silence_id_);
                std::vector<int> sil_idx(n_frames_seg, -1);

                if (n_sils > 0) {
                    float frames_per_sil = (float)n_frames_seg / n_sils;
                    for (int k = 0; k < n_sils; k++) {
                        int f_start = (int)(k * frames_per_sil);
                        int f_end = (int)((k + 1) * frames_per_sil);
                        for (int f = f_start; f < f_end && f < n_frames_seg; f++)
                            sil_idx[f] = seg.tgt_start + k;
                    }
                }

                all_frame_phonemes.insert(all_frame_phonemes.end(), sil_phonemes.begin(), sil_phonemes.end());
                all_frame_phonemes_idx.insert(all_frame_phonemes_idx.end(), sil_idx.begin(), sil_idx.end());
            } else {
                // Boundary padding
                int padded_start = std::max(0, seg.audio_start - boundary_pad);
                int padded_end = std::min(num_frames, seg.audio_end + boundary_pad);
                int pad_left = seg.audio_start - padded_start;

                std::vector<int> seg_target(true_sequence.begin() + seg.tgt_start,
                                            true_sequence.begin() + seg.tgt_end);
                std::vector<int> seg_target_idx(true_sequence_idx.begin() + seg.tgt_start,
                                                true_sequence_idx.begin() + seg.tgt_end);

                int seg_T = padded_end - padded_start;
                std::vector<float> seg_lp(seg_T * C);
                for (int t = 0; t < seg_T; t++)
                    for (int c = 0; c < C; c++)
                        seg_lp[t * C + c] = log_probs_flat[(padded_start + t) * C + c];

                if (seg_target.empty()) {
                    std::vector<int> blank_ph(n_frames_seg, blank_id_);
                    std::vector<int> blank_idx(n_frames_seg, -1);
                    all_frame_phonemes.insert(all_frame_phonemes.end(), blank_ph.begin(), blank_ph.end());
                    all_frame_phonemes_idx.insert(all_frame_phonemes_idx.end(), blank_idx.begin(), blank_idx.end());
                } else {
                    // Anchor sub-silences
                    auto sub_sil = detect_silence_segments(seg_lp, seg_T, C, 0.8f, min_sil_frames);
                    seg_lp = anchor_silence_in_log_probs(seg_lp, seg_T, C, sub_sil, 5.0f);

                    // Build CTC path
                    int seg_seq_len = (int)seg_target.size();
                    int stride = 4;
                    if ((stride * seg_seq_len + 1) > (int)(seg_T * 0.9)) stride = 3;
                    if ((stride * seg_seq_len + 1) > (int)(seg_T * 0.8)) stride = 2;
                    int expanded_len = stride * seg_seq_len + 1;

                    if (expanded_len > (int)(seg_T * 1.2))
                        return {{}, {}};  // trigger fallback

                    std::vector<int> ctc_path(expanded_len, blank_id_);
                    std::vector<int> ctc_path_idx(expanded_len, -1);
                    for (int j = 0; j < seg_seq_len; j++) {
                        int pos = 1 + j * stride;
                        if (pos < expanded_len) {
                            ctc_path[pos] = seg_target[j];
                            ctc_path_idx[pos] = seg_target_idx[j];
                        }
                    }

                    int band_width = (expanded_len > 60) ? std::max(expanded_len / 3, 30) : 0;
                    auto [seg_frames, seg_frames_idx] = viterbi_decode(
                        seg_lp, seg_T, C, ctc_path, expanded_len, ctc_path_idx, band_width);

                    // Trim padding
                    int trim_end = std::min(pad_left + n_frames_seg, (int)seg_frames.size());
                    all_frame_phonemes.insert(all_frame_phonemes.end(),
                        seg_frames.begin() + pad_left, seg_frames.begin() + trim_end);
                    all_frame_phonemes_idx.insert(all_frame_phonemes_idx.end(),
                        seg_frames_idx.begin() + pad_left, seg_frames_idx.begin() + trim_end);
                }
            }
        }

        // Step 5: concatenate and pad/trim to T frames
        if (all_frame_phonemes.empty()) return {{}, {}};

        if ((int)all_frame_phonemes.size() < num_frames) {
            all_frame_phonemes.resize(num_frames, blank_id_);
            all_frame_phonemes_idx.resize(num_frames, -1);
        } else if ((int)all_frame_phonemes.size() > num_frames) {
            all_frame_phonemes.resize(num_frames);
            all_frame_phonemes_idx.resize(num_frames);
        }

        return {all_frame_phonemes, all_frame_phonemes_idx};
    }

    /**
     * Full forced alignment with boosting, minimum enforcement, and segmented Viterbi.
     * Matches Python decode_with_forced_alignment (bfaonnx.py line 905).
     */
    ViterbiResult decode_with_forced_alignment(
        const std::vector<float>& log_probs_flat, int T, int C,
        const std::vector<int>& true_sequence,
        bool return_scores = false, bool boost_targets = true,
        bool enforce_minimum = true, bool anchor_pauses = true, bool debug = false)
    {
        int seq_len = (int)true_sequence.size();

        std::vector<int> true_sequence_idx(seq_len);
        std::iota(true_sequence_idx.begin(), true_sequence_idx.end(), 0);

        if (seq_len == 0) {
            std::vector<int> frame_ph(T, blank_id_);
            std::vector<int> frame_idx(T, -1);
            float score = 0.0f;
            if (return_scores) {
                for (int t = 0; t < T; t++)
                    score += log_probs_flat[t * C + blank_id_];
            }
            return {frame_ph, frame_idx, score};
        }

        std::vector<float> modified = log_probs_flat;

        if (boost_targets) {
            modified = boost_target_phonemes(modified, T, C, true_sequence);
            if (debug) std::cout << "Applied target phoneme boosting." << std::endl;
        }

        if (enforce_minimum) {
            modified = enforce_minimum_probabilities(modified, T, C, true_sequence);
            if (debug) std::cout << "Applied minimum probability enforcement." << std::endl;
        }

        // Try segmented alignment
        if (anchor_pauses) {
            if (debug) std::cout << "Attempting segmented Viterbi alignment..." << std::endl;
            auto [seg_frames, seg_idx] = segmented_viterbi_decode(
                modified, T, C, true_sequence, true_sequence_idx, 3, 20, debug);

            if (!seg_frames.empty() && (int)seg_frames.size() == T) {
                if (debug) std::cout << "Segmented Viterbi alignment successful." << std::endl;
                float score = 0.0f;
                if (return_scores)
                    score = calculate_alignment_score(log_probs_flat, T, C, seg_frames);
                return {seg_frames, seg_idx, score};
            }
        }

        if (debug) std::cout << "Running standard Viterbi..." << std::endl;

        // Fallback: standard Viterbi
        int stride = 4;
        if (stride * seq_len + 1 > (int)(T * 0.9)) stride = 3;
        if (stride * seq_len + 1 > (int)(T * 0.8)) stride = 2;
        int expanded_len = stride * seq_len + 1;

        if (expanded_len > T) {
            std::cerr << "ERROR: Target sequence too long to align: expanded "
                      << expanded_len << " > frames " << T << std::endl;
            return {{}, {}, 0.0f};
        }

        std::vector<int> ctc_path(expanded_len, blank_id_);
        std::vector<int> ctc_path_idx(expanded_len, -1);
        for (int j = 0; j < seq_len; j++) {
            int pos = 1 + j * stride;
            if (pos < expanded_len) {
                ctc_path[pos] = true_sequence[j];
                ctc_path_idx[pos] = true_sequence_idx[j];
            }
        }

        int band_width = (expanded_len > 60) ? std::max(expanded_len / 4, 20) : 0;
        auto [frame_phonemes, frame_phonemes_idx] = viterbi_decode(
            modified, T, C, ctc_path, expanded_len, ctc_path_idx, band_width);

        float score = 0.0f;
        if (return_scores)
            score = calculate_alignment_score(log_probs_flat, T, C, frame_phonemes);

        return {frame_phonemes, frame_phonemes_idx, score};
    }

    /**
     * Calculate total alignment score.
     * Matches Python _calculate_alignment_score (bfaonnx.py line 992).
     */
    float calculate_alignment_score(
        const std::vector<float>& log_probs_flat, int T, int C,
        const std::vector<int>& frame_phonemes)
    {
        float total_score = 0.0f;
        for (int t = 0; t < T && t < (int)frame_phonemes.size(); t++) {
            int phoneme_idx = frame_phonemes[t];
            if (phoneme_idx >= 0 && phoneme_idx < C)
                total_score += log_probs_flat[t * C + phoneme_idx];
        }
        return total_score;
    }
};

// ============================================================
// AlignmentUtils — matches Python decode_alignments_simple
// ============================================================

class AlignmentUtils {
public:
    int blank_id_;
    int silence_id_;
    int silence_anchors_;
    ViterbiDecoder viterbi_decoder_;

    AlignmentUtils(int blank_id, int silence_id, int silence_anchors = 10, bool ignore_noise = true)
        : blank_id_(blank_id), silence_id_(silence_id), silence_anchors_(silence_anchors),
          viterbi_decoder_(blank_id, silence_id, silence_anchors, 1e-8f, ignore_noise) {}

    /**
     * Simple forced alignment: CTC path construction + Viterbi + assort.
     * No boosting, no silence anchoring. Matches Python decode_alignments_simple.
     *
     * Args:
     *   log_probs_flat: flat [B * T_max * C] log-probability array
     *   batch_size, T_max, C: dimensions
     *   true_seqs: target phoneme sequences per batch item
     *   pred_lens: number of valid frames per batch item
     *   true_seqs_lens: target sequence lengths per batch item
     *
     * Returns:
     *   Per-batch list of FrameStamp segments.
     */
    std::vector<std::vector<FrameStamp>> decode_alignments_simple(
        const std::vector<float>& log_probs_flat,
        int batch_size, int T_max, int C,
        const std::vector<std::vector<int>>& true_seqs,
        const std::vector<int>& pred_lens,
        const std::vector<int>& true_seqs_lens)
    {
        int blank_id = viterbi_decoder_.blank_id_;
        std::vector<std::vector<FrameStamp>> assorted;

        for (int i = 0; i < batch_size; i++) {
            int T = pred_lens[i];
            int S = true_seqs_lens[i];

            // Extract per-sample log probs [T * C]
            std::vector<float> lp(T * C);
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < C; c++) {
                    lp[t * C + c] = log_probs_flat[(i * T_max + t) * C + c];
                }
            }

            // Adaptive stride to fit CTC path within available frames
            int stride = 4;
            if (stride * S + 1 > (int)(T * 0.9)) stride = 3;
            if (stride * S + 1 > (int)(T * 0.8)) stride = 2;
            int ctc_len = stride * S + 1;

            // Build CTC path: blank-phoneme-blank-phoneme-...-blank
            std::vector<int> ctc_path(ctc_len, blank_id);
            std::vector<int> ctc_path_idx(ctc_len, -1);
            for (int j = 0; j < S; j++) {
                int pos = 1 + j * stride;
                if (pos < ctc_len) {
                    ctc_path[pos] = true_seqs[i][j];
                    ctc_path_idx[pos] = j;
                }
            }

            // Band constraint for long sequences
            int band_width = (ctc_len > 60) ? std::max(ctc_len / 4, 20) : 0;

            // Viterbi decode
            auto [frame_phonemes, frame_phonemes_idx] = viterbi_decoder_.viterbi_decode(
                lp, T, C, ctc_path, ctc_len, ctc_path_idx, band_width);

            // Assort into segments
            auto frames = viterbi_decoder_.assort_frames(frame_phonemes, frame_phonemes_idx);
            assorted.push_back(frames);
        }

        return assorted;
    }

    /**
     * Full forced alignment with boosting, silence anchoring, and segmented Viterbi.
     * Matches Python decode_alignments (bfaonnx.py line 1137).
     */
    std::vector<std::vector<FrameStamp>> decode_alignments(
        const std::vector<float>& log_probs_flat,
        int batch_size, int T_max, int C,
        const std::vector<std::vector<int>>& true_seqs,
        const std::vector<int>& pred_lens,
        const std::vector<int>& true_seqs_lens,
        bool forced_alignment = true,
        bool boost_targets = true,
        bool enforce_minimum = true,
        bool debug = false)
    {
        std::vector<std::vector<FrameStamp>> assorted;

        if (forced_alignment) {
            for (int i = 0; i < batch_size; i++) {
                int T = pred_lens[i];
                int S = true_seqs_lens[i];

                // Extract per-sample log probs [T * C]
                std::vector<float> lp(T * C);
                for (int t = 0; t < T; t++)
                    for (int c = 0; c < C; c++)
                        lp[t * C + c] = log_probs_flat[(i * T_max + t) * C + c];

                if (S == 0) {
                    assorted.push_back({});
                    continue;
                }

                std::vector<int> true_seq(true_seqs[i].begin(), true_seqs[i].begin() + S);
                bool anchor_pauses = (silence_anchors_ > 0);

                auto result = viterbi_decoder_.decode_with_forced_alignment(
                    lp, T, C, true_seq, false,
                    boost_targets, enforce_minimum, anchor_pauses, debug);

                auto frames = viterbi_decoder_.assort_frames(
                    result.frame_phonemes, result.frame_phonemes_idx);
                assorted.push_back(frames);
            }
        } else {
            // Free decoding (argmax)
            for (int i = 0; i < batch_size; i++) {
                int T = pred_lens[i];
                std::vector<int> pred_phonemes(T);
                for (int t = 0; t < T; t++) {
                    int best_c = 0;
                    float best_val = log_probs_flat[(i * T_max + t) * C];
                    for (int c = 1; c < C; c++) {
                        float val = log_probs_flat[(i * T_max + t) * C + c];
                        if (val > best_val) { best_val = val; best_c = c; }
                    }
                    pred_phonemes[t] = best_c;
                }
                std::vector<int> neg_ones(T, -1);
                auto frames = viterbi_decoder_.assort_frames(pred_phonemes, neg_ones);
                assorted.push_back(frames);
            }
        }

        return assorted;
    }
};

class CUPEONNXPredictor
{
private:
    struct MemoryUsage {
        long rss_mb;
        long vms_mb;
        long max_rss_mb;
    };
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    

    static MemoryUsage get_memory_usage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        MemoryUsage mem;
        mem.rss_mb = usage.ru_maxrss / 1024;
        mem.max_rss_mb = usage.ru_maxrss / 1024;
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
    CUPEONNXPredictor(const std::string &onnx_path, int use_cuda_device = -1,
                  const std::vector<std::string> &providers = {"CUDAExecutionProvider", "CPUExecutionProvider"})
    {
        auto mem_before = get_memory_usage();
        std::cout << "=== MEMORY BEFORE MODEL LOAD ===" << std::endl;
        std::cout << "RSS: " << mem_before.rss_mb << " MB" << std::endl;

        // MUST create Env FIRST, before SessionOptions
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CUPEONNX");
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_cuda_device >= 0) {
            OrtCUDAProviderOptions cuda_options{};
            memset(&cuda_options, 0, sizeof(cuda_options));
            cuda_options.device_id = use_cuda_device;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

        memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), session_options);

        auto mem_after = get_memory_usage();
        std::cout << "=== MEMORY AFTER MODEL LOAD ===" << std::endl;
        std::cout << "RSS: " << mem_after.rss_mb << " MB (+D"
                  << (mem_after.rss_mb - mem_before.rss_mb) << " MB)" << std::endl;
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

        // Flatten batch for ONNX
        std::vector<float> flat_input(batch_size * audio_length);
        for (size_t b = 0; b < batch_size; ++b)
            std::copy(audio_batch[b].begin(), audio_batch[b].end(), flat_input.begin() + b * audio_length);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_, flat_input.data(), flat_input.size(), input_shape.data(), input_shape.size());

        const char *input_names[] = {"audio_window"};
        const char *output_names[] = {"phoneme_logits", "group_logits", "embeddings"};

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                            input_names, &input_tensor, 1,
                                            output_names, 3);

        const float *class_ptr = output_tensors[0].GetTensorData<float>();
        const float *group_ptr = output_tensors[1].GetTensorData<float>();

        auto class_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto group_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        auto emb_shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();

        std::vector<float> logits_class(class_ptr, class_ptr + std::accumulate(class_shape.begin(), class_shape.end(), 1LL, std::multiplies<>()));
        std::vector<float> logits_group(group_ptr, group_ptr + std::accumulate(group_shape.begin(), group_shape.end(), 1LL, std::multiplies<>()));

        std::vector<float> embeddings;
        if (return_embeddings && output_tensors.size() > 2) {
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

// ============================================================
// Timestamp result structures
// ============================================================

struct Timestamp {
    int phoneme_idx;
    int start_frame;
    int end_frame;
    float confidence;  // In simplified pipeline: target_seq_idx cast to float
    float start_ms;
    float end_ms;

    Timestamp(int idx, int sf, int ef, float conf, float sms, float ems)
        : phoneme_idx(idx), start_frame(sf), end_frame(ef),
          confidence(conf), start_ms(sms), end_ms(ems) {}
};

struct TimestampResult {
    std::vector<Timestamp> phoneme_timestamps;
};

// ============================================================
// Frame-to-millisecond conversion (matches Python convert_to_ms)
// ============================================================

std::vector<Timestamp> convert_to_ms(
    const std::vector<FrameStamp>& framestamps,
    int spectral_length,
    float start_offset_time,
    int wav_len,
    int sample_rate)
{
    float duration_in_seconds = static_cast<float>(wav_len) / sample_rate;
    float duration_per_frame = (spectral_length > 0)
        ? duration_in_seconds / spectral_length : 0.0f;

    std::vector<Timestamp> updated;
    for (const auto& fs : framestamps) {
        float start_sec = start_offset_time + (fs.start_frame * duration_per_frame);
        float end_sec = start_offset_time + (fs.end_frame * duration_per_frame);
        float start_ms = start_sec * 1000.0f;
        float end_ms = end_sec * 1000.0f;

        updated.emplace_back(fs.phoneme_id, fs.start_frame, fs.end_frame,
                             static_cast<float>(fs.target_seq_idx), start_ms, end_ms);
    }
    return updated;
}

// ============================================================
// Confidence calculation (matches Python _calculate_confidences)
// ============================================================

std::vector<FrameStamp> calculate_confidences(
    const std::vector<float>& log_probs_flat, int T, int C,
    const std::vector<FrameStamp>& framestamps)
{
    std::vector<FrameStamp> updated;

    for (const auto& fs : framestamps) {
        int start_frame = std::max(0, fs.start_frame);
        int end_frame = std::min(T, fs.end_frame);
        int phoneme_idx = fs.phoneme_id;

        float avg_confidence = std::exp(log_probs_flat[start_frame * C + phoneme_idx]);
        int new_end_frame = end_frame;

        if (start_frame < end_frame && phoneme_idx >= 0 && phoneme_idx < C) {
            float half_confidence = avg_confidence / 2.0f;
            int last_good_frame = start_frame;
            int total_good_frames = 1;

            for (int f = start_frame + 1; f < end_frame; f++) {
                float frame_prob = std::exp(log_probs_flat[f * C + phoneme_idx]);
                if (frame_prob > half_confidence || frame_prob > 0.1f) {
                    avg_confidence += frame_prob;
                    last_good_frame = f;
                    total_good_frames++;
                }
            }

            if (total_good_frames > 1) {
                avg_confidence /= total_good_frames;
                new_end_frame = std::min(T, last_good_frame + 1);

                // Check max confidence in adjusted range
                float max_confidence = 0.0f;
                for (int f = start_frame; f < new_end_frame; f++) {
                    float p = std::exp(log_probs_flat[f * C + phoneme_idx]);
                    if (p > max_confidence) max_confidence = p;
                }
                if (avg_confidence < max_confidence / 2.0f)
                    avg_confidence = max_confidence;
            }
        }

        FrameStamp updated_fs = {phoneme_idx, start_frame, new_end_frame,
                                 fs.target_seq_idx, avg_confidence};
        updated.push_back(updated_fs);
    }

    return updated;
}

/**
 * Frame-to-ms conversion using FrameStamp.confidence instead of target_seq_idx.
 * For use in the advanced pipeline after calculate_confidences.
 */
std::vector<Timestamp> convert_to_ms_with_confidence(
    const std::vector<FrameStamp>& framestamps,
    int spectral_length,
    float start_offset_time,
    int wav_len,
    int sample_rate)
{
    float duration_in_seconds = static_cast<float>(wav_len) / sample_rate;
    float duration_per_frame = (spectral_length > 0)
        ? duration_in_seconds / spectral_length : 0.0f;

    std::vector<Timestamp> updated;
    for (const auto& fs : framestamps) {
        float start_sec = start_offset_time + (fs.start_frame * duration_per_frame);
        float end_sec = start_offset_time + (fs.end_frame * duration_per_frame);
        float start_ms = start_sec * 1000.0f;
        float end_ms = end_sec * 1000.0f;

        updated.emplace_back(fs.phoneme_id, fs.start_frame, fs.end_frame,
                             fs.confidence, start_ms, end_ms);
    }
    return updated;
}

struct PhonemeResult {
    std::vector<int> ph66;
    std::vector<int> pg16;
    std::vector<std::string> ipa;
    std::vector<int> word_num;
    std::vector<std::string> words;
    std::string text;
};

// ============================================================
// PhonemeTimestampAligner — matches Python class in bfaonnx.py
// ============================================================

class PhonemeTimestampAligner
{
private:
    std::unique_ptr<CUPEONNXPredictor> extractor_;
    int resampler_sample_rate_ = 16000;
    int ph_seq_min_ = 1;

    double seg_duration_min_ = 0.05;
    int seg_duration_min_samples_;
    double seg_duration_max_;
    int wav_len_max_;

    std::string selected_mapper_;

    // Mappers
    std::unordered_map<int, std::string> phoneme_id_to_label_;
    std::unordered_map<std::string, int> phoneme_label_to_id_;
    std::unordered_map<int, std::string> group_id_to_label_;
    std::unordered_map<std::string, int> group_label_to_id_;
    std::unordered_map<int, int> phoneme_id_to_group_id_;

    int silence_anchors_;
    bool ignore_noise_;
    bool boost_targets_ = true;
    bool enforce_minimum_ = true;

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
    PhonemeTimestampAligner(const std::string &cupe_ckpt_path,
                            const std::string &lang = "en-us",
                            const std::string &mapper = "ph66",
                            double duration_max = 10.0,
                            const std::string &output_frames_key = "phoneme_idx",
                            int silence_anchors = 10,
                            bool ignore_noise = true,
                            bool boost_targets = true,
                            bool enforce_minimum = true);

    std::pair<std::vector<float>, int> chop_wav(
        const std::vector<float>& wav, int start_frame, int end_frame);

    TimestampResult extract_timestamps_from_segment_simplified(
        const std::vector<float>& wav, int wav_len,
        const std::vector<int>& phoneme_sequence,
        float start_offset_time = 0.0f);

    PhonemeResult phonemize_sentence(const std::string &text);

    std::vector<int> map_phonemes_to_groups(const std::vector<int>& phoneme_sequence);

    TimestampResult extract_timestamps_from_segment_advanced(
        const std::vector<float>& wav, int wav_len,
        const std::vector<int>& phoneme_sequence,
        float start_offset_time = 0.0f,
        bool debug = false);

private:
    void _setup_config(int window_size_ms = 120, int stride_ms = 80);
    void _setup_decoders();
    std::vector<float> rms_normalize(const std::vector<float>& audio);

    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
               int, int, int, int>
    cupe_prediction(const std::vector<std::vector<float>> &audio_batch, int wav_len);
};

// ============================================================
// PhonemeTimestampAligner implementation
// ============================================================

PhonemeTimestampAligner::PhonemeTimestampAligner(
    const std::string &cupe_ckpt_path,
    const std::string &lang,
    const std::string &mapper,
    double duration_max,
    const std::string &output_frames_key,
    int silence_anchors,
    bool ignore_noise,
    bool boost_targets,
    bool enforce_minimum)
    : selected_mapper_(mapper),
      seg_duration_max_(duration_max),
      silence_anchors_(silence_anchors),
      ignore_noise_(ignore_noise),
      boost_targets_(boost_targets),
      enforce_minimum_(enforce_minimum)
{
    if (cupe_ckpt_path.empty()) {
        throw std::runtime_error("cupe_ckpt_path must be provided.");
    }
    std::cout << "Using CUPE checkpoint: " << cupe_ckpt_path << std::endl;

    extractor_ = std::make_unique<CUPEONNXPredictor>(cupe_ckpt_path, -1);

    seg_duration_min_samples_ = static_cast<int>(seg_duration_min_ * resampler_sample_rate_);
    wav_len_max_ = static_cast<int>(seg_duration_max_ * resampler_sample_rate_);

    if (selected_mapper_ != "ph66")
        throw std::runtime_error("Currently only 'ph66' mapper is supported.");

    // Initialize mappers from global tables
    phoneme_label_to_id_ = phoneme_mapped_index;
    for (const auto &pair : phoneme_mapped_index)
        phoneme_id_to_label_[pair.second] = pair.first;

    group_label_to_id_ = phoneme_groups_index;
    for (const auto &pair : phoneme_groups_index)
        group_id_to_label_[pair.second] = pair.first;

    phoneme_id_to_group_id_ = phoneme_groups_mapper;

    _setup_config();
    _setup_decoders();
}

void PhonemeTimestampAligner::_setup_config(int window_size_ms, int stride_ms)
{
    window_size_ms_ = window_size_ms;
    stride_ms_ = stride_ms;
    sample_rate_ = 16000;

    window_size_wav_ = window_size_ms_ * sample_rate_ / 1000;
    stride_size_wav_ = stride_ms_ * sample_rate_ / 1000;

    phoneme_classes_ = phoneme_id_to_label_.size() - 1;  // exclude noise
    phoneme_groups_ = group_id_to_label_.size() - 1;     // exclude noise
    blank_class_ = phoneme_label_to_id_["noise"];   // 66
    blank_group_ = group_label_to_id_["noise"];     // 16
    silence_class_ = phoneme_label_to_id_["SIL"];   // 0
    silence_group_ = group_label_to_id_["SIL"];     // 0

    std::cout << "Config: phoneme_classes=" << phoneme_classes_
              << ", groups=" << phoneme_groups_
              << ", blank_class=" << blank_class_
              << ", blank_group=" << blank_group_
              << ", silence_class=" << silence_class_
              << ", silence_group=" << silence_group_ << std::endl;
}

void PhonemeTimestampAligner::_setup_decoders()
{
    alignment_utils_g_ = std::make_unique<AlignmentUtils>(
        blank_group_, silence_class_, silence_anchors_, ignore_noise_);
    alignment_utils_p_ = std::make_unique<AlignmentUtils>(
        blank_class_, silence_group_, silence_anchors_, ignore_noise_);
}

std::vector<float> PhonemeTimestampAligner::rms_normalize(const std::vector<float>& audio) {
    double sum_squares = 0.0;
    for (float sample : audio)
        sum_squares += sample * sample;

    double rms = std::sqrt(sum_squares / audio.size());
    std::vector<float> normalized(audio.size());

    if (rms > 0.0) {
        for (size_t i = 0; i < audio.size(); ++i)
            normalized[i] = audio[i] / static_cast<float>(rms);
    } else {
        normalized = audio;
    }
    return normalized;
}

std::pair<std::vector<float>, int> PhonemeTimestampAligner::chop_wav(
    const std::vector<float>& input_wav,
    int start_frame,
    int end_frame)
{
    int num_frames = (end_frame != -1) ? (end_frame - start_frame) : -1;

    if (num_frames != -1 && num_frames < seg_duration_min_samples_) {
        throw std::invalid_argument("Segment too short: " + std::to_string(num_frames) +
                                    " frames, minimum required is " +
                                    std::to_string(seg_duration_min_samples_));
    }

    // Slice audio
    std::vector<float> wav_slice;
    if (end_frame == -1)
        wav_slice.assign(input_wav.begin() + start_frame, input_wav.end());
    else
        wav_slice.assign(input_wav.begin() + start_frame, input_wav.begin() + end_frame);

    if (static_cast<int>(wav_slice.size()) < seg_duration_min_samples_) {
        throw std::runtime_error("Wav too small: size=" + std::to_string(wav_slice.size()));
    }

    // RMS normalize
    std::vector<float> wav_mono = rms_normalize(wav_slice);
    int wav_len = wav_mono.size();

    // Pad or truncate to wav_len_max_
    std::vector<float> wav_processed;
    if (wav_len > wav_len_max_) {
        wav_processed.assign(wav_mono.begin(), wav_mono.begin() + wav_len_max_);
        wav_len = wav_len_max_;
    } else {
        wav_processed.resize(wav_len_max_, 0.0f);
        std::copy(wav_mono.begin(), wav_mono.end(), wav_processed.begin());
    }

    return {wav_processed, wav_len};
}

PhonemeResult PhonemeTimestampAligner::phonemize_sentence(const std::string &text)
{
    // Stub — phonemization is handled externally in C++.
    // This returns a hardcoded example for testing only.
    // In production, call espeak-ng or your phonemizer and map to ph66 indices.
    std::cerr << "WARNING: phonemize_sentence is a stub. "
              << "Use pre-computed phoneme sequences instead." << std::endl;
    return {
        {29, 10, 58, 9, 43, 56, 23},                                   // ph66 for "butterfly"
        {7, 2, 14, 2, 8, 13, 5},                                        // pg16
        {"b", "\xca\x8c", "\xc9\xbe", "\xc9\x9a", "f", "l", "a\xc9\xaa"}, // ipa
        {0, 0, 0, 0, 0, 0, 0},                                          // word_num
        {"butterfly"},                                                    // words
        text                                                              // text
    };
}

std::vector<int> PhonemeTimestampAligner::map_phonemes_to_groups(
    const std::vector<int>& phoneme_sequence)
{
    std::vector<int> group_sequence;
    for (int ph_id : phoneme_sequence) {
        auto it = phoneme_id_to_group_id_.find(ph_id);
        group_sequence.push_back(it != phoneme_id_to_group_id_.end() ? it->second : blank_group_);
    }
    return group_sequence;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           int, int, int, int>
PhonemeTimestampAligner::cupe_prediction(
    const std::vector<std::vector<float>> &audio_batch,
    int wav_len)
{
    // 1. Windowing [1, num_windows, window_size]
    auto windowed_audio = slice_windows({audio_batch[0]}, 16000, window_size_ms_, stride_ms_);

    int num_windows = windowed_audio[0].size();
    int window_size = windowed_audio[0][0].size();

    std::cout << "Windowed audio: " << windowed_audio.size() << " x " << windowed_audio[0].size()
              << " x " << windowed_audio[0][0].size() << std::endl;

    // 2. Flatten to [num_windows, window_size]
    std::vector<std::vector<float>> windows_flat(num_windows);
    for (int i = 0; i < num_windows; i++)
        windows_flat[i] = windowed_audio[0][i];

    std::cout << "Windows flattened: " << windows_flat.size() << " x " << windows_flat[0].size() << std::endl;

    // 3. ONNX prediction
    auto [logits_class, class_shape, logits_group, group_shape, embeddings, emb_shape] =
        extractor_->predict(windows_flat, false);

    int frames_per_window = group_shape[1];
    int class_dim = class_shape[2];
    int group_dim = group_shape[2];

    int total_frames_check = static_cast<int>(logits_class.size() / class_dim);  //

    assert(logits_class.size() == static_cast<size_t>(total_frames_check * class_dim));



    std::cout << "Logits class: [" << class_shape[0] << ", " << class_shape[1] << ", " << class_shape[2] << "]" << std::endl;
    std::cout << "Logits group: [" << group_shape[0] << ", " << group_shape[1] << ", " << group_shape[2] << "]" << std::endl;

    // 4. Stitch window predictions
    int original_audio_length = audio_batch[0].size();
    auto logits_class_stitched = stitch_window_predictions_flat(
        logits_class, num_windows, frames_per_window, class_dim,
        original_audio_length, sample_rate_, window_size_ms_, stride_ms_);

    auto logits_group_stitched = stitch_window_predictions_flat(
        logits_group, num_windows, frames_per_window, group_dim,
        original_audio_length, sample_rate_, window_size_ms_, stride_ms_);

    int total_frames = num_windows * frames_per_window / 2;

    std::cout << "Stitched class: " << logits_class_stitched.size()
              << ", group: " << logits_group_stitched.size() << std::endl;

    // 5. Spectral length
    int spectral_len = calc_spec_len_ext(wav_len, window_size_ms_, stride_ms_,
                                          sample_rate_, frames_per_window);
    std::cout << "Spectral length: " << spectral_len << std::endl;

    return {logits_class_stitched, logits_group_stitched, embeddings,
            spectral_len, total_frames, class_dim, group_dim};
}

/**
 * Simplified timestamp extraction — matches Python extract_timestamps_from_segment_simplified.
 * Straight CUPE -> log_softmax -> Viterbi -> timestamps.
 * No group timestamps, no boosting, no silence anchoring.
 */
TimestampResult PhonemeTimestampAligner::extract_timestamps_from_segment_simplified(
    const std::vector<float>& wav,
    int wav_len,
    const std::vector<int>& phoneme_sequence,
    float start_offset_time)
{
    TimestampResult result;
    int ph_seq_len = phoneme_sequence.size();

    // CUPE forward pass
    std::vector<std::vector<float>> audio_batch = {wav};
    auto [logits_class, logits_group, embeddings, spectral_len,
          total_frames, num_phonemes, num_groups] = cupe_prediction(audio_batch, wav_len);

    if (spectral_len <= 0) {
        std::cerr << "ERROR: Invalid spectral_len = " << spectral_len << std::endl;
        return result;
    }

    // Log softmax over class dimension
    auto log_probs_p = log_softmax_batch(logits_class, 1, total_frames, num_phonemes);

    // Simple Viterbi alignment
    std::vector<std::vector<int>> true_seqs = {phoneme_sequence};
    std::vector<int> pred_lens = {spectral_len};
    std::vector<int> true_seqs_lens = {ph_seq_len};

    auto frame_phonemes = alignment_utils_p_->decode_alignments_simple(
        log_probs_p, 1, total_frames, num_phonemes,
        true_seqs, pred_lens, true_seqs_lens);

    // Convert frames to millisecond timestamps
    if (!frame_phonemes.empty() && !frame_phonemes[0].empty()) {
        result.phoneme_timestamps = convert_to_ms(
            frame_phonemes[0], spectral_len, start_offset_time,
            wav_len, resampler_sample_rate_);
    }

    return result;
}

/**
 * Advanced timestamp extraction — matches Python extract_timestamps_from_segment_advanced.
 * Uses full forced alignment with boosting, silence anchoring, and confidence scoring.
 */
TimestampResult PhonemeTimestampAligner::extract_timestamps_from_segment_advanced(
    const std::vector<float>& wav,
    int wav_len,
    const std::vector<int>& phoneme_sequence,
    float start_offset_time,
    bool debug)
{
    TimestampResult result;
    int ph_seq_len = (int)phoneme_sequence.size();

    // CUPE forward pass
    std::vector<std::vector<float>> audio_batch = {wav};
    auto [logits_class, logits_group, embeddings, spectral_len,
          total_frames, num_phonemes, num_groups] = cupe_prediction(audio_batch, wav_len);

    if (spectral_len <= 0) {
        std::cerr << "ERROR: Invalid spectral_len = " << spectral_len << std::endl;
        return result;
    }

    // Log softmax over class dimension
    auto log_probs_p = log_softmax_batch(logits_class, 1, total_frames, num_phonemes);

    // Full alignment with boosting and silence anchoring
    std::vector<std::vector<int>> true_seqs = {phoneme_sequence};
    std::vector<int> pred_lens = {spectral_len};
    std::vector<int> true_seqs_lens = {ph_seq_len};

    auto frame_phonemes = alignment_utils_p_->decode_alignments(
        log_probs_p, 1, total_frames, num_phonemes,
        true_seqs, pred_lens, true_seqs_lens,
        true, boost_targets_, enforce_minimum_, debug);

    if (!frame_phonemes.empty() && !frame_phonemes[0].empty()) {
        // Extract per-sample log probs for confidence calculation [spectral_len * C]
        std::vector<float> lp(spectral_len * num_phonemes);
        for (int t = 0; t < spectral_len; t++)
            for (int c = 0; c < num_phonemes; c++)
                lp[t * num_phonemes + c] = log_probs_p[t * num_phonemes + c];

        // Calculate confidence scores
        auto with_confidence = calculate_confidences(lp, spectral_len, num_phonemes,
                                                     frame_phonemes[0]);

        // Convert to ms timestamps (using confidence field)
        result.phoneme_timestamps = convert_to_ms_with_confidence(
            with_confidence, spectral_len, start_offset_time,
            wav_len, resampler_sample_rate_);

        // Sort by start_ms
        std::sort(result.phoneme_timestamps.begin(), result.phoneme_timestamps.end(),
                  [](const Timestamp& a, const Timestamp& b) { return a.start_ms < b.start_ms; });
    }

    return result;
}

// ============================================================
// Main — matches Python __main__ in bfaonnx.py
// ============================================================

int main()
{
    init_reverse_lookups();

    // Configuration — update paths to match your setup
    std::string onnx_path = "../../models/en_libri1000_ua01c_e4_val_GER=0.2186.ckpt.onnx";
    std::string audio_path = "../../examples/samples/audio/109867__timkahn__butterfly.wav";

    // Phoneme sequence for "butterfly" (pre-computed from espeak-ng)
    // IPA:  b  ʌ  ɾ  ɚ  f  l  aɪ
    // ph66: 29 10 58 9  43 56 23
    std::vector<int> phoneme_seq = {29, 10, 58, 9, 43, 56, 23};
    std::vector<std::string> ipa_labels = {"b", "\xca\x8c", "\xc9\xbe", "\xc9\x9a", "f", "l", "a\xc9\xaa"};

    std::cout << "Initializing CUPE Phoneme Aligner..." << std::endl;

    // Initialize aligner
    PhonemeTimestampAligner extractor(onnx_path, "en-us", "ph66", 10.0);

    // Load audio
    float original_duration = 0.0f;
    std::vector<float> wav_float = loadAudioFromWav(audio_path, original_duration);
    if (wav_float.empty()) {
        std::cerr << "Failed to load " << audio_path << std::endl;
        return -1;
    }
    std::cout << "Audio is loaded" << std::endl;
    std::cout << "Text: 'butterfly'" << std::endl;
    std::cout << "IPA:  ";
    for (const auto& ipa : ipa_labels) std::cout << ipa << " ";
    std::cout << std::endl;
    std::cout << "ph66: ";
    for (int p : phoneme_seq) std::cout << p << " ";
    std::cout << std::endl;

    // Chop/pad audio
    auto [wav, wav_len] = extractor.chop_wav(wav_float, 0, wav_float.size());

    // Run simplified pipeline
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = extractor.extract_timestamps_from_segment_simplified(
        wav, wav_len, phoneme_seq, 0.0f);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // Print results
    const auto& phoneme_ts = result.phoneme_timestamps;
    std::cout << "\nAligned " << phoneme_ts.size()
              << " phonemes in " << elapsed_ms << " ms:\n" << std::endl;

    for (size_t i = 0; i < phoneme_ts.size(); i++) {
        const auto& ts = phoneme_ts[i];
        std::string label = "?";
        auto it = index_to_plabel.find(ts.phoneme_idx);
        if (it != index_to_plabel.end()) label = it->second;

        // FIX: Use target_seq_idx (stored in confidence field) to index ipa_labels
        std::string ipa = "?";
        int target_idx = static_cast<int>(ts.confidence);
        if (target_idx >= 0 && target_idx < static_cast<int>(ipa_labels.size())) {
            ipa = ipa_labels[target_idx];
        }

        std::cout << "  " << std::setw(2) << (i + 1) << ": "
                  << std::setw(6) << std::right << label
                  << " (" << std::setw(3) << std::right << ipa << ")  "
                  << std::fixed << std::setprecision(1)
                  << std::setw(7) << ts.start_ms << " - "
                  << std::setw(7) << ts.end_ms << " ms" << std::endl;
    }

    // ---- Advanced pipeline (with boosting, silence anchoring, confidence) ----
    std::cout << "\n===== ADVANCED PIPELINE =====" << std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();
    auto result_adv = extractor.extract_timestamps_from_segment_advanced(
        wav, wav_len, phoneme_seq, 0.0f, false);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto elapsed_adv = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    const auto& adv_ts = result_adv.phoneme_timestamps;
    std::cout << "\nAligned " << adv_ts.size()
              << " phonemes (advanced) in " << elapsed_adv << " ms:\n" << std::endl;

    for (size_t i = 0; i < adv_ts.size(); i++) {
        const auto& ts = adv_ts[i];
        std::string label = "?";
        auto it = index_to_plabel.find(ts.phoneme_idx);
        if (it != index_to_plabel.end()) label = it->second;

        std::cout << "  " << std::setw(2) << (i + 1) << ": "
                  << std::setw(6) << std::right << label
                  << "  conf=" << std::fixed << std::setprecision(3)
                  << std::setw(6) << ts.confidence << "  "
                  << std::setprecision(1)
                  << std::setw(7) << ts.start_ms << " - "
                  << std::setw(7) << ts.end_ms << " ms" << std::endl;
    }

    return 0;
}


'''

## espeak-ng integration
### The command structure:
espeak-ng [options] ["<words>"]
```bash
espeak-ng --ipa ["hello"]
```
It prints the IPA phonemes for "hello". followed by a newline. Then there are a lot of error messages about "Failed to open voice data file". These can be ignored as long as the IPA output is correct. The phonemes can be mapped to ph66 indices using the `phoneme_mapped_index` table.

First, for a sentence, we need to break it into words, then phonemize each word separately to get the phoneme sequence. This is because espeak-ng\'s output can differ based on word boundaries and context.


## Notes for cli options:
- onnx_path: path to ONNX model file (required)
- audio_path: path to input WAV file (required)
- mapper: phoneme mapping to use (default "ph66", currently only ph66 is supported)
- duration_max: maximum segment duration in seconds (default 10.0)

'''