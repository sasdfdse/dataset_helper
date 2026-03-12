#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
}

#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <functional>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

using ProgressCb = std::function<void(int pct, const std::string& msg)>;

// ─── Utility ──────────────────────────────────────────────────────
inline std::string humanSize(uintmax_t bytes) {
    const char* units[] = {"B","KB","MB","GB","TB"};
    double val = (double)bytes;
    int idx = 0;
    while (val >= 1024.0 && idx < 4) { val /= 1024.0; idx++; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << val << " " << units[idx];
    return ss.str();
}

// ─── Codec compatibility check ────────────────────────────────────
inline bool isMp4CompatibleVideo(AVCodecID id) {
    return id == AV_CODEC_ID_H264 || id == AV_CODEC_ID_HEVC ||
           id == AV_CODEC_ID_MPEG4 || id == AV_CODEC_ID_MPEG2VIDEO ||
           id == AV_CODEC_ID_H263;
}

inline bool isMp4CompatibleAudio(AVCodecID id) {
    return id == AV_CODEC_ID_AAC || id == AV_CODEC_ID_MP3 ||
           id == AV_CODEC_ID_AC3 || id == AV_CODEC_ID_MP2;
}

// ─── saveFrameAsJPEG ──────────────────────────────────────────────
inline bool saveFrameAsJPEG(AVFrame* frame, const std::string& path) {
    const AVCodec* jpegCodec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!jpegCodec) return false;
    AVCodecContext* ctx = avcodec_alloc_context3(jpegCodec);
    if (!ctx) return false;

    ctx->width    = frame->width;
    ctx->height   = frame->height;
    ctx->pix_fmt  = AV_PIX_FMT_YUVJ420P;
    ctx->time_base = {1, 25};
    av_opt_set_int(ctx, "qscale", 2, AV_OPT_SEARCH_CHILDREN);

    if (avcodec_open2(ctx, jpegCodec, nullptr) < 0) {
        avcodec_free_context(&ctx); return false;
    }

    SwsContext* sws = sws_getContext(
        frame->width, frame->height, (AVPixelFormat)frame->format,
        frame->width, frame->height, AV_PIX_FMT_YUVJ420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVFrame* dst = av_frame_alloc();
    dst->format = AV_PIX_FMT_YUVJ420P;
    dst->width  = frame->width;
    dst->height = frame->height;
    av_frame_get_buffer(dst, 32);
    sws_scale(sws, frame->data, frame->linesize, 0, frame->height, dst->data, dst->linesize);

    AVPacket* pkt = av_packet_alloc();
    bool ok = false;
    if (avcodec_send_frame(ctx, dst) == 0) {
        if (avcodec_receive_packet(ctx, pkt) == 0) {
            std::ofstream ofs(path, std::ios::binary);
            if (ofs) { ofs.write((char*)pkt->data, pkt->size); ok = ofs.good(); }
        }
    }

    av_packet_free(&pkt);
    av_frame_free(&dst);
    sws_freeContext(sws);
    avcodec_free_context(&ctx);
    return ok;
}

// ═══════════════════════════════════════════════════════════════════
// convertWebmToMp4  (with re-encoding for incompatible codecs)
// ═══════════════════════════════════════════════════════════════════

struct ConvertResult {
    bool success = false;
    std::string message;
    double elapsedSec = 0;
    uintmax_t outputSize = 0;
};

struct TranscodeStream {
    int inIdx  = -1;
    int outIdx = -1;
    bool recode = false;
    AVStream*       inStream  = nullptr;
    AVStream*       outStream = nullptr;
    AVCodecContext* dec = nullptr;
    AVCodecContext* enc = nullptr;
    SwsContext*     sws = nullptr;  // video pixel format conversion
    SwrContext*     swr = nullptr;  // audio sample format conversion
};

inline ConvertResult convertWebmToMp4(const std::string& inputPath,
                                       const std::string& outputPath,
                                       ProgressCb progress = nullptr) {
    auto t0 = std::chrono::steady_clock::now();
    ConvertResult result;

    // ── Open input
    AVFormatContext* inFmt = nullptr;
    if (avformat_open_input(&inFmt, inputPath.c_str(), nullptr, nullptr) < 0) {
        result.message = "입력 파일을 열 수 없습니다: " + inputPath; return result;
    }
    if (avformat_find_stream_info(inFmt, nullptr) < 0) {
        avformat_close_input(&inFmt);
        result.message = "스트림 정보를 읽을 수 없습니다."; return result;
    }

    // ── Open output
    AVFormatContext* outFmt = nullptr;
    if (avformat_alloc_output_context2(&outFmt, nullptr, "mp4", outputPath.c_str()) < 0) {
        avformat_close_input(&inFmt);
        result.message = "출력 컨텍스트 생성 실패"; return result;
    }

    // ── Set up streams (decoder + encoder where needed)
    std::vector<TranscodeStream> streams;
    std::vector<int> streamMap(inFmt->nb_streams, -1);

    for (unsigned i = 0; i < inFmt->nb_streams; i++) {
        AVStream* inSt = inFmt->streams[i];
        AVMediaType type = inSt->codecpar->codec_type;
        if (type != AVMEDIA_TYPE_VIDEO && type != AVMEDIA_TYPE_AUDIO) continue;

        TranscodeStream ts;
        ts.inIdx    = (int)i;
        ts.inStream = inSt;
        ts.recode   = (type == AVMEDIA_TYPE_VIDEO)
                      ? !isMp4CompatibleVideo(inSt->codecpar->codec_id)
                      : !isMp4CompatibleAudio(inSt->codecpar->codec_id);

        if (ts.recode) {
            // ── Decoder
            const AVCodec* decCodec = avcodec_find_decoder(inSt->codecpar->codec_id);
            if (!decCodec) continue;
            ts.dec = avcodec_alloc_context3(decCodec);
            avcodec_parameters_to_context(ts.dec, inSt->codecpar);
            if (avcodec_open2(ts.dec, decCodec, nullptr) < 0) {
                avcodec_free_context(&ts.dec); continue;
            }

            // ── Encoder
            if (type == AVMEDIA_TYPE_VIDEO) {
                const AVCodec* encCodec = avcodec_find_encoder_by_name("libx264");
                if (!encCodec) encCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
                if (!encCodec) { avcodec_free_context(&ts.dec); continue; }
                ts.enc = avcodec_alloc_context3(encCodec);
                ts.enc->width    = inSt->codecpar->width;
                ts.enc->height   = inSt->codecpar->height;
                ts.enc->pix_fmt  = AV_PIX_FMT_YUV420P;
                ts.enc->time_base = (inSt->avg_frame_rate.num > 0)
                                    ? av_inv_q(inSt->avg_frame_rate)
                                    : AVRational{1, 30};
                if (outFmt->oformat->flags & AVFMT_GLOBALHEADER)
                    ts.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                AVDictionary* encOpts = nullptr;
                av_dict_set(&encOpts, "preset", "fast", 0);
                av_dict_set(&encOpts, "crf", "23", 0);
                if (avcodec_open2(ts.enc, encCodec, &encOpts) < 0) {
                    av_dict_free(&encOpts);
                    avcodec_free_context(&ts.dec); avcodec_free_context(&ts.enc); continue;
                }
                av_dict_free(&encOpts);
            } else {
                const AVCodec* encCodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
                if (!encCodec) { avcodec_free_context(&ts.dec); continue; }
                ts.enc = avcodec_alloc_context3(encCodec);
                ts.enc->sample_rate    = inSt->codecpar->sample_rate;
                ts.enc->channels       = inSt->codecpar->channels;
                ts.enc->channel_layout = inSt->codecpar->channel_layout
                                         ? inSt->codecpar->channel_layout
                                         : av_get_default_channel_layout(inSt->codecpar->channels);
                ts.enc->sample_fmt  = AV_SAMPLE_FMT_FLTP;
                ts.enc->bit_rate    = 128000;
                ts.enc->time_base   = {1, inSt->codecpar->sample_rate};
                if (outFmt->oformat->flags & AVFMT_GLOBALHEADER)
                    ts.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                if (avcodec_open2(ts.enc, encCodec, nullptr) < 0) {
                    avcodec_free_context(&ts.dec); avcodec_free_context(&ts.enc); continue;
                }
            }
        }

        // ── Output stream
        AVStream* outSt = avformat_new_stream(outFmt, nullptr);
        if (!outSt) {
            if (ts.dec) avcodec_free_context(&ts.dec);
            if (ts.enc) avcodec_free_context(&ts.enc);
            continue;
        }
        ts.outStream = outSt;
        ts.outIdx    = outSt->index;
        streamMap[i] = outSt->index;

        if (ts.recode) {
            avcodec_parameters_from_context(outSt->codecpar, ts.enc);
            outSt->time_base = ts.enc->time_base;
        } else {
            avcodec_parameters_copy(outSt->codecpar, inSt->codecpar);
            outSt->codecpar->codec_tag = 0;
            outSt->time_base = inSt->time_base;
        }
        streams.push_back(ts);
    }

    // ── Open output file
    if (!(outFmt->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&outFmt->pb, outputPath.c_str(), AVIO_FLAG_WRITE) < 0) {
            avformat_close_input(&inFmt);
            for (auto& s : streams) {
                if (s.dec) avcodec_free_context(&s.dec);
                if (s.enc) avcodec_free_context(&s.enc);
            }
            avformat_free_context(outFmt);
            result.message = "출력 파일을 열 수 없습니다."; return result;
        }
    }

    AVDictionary* muxOpts = nullptr;
    av_dict_set(&muxOpts, "movflags", "faststart", 0);
    if (avformat_write_header(outFmt, &muxOpts) < 0) {
        av_dict_free(&muxOpts);
        if (!(outFmt->oformat->flags & AVFMT_NOFILE)) avio_closep(&outFmt->pb);
        avformat_close_input(&inFmt);
        avformat_free_context(outFmt);
        for (auto& s : streams) {
            if (s.dec) avcodec_free_context(&s.dec);
            if (s.enc) avcodec_free_context(&s.enc);
        }
        result.message = "헤더 쓰기 실패"; return result;
    }
    av_dict_free(&muxOpts);

    // ── Process packets
    auto findTs = [&](int inIdx) -> TranscodeStream* {
        for (auto& s : streams) if (s.inIdx == inIdx) return &s;
        return nullptr;
    };

    int64_t duration = inFmt->duration;
    AVPacket* pkt    = av_packet_alloc();
    AVFrame*  frame  = av_frame_alloc();

    while (av_read_frame(inFmt, pkt) >= 0) {
        int si = pkt->stream_index;
        if (si < 0 || (unsigned)si >= inFmt->nb_streams || streamMap[si] < 0) {
            av_packet_unref(pkt); continue;
        }
        TranscodeStream* ts = findTs(si);
        if (!ts) { av_packet_unref(pkt); continue; }

        if (!ts->recode) {
            // ── Stream copy
            av_packet_rescale_ts(pkt, ts->inStream->time_base, ts->outStream->time_base);
            pkt->stream_index = ts->outIdx;
            pkt->pos = -1;
            if (pkt->pts != AV_NOPTS_VALUE && progress && duration > 0) {
                int64_t pts_us = av_rescale_q(pkt->pts, ts->outStream->time_base, {1, 1000000});
                progress(std::min((int)(100.0 * pts_us / duration), 99), "변환 중...");
            }
            av_interleaved_write_frame(outFmt, pkt);
        } else {
            // ── Decode → re-encode
            if (avcodec_send_packet(ts->dec, pkt) == 0) {
                while (avcodec_receive_frame(ts->dec, frame) == 0) {
                    // Progress (video only)
                    if (progress && duration > 0 && frame->pts != AV_NOPTS_VALUE &&
                        ts->inStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                        int64_t pts_us = av_rescale_q(frame->pts,
                                                       ts->inStream->time_base, {1, 1000000});
                        progress(std::min((int)(100.0 * pts_us / duration), 99), "변환 중...");
                    }

                    AVFrame* toEncode = frame;
                    AVFrame* convFrame = nullptr;

                    if (ts->inStream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                        // Pixel format conversion if needed
                        if ((AVPixelFormat)frame->format != ts->enc->pix_fmt) {
                            if (!ts->sws) {
                                ts->sws = sws_getContext(
                                    frame->width, frame->height, (AVPixelFormat)frame->format,
                                    ts->enc->width, ts->enc->height, ts->enc->pix_fmt,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                            }
                            convFrame = av_frame_alloc();
                            convFrame->format = ts->enc->pix_fmt;
                            convFrame->width  = ts->enc->width;
                            convFrame->height = ts->enc->height;
                            av_frame_get_buffer(convFrame, 32);
                            sws_scale(ts->sws, frame->data, frame->linesize, 0, frame->height,
                                      convFrame->data, convFrame->linesize);
                            convFrame->pts = frame->pts;
                            toEncode = convFrame;
                        }
                    } else {
                        // Sample format conversion if needed
                        if ((AVSampleFormat)frame->format != ts->enc->sample_fmt ||
                            frame->sample_rate != ts->enc->sample_rate) {
                            if (!ts->swr) {
                                ts->swr = swr_alloc();
                                av_opt_set_int(ts->swr, "in_channel_layout",
                                               frame->channel_layout
                                               ? frame->channel_layout
                                               : av_get_default_channel_layout(frame->channels), 0);
                                av_opt_set_int(ts->swr, "out_channel_layout",
                                               ts->enc->channel_layout, 0);
                                av_opt_set_int(ts->swr, "in_sample_rate",  frame->sample_rate, 0);
                                av_opt_set_int(ts->swr, "out_sample_rate", ts->enc->sample_rate, 0);
                                av_opt_set_sample_fmt(ts->swr, "in_sample_fmt",
                                                      (AVSampleFormat)frame->format, 0);
                                av_opt_set_sample_fmt(ts->swr, "out_sample_fmt",
                                                      ts->enc->sample_fmt, 0);
                                swr_init(ts->swr);
                            }
                            convFrame = av_frame_alloc();
                            convFrame->format         = ts->enc->sample_fmt;
                            convFrame->channel_layout = ts->enc->channel_layout;
                            convFrame->channels       = ts->enc->channels;
                            convFrame->sample_rate    = ts->enc->sample_rate;
                            convFrame->nb_samples     = frame->nb_samples;
                            av_frame_get_buffer(convFrame, 0);
                            swr_convert(ts->swr,
                                        convFrame->data, convFrame->nb_samples,
                                        (const uint8_t**)frame->data, frame->nb_samples);
                            convFrame->pts = av_rescale_q(frame->pts,
                                                           ts->inStream->time_base,
                                                           ts->enc->time_base);
                            toEncode = convFrame;
                        }
                    }

                    // Encode
                    AVPacket* outPkt = av_packet_alloc();
                    if (avcodec_send_frame(ts->enc, toEncode) == 0) {
                        while (avcodec_receive_packet(ts->enc, outPkt) == 0) {
                            av_packet_rescale_ts(outPkt,
                                                  ts->enc->time_base,
                                                  ts->outStream->time_base);
                            outPkt->stream_index = ts->outIdx;
                            av_interleaved_write_frame(outFmt, outPkt);
                            av_packet_unref(outPkt);
                        }
                    }
                    av_packet_free(&outPkt);

                    if (convFrame) {
                        av_frame_free(&convFrame);
                    }
                    av_frame_unref(frame);
                }
            }
        }
        av_packet_unref(pkt);
    }

    // ── Flush encoders
    for (auto& ts : streams) {
        if (!ts.recode || !ts.enc) continue;
        avcodec_send_frame(ts.enc, nullptr);
        AVPacket* outPkt = av_packet_alloc();
        while (avcodec_receive_packet(ts.enc, outPkt) == 0) {
            av_packet_rescale_ts(outPkt, ts.enc->time_base, ts.outStream->time_base);
            outPkt->stream_index = ts.outIdx;
            av_interleaved_write_frame(outFmt, outPkt);
            av_packet_unref(outPkt);
        }
        av_packet_free(&outPkt);
    }

    av_write_trailer(outFmt);
    av_frame_free(&frame);
    av_packet_free(&pkt);

    for (auto& ts : streams) {
        if (ts.sws) sws_freeContext(ts.sws);
        if (ts.swr) swr_free(&ts.swr);
        if (ts.dec) avcodec_free_context(&ts.dec);
        if (ts.enc) avcodec_free_context(&ts.enc);
    }

    if (!(outFmt->oformat->flags & AVFMT_NOFILE)) avio_closep(&outFmt->pb);
    avformat_close_input(&inFmt);
    avformat_free_context(outFmt);

    if (progress) progress(100, "완료!");
    auto t1 = std::chrono::steady_clock::now();
    result.success    = true;
    result.elapsedSec = std::chrono::duration<double>(t1 - t0).count();
    if (fs::exists(outputPath)) result.outputSize = fs::file_size(outputPath);
    return result;
}

// ═══════════════════════════════════════════════════════════════════
// extractFrames
// ═══════════════════════════════════════════════════════════════════

struct ExtractOptions {
    int  ratio     = 1;
    bool grayscale = false;
    std::string format = "jpg";
    std::string outDir;
};

struct ExtractResult {
    bool success = false;
    std::string message;
    int totalFrames = 0;
    int savedFrames = 0;
    double elapsedSec = 0;
};

inline ExtractResult extractFrames(const std::string& videoPath,
                                    const ExtractOptions& opts,
                                    ProgressCb progress = nullptr) {
    auto t0 = std::chrono::steady_clock::now();
    ExtractResult res;

    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, videoPath.c_str(), nullptr, nullptr) < 0) {
        res.message = "파일을 열 수 없습니다: " + videoPath; return res;
    }
    avformat_find_stream_info(fmt, nullptr);

    int vidIdx = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            vidIdx = (int)i; break;
        }
    }
    if (vidIdx < 0) {
        avformat_close_input(&fmt);
        res.message = "비디오 스트림을 찾을 수 없습니다."; return res;
    }

    AVStream* vidStream = fmt->streams[vidIdx];
    const AVCodec* codec = avcodec_find_decoder(vidStream->codecpar->codec_id);
    if (!codec) {
        avformat_close_input(&fmt);
        res.message = "디코더를 찾을 수 없습니다."; return res;
    }

    AVCodecContext* dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, vidStream->codecpar);
    avcodec_open2(dec, codec, nullptr);
    fs::create_directories(opts.outDir);

    int64_t totalEst = 0;
    if (vidStream->nb_frames > 0)
        totalEst = vidStream->nb_frames;
    else if (fmt->duration > 0 && vidStream->avg_frame_rate.num > 0)
        totalEst = (int64_t)(fmt->duration / AV_TIME_BASE * av_q2d(vidStream->avg_frame_rate));

    AVPacket*   pkt   = av_packet_alloc();
    AVFrame*    frame = av_frame_alloc();
    AVFrame*    dst   = av_frame_alloc();
    SwsContext* sws   = nullptr;
    AVPixelFormat dstFmt = opts.grayscale ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_YUV420P;

    int frameCount = 0, saveCount = 0;
    std::string baseName = fs::path(videoPath).stem().string();

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != vidIdx) { av_packet_unref(pkt); continue; }
        if (avcodec_send_packet(dec, pkt) == 0) {
            while (avcodec_receive_frame(dec, frame) == 0) {
                frameCount++;
                bool doSave = (opts.ratio == 1) ? true : (frameCount % opts.ratio == 0);
                if (doSave) {
                    if (!sws) {
                        sws = sws_getContext(
                            frame->width, frame->height, (AVPixelFormat)frame->format,
                            frame->width, frame->height, dstFmt,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                        dst->format = dstFmt;
                        dst->width  = frame->width;
                        dst->height = frame->height;
                        av_frame_get_buffer(dst, 32);
                    }
                    sws_scale(sws, frame->data, frame->linesize, 0, frame->height,
                              dst->data, dst->linesize);

                    std::ostringstream name;
                    name << opts.outDir << "/" << baseName
                         << "_" << std::setw(6) << std::setfill('0') << saveCount << ".jpg";

                    AVFrame* toSave = frame;
                    SwsContext* gSws  = nullptr;
                    AVFrame*    gFrame = nullptr;
                    if (opts.grayscale) {
                        gSws = sws_getContext(
                            frame->width, frame->height, (AVPixelFormat)frame->format,
                            frame->width, frame->height, AV_PIX_FMT_GRAY8,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                        gFrame = av_frame_alloc();
                        gFrame->format = AV_PIX_FMT_GRAY8;
                        gFrame->width  = frame->width;
                        gFrame->height = frame->height;
                        av_frame_get_buffer(gFrame, 32);
                        sws_scale(gSws, frame->data, frame->linesize, 0, frame->height,
                                  gFrame->data, gFrame->linesize);
                        toSave = gFrame;
                    }

                    if (saveFrameAsJPEG(toSave, name.str())) saveCount++;
                    if (gFrame) { av_frame_free(&gFrame); }
                    if (gSws) sws_freeContext(gSws);
                }
                av_frame_unref(frame);
                if (progress && totalEst > 0)
                    progress((int)(100.0 * frameCount / totalEst), "프레임 추출 중...");
            }
        }
        av_packet_unref(pkt);
    }

    avcodec_send_packet(dec, nullptr);
    while (avcodec_receive_frame(dec, frame) == 0) av_frame_unref(frame);

    av_frame_free(&dst);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    if (sws) sws_freeContext(sws);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    if (progress) progress(100, "완료!");

    auto t1 = std::chrono::steady_clock::now();
    res.success     = true;
    res.totalFrames = frameCount;
    res.savedFrames = saveCount;
    res.elapsedSec  = std::chrono::duration<double>(t1 - t0).count();
    return res;
}

// ═══════════════════════════════════════════════════════════════════
// getVideoInfo
// ═══════════════════════════════════════════════════════════════════

struct VideoInfo {
    bool valid = false;
    std::string error;
    std::string format;
    int durationSec = 0;
    long long bitrate = 0;
    std::string videoCodec;
    int width = 0, height = 0;
    double fps = 0;
    int64_t nbFrames = 0;
    std::string audioCodec;
    int sampleRate = 0;
    int channels = 0;
};

inline VideoInfo getVideoInfo(const std::string& path) {
    VideoInfo info;
    AVFormatContext* fmt = nullptr;
    if (avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) < 0) {
        info.error = "파일을 열 수 없습니다."; return info;
    }
    avformat_find_stream_info(fmt, nullptr);
    info.valid  = true;
    info.format = fmt->iformat ? fmt->iformat->long_name : "?";
    if (fmt->duration > 0)  info.durationSec = (int)(fmt->duration / AV_TIME_BASE);
    if (fmt->bit_rate > 0)  info.bitrate = fmt->bit_rate;

    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        AVStream* st = fmt->streams[i];
        AVCodecParameters* cp = st->codecpar;
        const AVCodecDescriptor* desc = avcodec_descriptor_get(cp->codec_id);
        if (cp->codec_type == AVMEDIA_TYPE_VIDEO) {
            info.videoCodec = desc ? desc->name : "?";
            info.width  = cp->width;
            info.height = cp->height;
            if (st->avg_frame_rate.den > 0) info.fps = av_q2d(st->avg_frame_rate);
            info.nbFrames = st->nb_frames;
        } else if (cp->codec_type == AVMEDIA_TYPE_AUDIO) {
            info.audioCodec = desc ? desc->name : "?";
            info.sampleRate = cp->sample_rate;
            info.channels   = cp->channels;
        }
    }
    avformat_close_input(&fmt);
    return info;
}
