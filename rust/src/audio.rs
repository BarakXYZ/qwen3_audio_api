use crate::config::ResponseFormat;
use crate::error::ApiError;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::io::Cursor;

/// Encode f32 audio samples to the requested format.
pub fn encode_audio(
    samples: &[f32],
    sample_rate: u32,
    format: ResponseFormat,
) -> Result<Vec<u8>, ApiError> {
    match format {
        ResponseFormat::Wav => encode_wav(samples, sample_rate),
        ResponseFormat::Pcm => Ok(encode_pcm(samples)),
        ResponseFormat::Flac | ResponseFormat::Mp3 | ResponseFormat::Opus | ResponseFormat::Aac => {
            encode_with_ffmpeg_lib(samples, sample_rate, format)
        }
    }
}

/// Encode as 16-bit PCM WAV (mono).
fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, ApiError> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buffer = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buffer, spec)
            .map_err(|e| ApiError::internal(format!("WAV encode error: {e}")))?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_val = (clamped * 32767.0) as i16;
            writer
                .write_sample(i16_val)
                .map_err(|e| ApiError::internal(format!("WAV write error: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| ApiError::internal(format!("WAV finalize error: {e}")))?;
    }
    Ok(buffer.into_inner())
}

/// Encode as raw 16-bit signed little-endian PCM.
fn encode_pcm(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&i16_val.to_le_bytes());
    }
    bytes
}

/// Encode audio using the statically-linked ffmpeg library.
fn encode_with_ffmpeg_lib(
    samples: &[f32],
    sample_rate: u32,
    format: ResponseFormat,
) -> Result<Vec<u8>, ApiError> {
    // First encode to WAV in memory, then transcode via ffmpeg lib
    let wav_bytes = encode_wav(samples, sample_rate)?;

    // Write WAV to temp file (ffmpeg needs seekable input for some formats)
    let mut src_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| ApiError::internal(format!("Failed to create temp file: {e}")))?;
    {
        use std::io::Write;
        src_file
            .write_all(&wav_bytes)
            .map_err(|e| ApiError::internal(format!("Failed to write temp file: {e}")))?;
    }
    let src_path = src_file.path().to_string_lossy().to_string();

    let mut dst_file = tempfile::Builder::new()
        .suffix(format.extension())
        .tempfile()
        .map_err(|e| ApiError::internal(format!("Failed to create output temp file: {e}")))?;
    let dst_path = dst_file.path().to_string_lossy().to_string();

    transcode_audio(&src_path, &dst_path, format)?;

    use std::io::Read;
    let mut output = Vec::new();
    dst_file
        .read_to_end(&mut output)
        .map_err(|e| ApiError::internal(format!("Failed to read encoded output: {e}")))?;

    Ok(output)
}

/// Transcode an audio file from one format to another using the ffmpeg library.
fn transcode_audio(
    input_path: &str,
    output_path: &str,
    format: ResponseFormat,
) -> Result<(), ApiError> {
    ffmpeg_next::init()
        .map_err(|e| ApiError::internal(format!("Failed to initialize ffmpeg: {e}")))?;

    let mut ictx = ffmpeg_next::format::input(input_path)
        .map_err(|e| ApiError::internal(format!("Failed to open input: {e}")))?;

    let input_stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .ok_or_else(|| ApiError::internal("No audio stream found in input"))?;
    let input_stream_index = input_stream.index();

    let context_decoder =
        ffmpeg_next::codec::context::Context::from_parameters(input_stream.parameters())
            .map_err(|e| ApiError::internal(format!("Failed to create decoder context: {e}")))?;
    let mut decoder = context_decoder
        .decoder()
        .audio()
        .map_err(|e| ApiError::internal(format!("Failed to open decoder: {e}")))?;

    let codec_id = match format {
        ResponseFormat::Mp3 => ffmpeg_next::codec::Id::MP3,
        ResponseFormat::Opus => ffmpeg_next::codec::Id::OPUS,
        ResponseFormat::Aac => ffmpeg_next::codec::Id::AAC,
        ResponseFormat::Flac => ffmpeg_next::codec::Id::FLAC,
        _ => unreachable!(),
    };

    let encoder_codec = ffmpeg_next::encoder::find(codec_id)
        .ok_or_else(|| ApiError::internal(format!("Encoder not found for {codec_id:?}")))?;

    let mut octx = ffmpeg_next::format::output(output_path)
        .map_err(|e| ApiError::internal(format!("Failed to open output: {e}")))?;

    let mut output_stream = octx
        .add_stream(encoder_codec)
        .map_err(|e| ApiError::internal(format!("Failed to add output stream: {e}")))?;

    let context_encoder =
        ffmpeg_next::codec::context::Context::new_with_codec(encoder_codec);
    let mut encoder = context_encoder
        .encoder()
        .audio()
        .map_err(|e| ApiError::internal(format!("Failed to create encoder: {e}")))?;

    // Configure encoder
    let channel_layout = decoder.channel_layout();
    let channel_layout = if channel_layout.is_empty() {
        ffmpeg_next::ChannelLayout::MONO
    } else {
        channel_layout
    };

    encoder.set_rate(decoder.rate() as i32);
    encoder.set_channel_layout(channel_layout);
    let default_sample_fmt = ffmpeg_next::format::Sample::I16(
        ffmpeg_next::format::sample::Type::Packed,
    );
    let sample_format = encoder_codec
        .audio()
        .ok()
        .and_then(|a| a.formats())
        .and_then(|mut f| f.next())
        .unwrap_or(default_sample_fmt);
    encoder.set_format(sample_format);
    if codec_id == ffmpeg_next::codec::Id::OPUS {
        // Opus requires 48kHz
        encoder.set_rate(48000);
    }

    let mut encoder = encoder
        .open_as(encoder_codec)
        .map_err(|e| ApiError::internal(format!("Failed to open encoder: {e}")))?;

    output_stream.set_parameters(&encoder);

    octx.write_header()
        .map_err(|e| ApiError::internal(format!("Failed to write output header: {e}")))?;

    let output_stream_time_base = octx.stream(0).unwrap().time_base();

    // Set up resampler if needed
    let mut resampler = if decoder.format() != encoder.format()
        || decoder.rate() != encoder.rate()
        || decoder.channel_layout() != encoder.channel_layout()
    {
        Some(
            ffmpeg_next::software::resampling::Context::get(
                decoder.format(),
                decoder.channel_layout(),
                decoder.rate(),
                encoder.format(),
                encoder.channel_layout(),
                encoder.rate(),
            )
            .map_err(|e| ApiError::internal(format!("Failed to create resampler: {e}")))?,
        )
    } else {
        None
    };

    let mut decoded_frame = ffmpeg_next::frame::Audio::empty();

    // Process packets
    for (stream, packet) in ictx.packets() {
        if stream.index() != input_stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| ApiError::internal(format!("Decoder send_packet error: {e}")))?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            let frame_to_encode = if let Some(ref mut resampler) = resampler {
                let mut resampled = ffmpeg_next::frame::Audio::empty();
                resampler
                    .run(&decoded_frame, &mut resampled)
                    .map_err(|e| ApiError::internal(format!("Resampler error: {e}")))?;
                resampled
            } else {
                decoded_frame.clone()
            };

            encoder
                .send_frame(&frame_to_encode)
                .map_err(|e| ApiError::internal(format!("Encoder send_frame error: {e}")))?;

            receive_and_write_packets(&mut encoder, &mut octx, output_stream_time_base)?;
        }
    }

    // Flush decoder
    decoder
        .send_eof()
        .map_err(|e| ApiError::internal(format!("Decoder send_eof error: {e}")))?;
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let frame_to_encode = if let Some(ref mut resampler) = resampler {
            let mut resampled = ffmpeg_next::frame::Audio::empty();
            resampler
                .run(&decoded_frame, &mut resampled)
                .map_err(|e| ApiError::internal(format!("Resampler error: {e}")))?;
            resampled
        } else {
            decoded_frame.clone()
        };

        encoder
            .send_frame(&frame_to_encode)
            .map_err(|e| ApiError::internal(format!("Encoder send_frame error: {e}")))?;

        receive_and_write_packets(&mut encoder, &mut octx, output_stream_time_base)?;
    }

    // Flush encoder
    encoder
        .send_eof()
        .map_err(|e| ApiError::internal(format!("Encoder send_eof error: {e}")))?;
    receive_and_write_packets(&mut encoder, &mut octx, output_stream_time_base)?;

    octx.write_trailer()
        .map_err(|e| ApiError::internal(format!("Failed to write output trailer: {e}")))?;

    Ok(())
}

fn receive_and_write_packets(
    encoder: &mut ffmpeg_next::encoder::Audio,
    octx: &mut ffmpeg_next::format::context::Output,
    time_base: ffmpeg_next::Rational,
) -> Result<(), ApiError> {
    let mut encoded_packet = ffmpeg_next::Packet::empty();
    while encoder.receive_packet(&mut encoded_packet).is_ok() {
        encoded_packet.set_stream(0);
        encoded_packet.rescale_ts(encoder.time_base(), time_base);
        encoded_packet
            .write_interleaved(octx)
            .map_err(|e| ApiError::internal(format!("Failed to write packet: {e}")))?;
    }
    Ok(())
}

/// Apply speed adjustment by resampling via linear interpolation.
/// Speed > 1.0 = faster (shorter audio), speed < 1.0 = slower (longer audio).
pub fn apply_speed(samples: &[f32], speed: f32) -> Vec<f32> {
    if (speed - 1.0).abs() < f32::EPSILON || samples.is_empty() {
        return samples.to_vec();
    }

    let new_length = (samples.len() as f64 / speed as f64) as usize;
    if new_length == 0 {
        return samples.to_vec();
    }

    let mut output = Vec::with_capacity(new_length);
    for i in 0..new_length {
        let src_pos = i as f64 * speed as f64;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;
        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }
    output
}

/// Convert audio bytes to WAV (mono, target_rate Hz, s16) using the ffmpeg library.
/// Returns the WAV file bytes.
pub fn convert_audio_to_wav_bytes(
    audio_bytes: &[u8],
    suffix: &str,
    target_rate: u32,
) -> Result<Vec<u8>, ApiError> {
    // Write source audio to a temp file
    let mut src_file = tempfile::Builder::new()
        .suffix(suffix)
        .tempfile()
        .map_err(|e| ApiError::internal(format!("Failed to create temp file: {e}")))?;
    {
        use std::io::Write;
        src_file
            .write_all(audio_bytes)
            .map_err(|e| ApiError::internal(format!("Failed to write temp file: {e}")))?;
    }
    let src_path = src_file.path().to_string_lossy().to_string();

    let mut dst_file = tempfile::Builder::new()
        .suffix(".wav")
        .tempfile()
        .map_err(|e| ApiError::internal(format!("Failed to create WAV temp file: {e}")))?;
    let dst_path = dst_file.path().to_string_lossy().to_string();

    transcode_to_wav(&src_path, &dst_path, target_rate)?;

    use std::io::Read;
    let mut output = Vec::new();
    dst_file
        .read_to_end(&mut output)
        .map_err(|e| ApiError::internal(format!("Failed to read WAV output: {e}")))?;

    Ok(output)
}

/// Transcode audio to mono WAV at the given sample rate using the ffmpeg library.
fn transcode_to_wav(
    input_path: &str,
    output_path: &str,
    target_rate: u32,
) -> Result<(), ApiError> {
    ffmpeg_next::init()
        .map_err(|e| ApiError::internal(format!("Failed to initialize ffmpeg: {e}")))?;

    let mut ictx = ffmpeg_next::format::input(input_path)
        .map_err(|e| ApiError::internal(format!("Failed to open input: {e}")))?;

    let input_stream = ictx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .ok_or_else(|| ApiError::internal("No audio stream found in input"))?;
    let input_stream_index = input_stream.index();

    let context_decoder =
        ffmpeg_next::codec::context::Context::from_parameters(input_stream.parameters())
            .map_err(|e| ApiError::internal(format!("Failed to create decoder context: {e}")))?;
    let mut decoder = context_decoder
        .decoder()
        .audio()
        .map_err(|e| ApiError::internal(format!("Failed to open decoder: {e}")))?;

    let encoder_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::PCM_S16LE)
        .ok_or_else(|| ApiError::internal("PCM S16LE encoder not found"))?;

    let mut octx = ffmpeg_next::format::output(output_path)
        .map_err(|e| ApiError::internal(format!("Failed to open output: {e}")))?;

    let mut output_stream = octx
        .add_stream(encoder_codec)
        .map_err(|e| ApiError::internal(format!("Failed to add output stream: {e}")))?;

    let context_encoder = ffmpeg_next::codec::context::Context::new_with_codec(encoder_codec);
    let mut encoder = context_encoder
        .encoder()
        .audio()
        .map_err(|e| ApiError::internal(format!("Failed to create encoder: {e}")))?;

    encoder.set_rate(target_rate as i32);
    encoder.set_channel_layout(ffmpeg_next::ChannelLayout::MONO);
    encoder.set_format(ffmpeg_next::format::Sample::I16(
        ffmpeg_next::format::sample::Type::Packed,
    ));

    let mut encoder = encoder
        .open_as(encoder_codec)
        .map_err(|e| ApiError::internal(format!("Failed to open encoder: {e}")))?;

    output_stream.set_parameters(&encoder);

    octx.write_header()
        .map_err(|e| ApiError::internal(format!("Failed to write output header: {e}")))?;

    let output_stream_time_base = octx.stream(0).unwrap().time_base();

    // Set up resampler (decode format -> s16 mono at target_rate)
    let dec_channel_layout = decoder.channel_layout();
    let dec_channel_layout = if dec_channel_layout.is_empty() {
        ffmpeg_next::ChannelLayout::MONO
    } else {
        dec_channel_layout
    };

    let mut resampler = ffmpeg_next::software::resampling::Context::get(
        decoder.format(),
        dec_channel_layout,
        decoder.rate(),
        encoder.format(),
        encoder.channel_layout(),
        encoder.rate(),
    )
    .map_err(|e| ApiError::internal(format!("Failed to create resampler: {e}")))?;

    let mut decoded_frame = ffmpeg_next::frame::Audio::empty();

    for (stream, packet) in ictx.packets() {
        if stream.index() != input_stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| ApiError::internal(format!("Decoder send_packet error: {e}")))?;

        while decoder.receive_frame(&mut decoded_frame).is_ok() {
            let mut resampled = ffmpeg_next::frame::Audio::empty();
            resampler
                .run(&decoded_frame, &mut resampled)
                .map_err(|e| ApiError::internal(format!("Resampler error: {e}")))?;

            encoder
                .send_frame(&resampled)
                .map_err(|e| ApiError::internal(format!("Encoder send_frame error: {e}")))?;

            receive_and_write_packets(&mut encoder, &mut octx, output_stream_time_base)?;
        }
    }

    // Flush decoder
    decoder
        .send_eof()
        .map_err(|e| ApiError::internal(format!("Decoder send_eof error: {e}")))?;
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let mut resampled = ffmpeg_next::frame::Audio::empty();
        resampler
            .run(&decoded_frame, &mut resampled)
            .map_err(|e| ApiError::internal(format!("Resampler error: {e}")))?;

        encoder
            .send_frame(&resampled)
            .map_err(|e| ApiError::internal(format!("Encoder send_frame error: {e}")))?;

        receive_and_write_packets(&mut encoder, &mut octx, output_stream_time_base)?;
    }

    // Flush encoder
    encoder
        .send_eof()
        .map_err(|e| ApiError::internal(format!("Encoder send_eof error: {e}")))?;
    receive_and_write_packets(&mut encoder, &mut octx, output_stream_time_base)?;

    octx.write_trailer()
        .map_err(|e| ApiError::internal(format!("Failed to write output trailer: {e}")))?;

    Ok(())
}
