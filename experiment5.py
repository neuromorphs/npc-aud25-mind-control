#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Muse neurofeedback + calibration with alpha/(theta+beta) ratio

Note: current_alpha now refers to the ratio alpha/(theta+beta).
"""

import pygame
import numpy as np
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, local_clock
from scipy.signal import firwin, lfilter, lfilter_zi, welch
import time
import sounddevice as sd
import pandas as pd
import os
from pydub import AudioSegment
from scipy.signal import butter, filtfilt

'''
# === Play tone ===
def play_audio_file(filename):
    sound = AudioSegment.from_file(filename)  # handles mp3, wav, etc.
    samples = np.array(sound.get_array_of_samples()).astype(np.float32)
    if sound.channels == 2:
        samples = samples.reshape((-1, 2))
    else:
        samples = samples.reshape((-1, 1))
    samples /= np.iinfo(sound.array_type).max  # normalize to -1.0 to 1.0
    sd.play(samples, samplerate=sound.frame_rate)
    sd.wait()
'''
import threading

def play_audio_file(filename):
    def _play():
        sound = AudioSegment.from_file(filename)
        samples = np.array(sound.get_array_of_samples()).astype(np.float32)
        if sound.channels == 2:
            samples = samples.reshape((-1, 2))
        else:
            samples = samples.reshape((-1, 1))
        samples /= np.iinfo(sound.array_type).max
        sd.play(samples, samplerate=sound.frame_rate)
    
    threading.Thread(target=_play, daemon=True).start()


# === Compute band power ===
'''
def compute_band_power(epoch, band, fs):
    epoch = np.atleast_2d(epoch)
    if epoch.shape[0] < epoch.shape[1]:
        epoch = epoch.T

    fmin, fmax = band
    psd, freqs = welch(epoch, fs=fs, nperseg=fs*2, axis=0)
    freqs = np.squeeze(freqs)
    if freqs.ndim > 1:
        freqs = freqs[:, 0]

    idx_band = (freqs >= fmin) & (freqs <= fmax)

    if psd.ndim == 1:
        return np.sum(psd[idx_band])
    else:
        return np.sum(psd[idx_band, :], axis=0)
'''
def compute_band_power(epoch, band, fs, order=4):
    epoch = np.atleast_2d(epoch)
    if epoch.shape[0] < epoch.shape[1]:
        epoch = epoch.T  # Ensure shape is (samples, channels)
    
    nyq = 0.5 * fs
    low, high = band[0] / nyq, band[1] / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, epoch, axis=0)

    power = np.mean(filtered ** 2, axis=0)  # mean power per channel
    return power

# === Parameters ===
fs = 256
n_channels = 4
channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
band_defs = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
saturation_thresh = 50
calibration_duration = 15
buffer_size = 2

# Filter : do we really need it?
bf = firwin(64, np.array([1, 40]) / (fs / 2.), width=0.05, pass_zero=False)
af = [1.0]
zi_init = lfilter_zi(bf, af)

#initialize the interface
pygame.init()
screen = pygame.display.set_mode((1200, 720))
pygame.display.set_caption("EEG BCI Interface")
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()
pygame.mixer.init()

# Get EEG stream
eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
eeg_inlet = StreamInlet(eeg_streams[0])
info = eeg_inlet.info()
desc = info.desc()

# Get channel labels
channels = desc.child("channels").child("channel")
ch_names = []
for _ in range(info.channel_count()):
    ch_names.append(channels.child_value("label"))
    channels = channels.next_sibling()

# Keep only EEG channels, exclude AUX
exclude_names = ['Right AUX', 'Aux Right', 'AUX', 'Accelerometer']
valid_indices = [i for i, name in enumerate(ch_names) if name not in exclude_names]
channel_names = [ch_names[i] for i in valid_indices]
n_channels = len(valid_indices)

print("✅ Channels used:", channel_names)

# Marker stream for triggers
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string')
marker_outlet = StreamOutlet(marker_info)

#For channel initation
data_f = np.zeros((int(fs * 4.0), n_channels))
filt_state = np.tile(zi_init, (n_channels, 1)).T
stds = [0] * n_channels
statuses = ["Unknown"] * n_channels

start_btn = pygame.Rect(1000, 100, 150, 50)
done_btn = pygame.Rect(500, 650, 150, 50)

screen_state = "waiting_record"
calib_start_time = None
bip_times = [0, calibration_duration, calibration_duration*2]
bip_played = [False] * 3
log_calib = []
eeg_during_calib = []
alpha_summary = None
feedback_buffer = []
last_feedback_time = time.time()
alpha_window = []
current_alpha = None

running = True
while running:
    screen.fill((0, 0, 0))

    samples, _ = eeg_inlet.pull_chunk(timeout=0.0, max_samples=fs)
    if samples:
        samples = np.array(samples)
        filt_chunk, filt_state = lfilter(bf, af, samples[:, :n_channels], axis=0, zi=filt_state)
        data_f = np.vstack([data_f, filt_chunk])[-int(fs * 4.0):]
        if screen_state == "calibrating":
            eeg_during_calib.extend(filt_chunk)
        elif screen_state == "feedback":
            feedback_buffer.extend(filt_chunk.tolist())
            if len(feedback_buffer) > fs * buffer_size:
                feedback_buffer = feedback_buffer[-fs * buffer_size:]

            if time.time() - last_feedback_time >= 1.0 and len(feedback_buffer) >= fs * buffer_size:

                feedback_array = np.array(feedback_buffer)
                all_channels = feedback_array[:, :]
                if np.std(all_channels) > 5:
                    alpha = compute_band_power(all_channels, band_defs["Alpha"], fs)
                    theta = compute_band_power(all_channels, band_defs["Theta"], fs)
                    beta  = compute_band_power(all_channels, band_defs["Beta"], fs)
                    ratio = alpha / (theta * beta)
                    alpha_window.append(ratio.mean())
                    if len(alpha_window) > 5:
                        alpha_window.pop(0)
                    current_alpha = np.mean(alpha_window)
                last_feedback_time = time.time()

        stds = np.std(data_f, axis=0, ddof=1)
        means = np.mean(data_f, axis=0)
        statuses = [
            "Disconnected" if std == 0 else
            "Noisy" if std > 35 else
            "Saturated" if abs(mean) >= saturation_thresh else
            "OK"
            for std, mean in zip(stds, means)
        ]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if screen_state == "waiting_record" and start_btn.collidepoint(event.pos):
                screen_state = "waiting_calibration"
            elif screen_state == "waiting_calibration" and start_btn.collidepoint(event.pos):
                screen_state = "calibrating"
                calib_start_time = time.time()
            elif screen_state == "done_calibration" and done_btn.collidepoint(event.pos):
                screen_state = "feedback"
                feedback_buffer = []
                alpha_window = []
                last_feedback_time = time.time()

    if screen_state == "waiting_record":
        screen.blit(font.render("Start to record EEG signal", True, (255, 255, 255)), (400, 300))
        pygame.draw.rect(screen, (0, 100, 255), start_btn)
        screen.blit(font.render("Done", True, (255, 255, 255)), (start_btn.x + 45, start_btn.y + 10))

    elif screen_state == "waiting_calibration":
        for i, line in enumerate([
            "When you hear a tone followed by upbeat music, please look at the light",
            "When you hear a gong followed by calm music, please close your eyes and try to relax",
            "Open your eyes when the music stops"]):
            screen.blit(font.render(line, True, (255, 255, 255)), (300, 200 + i * 40))

        pygame.draw.rect(screen, (0, 200, 100), start_btn)
        screen.blit(font.render("Start Calibration", True, (255, 255, 255)), (start_btn.x + 10, start_btn.y + 10))

    elif screen_state == "calibrating":
        elapsed = time.time() - calib_start_time
        if elapsed < calibration_duration:
            screen.blit(font.render("Look at the light", True, (255, 255, 255)), (450, 300))
            screen.blit(font.render("+", True, (255, 255, 255)), (600, 360))
            if not bip_played[0]:
                play_audio_file("eyetiger.mp3")
                bip_played[0] = True
        elif elapsed < calibration_duration * 2:
            screen.blit(font.render("Close your eyes", True, (255, 255, 255)), (480, 300))
            screen.blit(font.render("+", True, (255, 255, 255)), (600, 360))
            if not bip_played[1]:
                play_audio_file("relax.mp3")
                bip_played[1] = True


        else:
            pygame.draw.line(screen, (255, 255, 255), (600, 340), (600, 380), 4)
            pygame.draw.line(screen, (255, 255, 255), (580, 360), (620, 360), 4)
        
        for i, t in enumerate(bip_times):
            if elapsed >= t and not bip_played[i]:
                ts = local_clock()
                #play_audio_file("1gong.mp3")
                marker_outlet.push_sample([f"calib_bip_{i+1}"], timestamp=ts)
                log_calib.append({"bip": f"calib_bip_{i+1}", "timestamp": ts})
                bip_played[i] = True
        
        if elapsed >= 33 and screen_state != "done_calibration":
            screen_state = "done_calibration"
            if not os.path.exists("logs"):
                os.makedirs("logs")
            pd.DataFrame(log_calib).to_csv("logs/calibration_bips.csv", index=False)
            eeg_array = np.array(eeg_during_calib)
            if eeg_array.shape[0] >= fs * calibration_duration:
                eyes_open = eeg_array[fs*10:fs*calibration_duration, :]
                eyes_closed = eeg_array[fs*10+calibration_duration:fs*calibration_duration*2, :]
                alpha_closed = compute_band_power(eyes_closed, band_defs["Alpha"], fs)
                theta_closed = compute_band_power(eyes_closed, band_defs["Theta"], fs)
                beta_closed  = compute_band_power(eyes_closed, band_defs["Beta"], fs)
                ratio_closed = alpha_closed / (theta_closed + beta_closed)*100

                alpha_open = compute_band_power(eyes_open, band_defs["Alpha"], fs)
                theta_open = compute_band_power(eyes_open, band_defs["Theta"], fs)
                beta_open  = compute_band_power(eyes_open, band_defs["Beta"], fs)
                ratio_open = alpha_open / (theta_open + beta_open)*100

                alpha_summary = {
                    "closed": ratio_closed.mean(),
                    "open": ratio_open.mean(),
                    "diff": (ratio_closed - ratio_open).mean()
                }
                pd.DataFrame([alpha_summary]).to_csv("logs/alpha_calibration.csv", index=False)
                calibration_failed = False

                if alpha_summary["closed"] <= alpha_summary["open"]:
                    print("⚠️ Calibration failed: Eyes closed alpha is not greater than eyes open.")
                    calibration_failed = True
                else:
                    calibration_failed = False

                # Calcul du seuil selon le cas
                if calibration_failed:
                    threshold = alpha_summary["open"] - 0.5 * (alpha_summary["open"] - alpha_summary["closed"])
                else:
                    threshold = (alpha_summary["open"] + alpha_summary["closed"]) / 2

                alpha_summary["threshold"] = threshold
                print(f"✅ Alpha threshold set to: {alpha_summary['threshold']:.2f}")


    elif screen_state == "done_calibration":
        screen.blit(font.render("Calibration done.", True, (255, 255, 255)), (450, 250))
        if alpha_summary:
            screen.blit(font.render(f"Calm / Meditative : {alpha_summary['closed']:.2f}", True, (255, 255, 255)), (380, 300))
            screen.blit(font.render(f"Not calm : {alpha_summary['open']:.2f}", True, (255, 255, 255)), (380, 340))
            screen.blit(font.render(f"Difference : {alpha_summary['diff']:.2f}", True, (255, 255, 255)), (380, 380))
        pygame.draw.rect(screen, (0, 255, 150), done_btn)
        screen.blit(font.render("Next", True, (0, 0, 0)), (done_btn.x + 45, done_btn.y + 10))

    elif screen_state == "feedback":
        screen.blit(font.render("Neurofeedback ", True, (255, 255, 255)), (420, 100))

        if alpha_summary:
            alpha_min = alpha_summary['open']
            alpha_max = alpha_summary['closed']
            alpha_range = alpha_max - alpha_min

            if alpha_range > 0:
                norm_alpha = (current_alpha - alpha_min) / alpha_range
                radius = int(np.clip(30 + norm_alpha * (150 - 30), 30, 150))
            else:
                radius = 30
        else:
            radius = 30

        pygame.draw.circle(screen, (0, 100, 255), (600, 400), radius)

    for i in range(n_channels):
        color = {
            "Disconnected": (150, 150, 150),
            "Noisy": (255, 255, 0),
            "Saturated": (200, 0, 0),
            "OK": (0, 200, 0)
        }.get(statuses[i], (100, 100, 100))
        pygame.draw.circle(screen, color, (850, 20 + i * 60), 15)
        label = font.render(f"{channel_names[i]} {stds[i]:.1f}", True, (255, 255, 255))
        screen.blit(label, (875, 10 + i * 60))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
