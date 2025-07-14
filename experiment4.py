#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Muse neurofeedback + calibration

Note for NPS: current_alpha is the alpha you need (unless we do delta)
Calibration so far doesn't really work, so to be faster with your check, you can change 

bip_times = [0, 60, 120]
to 
bip_times = [0, 2, 4]

and

elapsed >= 123 and screen_state != "done_calibration":
to
elapsed >= 5 and screen_state != "done_calibration":

"""

import pygame
import numpy as np
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, local_clock
from scipy.signal import firwin, lfilter, lfilter_zi, welch
import time
import sounddevice as sd
import pandas as pd
import os

import serial
ser = serial.Serial('/dev/rfcomm0', 115200)




# === Play tone ===
def play_clean_bip_and_stream(freq=750, duration=0.3, amplitude=0.1, fs=44100):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    fade_len = int(0.02 * fs)
    envelope = np.ones_like(t)
    envelope[:fade_len] = np.sin(np.linspace(0, np.pi/2, fade_len))**2
    envelope[-fade_len:] = np.cos(np.linspace(0, np.pi/2, fade_len))**2
    tone = amplitude * np.sin(2 * np.pi * freq * t) * envelope
    tone = tone.astype(np.float32)

    def callback(indata, outdata, frames, time_info, status):
        outdata[:] = tone[:frames].reshape(-1, 1)

    with sd.Stream(samplerate=fs, channels=1, dtype='float32', callback=callback):
        sd.sleep(int(duration * 1000))

# === Compute band power ===
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

# === Parameters ===
fs = 256
n_channels = 4
channel_names = ['TP9', 'AF7', 'AF8', 'TP10']
band_defs = {"Delta": (1, 4), "Theta": (4, 8), "Alpha": (8, 12), "Beta": (13, 30)}
saturation_thresh = 50

bf = firwin(64, np.array([1, 40]) / (fs / 2.), width=0.05, pass_zero=False)
af = [1.0]
zi_init = lfilter_zi(bf, af)

pygame.init()
screen = pygame.display.set_mode((1200, 720))
pygame.display.set_caption("EEG BCI Interface")
font = pygame.font.SysFont("Arial", 24)
clock = pygame.time.Clock()
pygame.mixer.init()

print("🔍 Searching for EEG stream...")
eeg_streams = resolve_byprop('type', 'EEG', timeout=5)
eeg_inlet = StreamInlet(eeg_streams[0])
marker_info = StreamInfo('Markers', 'Markers', 1, 0, 'string')
marker_outlet = StreamOutlet(marker_info)

data_f = np.zeros((int(fs * 4.0), n_channels))
filt_state = np.tile(zi_init, (n_channels, 1)).T
stds = [0] * n_channels
statuses = ["Unknown"] * n_channels

start_btn = pygame.Rect(1000, 100, 150, 50)
done_btn = pygame.Rect(500, 650, 150, 50)

screen_state = "waiting_record"
calib_start_time = None
bip_times = [0, 60, 120]
bip_played = [False] * 3
log_calib = []
eeg_during_calib = []
alpha_summary = None
feedback_buffer = []
last_feedback_time = time.time()
alpha_window = []
current_alpha = None

calib_time  = 23 #123
open_time = 10 #60
closed_time = open_time * 2

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
            if len(feedback_buffer) > fs * 2:
                feedback_buffer = feedback_buffer[-fs * 2:]
            if time.time() - last_feedback_time >= 1.0 and len(feedback_buffer) >= fs * 2:
                feedback_array = np.array(feedback_buffer)
                all_channels = feedback_array[:, :]
                if np.std(all_channels) > 5:
                    alpha_power = compute_band_power(all_channels, band_defs["Alpha"], fs).mean()
                    alpha_window.append(alpha_power)
                    if len(alpha_window) > 5:
                        alpha_window.pop(0)
                    current_alpha = np.mean(alpha_window)
                last_feedback_time = time.time()

        stds = np.std(data_f, axis=0, ddof=1)
        means = np.mean(data_f, axis=0)
        statuses = [
            "Disconnected" if std == 0 else
            "Noisy" if std > 30 else
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
            "Start calibration",
            "When you hear a tone, open your eyes",
            "Close them when you hear another tone",
            "Fix the cross until the third tone"]):
            screen.blit(font.render(line, True, (255, 255, 255)), (300, 200 + i * 40))
        pygame.draw.rect(screen, (0, 200, 100), start_btn)
        screen.blit(font.render("Start Calibration", True, (255, 255, 255)), (start_btn.x + 10, start_btn.y + 10))

    elif screen_state == "calibrating":
        elapsed = time.time() - calib_start_time
        if elapsed < open_time:
            screen.blit(font.render("Open your eyes", True, (255, 255, 255)), (500, 340))
        elif elapsed < closed_time:
            screen.blit(font.render("Close your eyes", True, (255, 255, 255)), (500, 340))
        else:
            pygame.draw.line(screen, (255, 255, 255), (600, 340), (600, 380), 4)
            pygame.draw.line(screen, (255, 255, 255), (580, 360), (620, 360), 4)

        for i, t in enumerate(bip_times):
            if elapsed >= t and not bip_played[i]:
                ts = local_clock()
                play_clean_bip_and_stream(freq=2000, duration=0.2, amplitude=0.1)
                marker_outlet.push_sample([f"calib_bip_{i+1}"], timestamp=ts)
                log_calib.append({"bip": f"calib_bip_{i+1}", "timestamp": ts})
                bip_played[i] = True

        if elapsed >= calib_time and screen_state != "done_calibration":
            screen_state = "done_calibration"
            if not os.path.exists("logs"):
                os.makedirs("logs")
            pd.DataFrame(log_calib).to_csv("logs/calibration_bips.csv", index=False)
            eeg_array = np.array(eeg_during_calib)
            if eeg_array.shape[0] >= fs * 120:
                eyes_open = eeg_array[fs*0:fs*60, :]
                eyes_closed = eeg_array[fs*60:fs*120, :]
                alpha_closed = compute_band_power(eyes_closed, band_defs["Alpha"], fs)
                alpha_open = compute_band_power(eyes_open, band_defs["Alpha"], fs)
                alpha_summary = {
                    "closed": alpha_closed.mean(),
                    "open": alpha_open.mean(),
                    "diff": (alpha_closed - alpha_open).mean()
                }
                pd.DataFrame([alpha_summary]).to_csv("logs/alpha_calibration.csv", index=False)

    elif screen_state == "done_calibration":
        screen.blit(font.render("Calibration terminée.", True, (255, 255, 255)), (450, 250))
        if alpha_summary:
            screen.blit(font.render(f"Alpha yeux fermés : {alpha_summary['closed']:.2f}", True, (255, 255, 255)), (400, 300))
            screen.blit(font.render(f"Alpha yeux ouverts : {alpha_summary['open']:.2f}", True, (255, 255, 255)), (400, 340))
            screen.blit(font.render(f"Différence Alpha   : {alpha_summary['diff']:.2f}", True, (255, 255, 255)), (400, 380))
        pygame.draw.rect(screen, (0, 255, 150), done_btn)
        screen.blit(font.render("Next", True, (0, 0, 0)), (done_btn.x + 45, done_btn.y + 10))

    elif screen_state == "feedback":
        screen.blit(font.render("Neurofeedback Alpha", True, (255, 255, 255)), (450, 100))
        if current_alpha is not None:
            screen.blit(font.render(f"Alpha (All channels) = {current_alpha:.2f}", True, (255, 255, 255)), (420, 200))

            print(f'writing to serial port, {current_alpha}', flush=True)

            if current_alpha < 20.:
                ser.write("kbk".encode('utf-8'))
                ser.flush()
                time.sleep(1)
            elif current_alpha > 80:
                ser.write('kwkF'.encode('utf-8'))
                ser.flush()
                time.sleep(1)
            else:
                ser.write('kbalance'.encode('utf-8'))
                ser.flush()
                time.sleep(1)
            # Use calibration-based min and max for normalization
            if alpha_summary:
                alpha_min = alpha_summary['open']
                alpha_max = alpha_summary['closed']
                alpha_range = alpha_max - alpha_min

                # Avoid division by zero
                if alpha_range > 0:
                    norm_alpha = (current_alpha - alpha_min) / alpha_range

                    radius = int(np.clip(30 + norm_alpha * (150 - 30), 30, 150))
                else:
                    radius = 30
            else:
                radius = 30  # fallback if calibration not available

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
