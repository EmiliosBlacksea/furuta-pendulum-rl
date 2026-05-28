"""
dashboard.py  –  Real-Time Standalone Test Dashboard
=====================================================
Connects to the Teensy over Serial in STANDALONE mode and shows
live pendulum angle and motor position.  Press SPACE to start / pause.

Usage:
    python scripts/dashboard.py
"""
import math
import sys

import pygame
import serial

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
SERIAL_PORT = "COM3"   # ← change to your port
BAUD_RATE   = 115200

ENCODER_CPR      = 2400.0
ENC_TICKS_TO_RADS = (2.0 * math.pi) / ENCODER_CPR
STEPS_TO_RADS    = (2.0 * math.pi) / 3200.0


def main():
    print(f"[*] Connecting to Teensy on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        ser.write(b"M:STANDALONE\n")
        ser.write(b"P\n")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return

    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Edge AI – Live Dashboard")

    font_title = pygame.font.SysFont("Arial",    42, bold=True)
    font_large = pygame.font.SysFont("Consolas", 32, bold=True)
    font_small = pygame.font.SysFont("Arial",    20)

    running        = True
    is_active      = False
    enc_offset     = None
    cal_samples    = []
    display_angle  = 0.0
    motor_deg      = 0.0

    print("[*] Waiting for calibration (leave pendulum hanging still)...")

    while running:
        # --- Read telemetry ---
        while ser.in_waiting:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line.startswith("TEL:"):
                continue

            parts = line[4:].split(",")
            if len(parts) != 2:
                continue

            try:
                motor_steps = int(parts[0])
                enc_ticks   = int(parts[1])
            except ValueError:
                continue

            # Calibration: collect 10 samples then compute offset
            if enc_offset is None:
                cal_samples.append(enc_ticks)
                if len(cal_samples) == 10:
                    enc_offset = sum(cal_samples) / 10
                    print(f"[OK] Calibrated. Offset = {enc_offset:.1f}")
                continue

            enc_rads   = -(enc_ticks - enc_offset) * ENC_TICKS_TO_RADS
            motor_rads = motor_steps * STEPS_TO_RADS

            raw_deg     = abs(math.degrees(enc_rads)) % 360.0
            display_angle = raw_deg if raw_deg <= 180.0 else 360.0 - raw_deg
            motor_deg   = math.degrees(motor_rads)

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_active = not is_active
                    ser.write(b"S\n" if is_active else b"P\n")
                    print("[*] START!" if is_active else "[*] PAUSED!")
                elif event.key == pygame.K_ESCAPE:
                    ser.write(b"P\n")
                    running = False

        # --- Draw ---
        if enc_offset is None:
            bg      = (30, 30, 35)
            status  = font_title.render("CALIBRATING...", True, (255, 200, 50))
        elif is_active:
            bg      = (10, 40, 10)
            status  = font_title.render("AI RUNNING",    True, (50, 255, 50))
        else:
            bg      = (40, 10, 10)
            status  = font_title.render("PAUSED",        True, (255, 100, 100))

        screen.fill(bg)
        screen.blit(status, status.get_rect(center=(300, 50)))
        pygame.draw.line(screen, (100, 100, 100), (50, 100), (550, 100), 2)

        angle_color = (50, 255, 50) if display_angle > 165 else (255, 150, 50)
        screen.blit(font_large.render(f"Pendulum : {display_angle:>5.1f}°", True, angle_color),     (40, 140))
        screen.blit(font_large.render(f"Motor    : {motor_deg:>5.1f}°",     True, (150, 200, 255)), (40, 200))

        pygame.draw.line(screen, (100, 100, 100), (50, 280), (550, 280), 2)
        screen.blit(font_small.render("[SPACE] Start / Pause",           True, (200, 200, 200)), (200, 300))
        screen.blit(font_small.render("[ESC]   Quit",                    True, (200, 200, 200)), (200, 330))

        pygame.display.flip()
        pygame.time.delay(20)

    ser.write(b"P\n")
    ser.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
