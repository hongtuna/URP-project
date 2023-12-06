from pynput.mouse import Controller
import time
import os
import numpy as np
import RPi.GPIO as GPIO
import hbcvt
import numpy as np
import pyautogui
import atexit
import signal
import sys
import Adafruit_ADS1x15

# Cell position: (1,1), (2,1), (3,1), (1,2), (2,2), (3,2)
GPIOs = [17, 27, 22, 18, 23, 24]
GPIO_SIGNAL = [GPIO.LOW, GPIO.HIGH]

def read_text(filepath):
    f = open(filepath, 'r')
    text_file = f.read()

    print('Reading texts is done!\n')

    return text_file



def txt_2_braille(txt):

    list_braille = [ [0,0,0,0,0,0] ]
    list_txt = [ '' ]
    raw_braille = list(hbcvt.h2b.text(list(txt)))
    for i in range(len(raw_braille)):
        char = raw_braille[i][0]
        brailles = raw_braille[i][1][0][1]
        
        for braille in brailles:
            list_braille.append(braille)
            list_txt.append(char)

    list_braille.append([0,0,0,0,0,0])
    list_txt.append( '' )
    print('Brailles are converted!\n')

    return list_txt, list_braille


@atexit.register
def init_solenoids():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for gpio in GPIOs:
        GPIO.setup(gpio, GPIO.OUT)
        GPIO.output(gpio, GPIO_SIGNAL[0])

    print('All solenoids are initiated!\n')

def terminate(signal, frame):
    init_solenoids()
    sys.exit(0)

def move(list_braille, list_txt, cur_index, verbose=0):
    for i in range(6):
        GPIO.output(GPIOs[i], GPIO_SIGNAL[list_braille[cur_index][i]])
    if verbose == 1:
        print('#{}/{} {} {}'.format(cur_index, len(list_braille), list_txt[cur_index], list_braille[cur_index]))


# 일정 Interval로 자동으로 한글자씩 넘김
def autoread(textpath, interval = 1, verbose=0):

    init_solenoids()

    txt = read_text(textpath)
    list_txt, list_braille = txt_2_braille(txt)

    for i, braille in enumerate(list_braille):        
        move(list_braille, list_txt, i, verbose) 
        time.sleep(interval)


def clipping_index(max_len, index):
    if max_len-1 < index:
        index = max_len-1
    elif index < 0:
        index = 0
    
    return index
    
        

def handread(is_vector, textpath, joystick_tolerance = 0.2, move_tolerance =0.25, verbose=0):

    GAIN = 1
    adc = Adafruit_ADS1x15.ADS1015()
    
    init_solenoids()

    txt = read_text(textpath)
    list_txt, list_braille = txt_2_braille(txt)
    screen_width, screen_height = pyautogui.size()
    
    cur_index = -1
    prev_move_time = 0.0
    prev_joystick_time = 0.0
    
    while(True):
        
        Controller().position = (screen_width/2, 0)
        cur_x = ( Controller().position[0] - screen_width/2 ) / float(screen_width) # -0.5 ~ 0.5
        cur_time = time.time()

        # 조이스틱 처리
        joystep = 0
        joystick_y = adc.read_adc(1, gain=GAIN)
        if (joystick_y <= 200):
            joystep = 1
        elif(joystick_y >= 1000):
            joystep = -1
        
        if joystep != 0 and abs(prev_joystick_time - cur_time) > joystick_tolerance:
            cur_index = clipping_index(len(list_braille), cur_index + joystep)
            move(list_braille, list_txt, cur_index, verbose)
            prev_joystick_time = cur_time
       
        # 마우스 처리 
        if is_vector == True and cur_x > 0.01 and abs(prev_move_time - cur_time) > move_tolerance:
            cur_index = clipping_index(len(list_braille), cur_index+1)
            move(list_braille, list_txt, cur_index, verbose)
            prev_move_time = cur_time

######################################################3

PRESET = {'LOW':0.07, 'MID':0.3, 'HIGH':0.5}

# Ctrl + C 로 중간에 종료시 초기화 하고 종료
signal.signal(signal.SIGINT, terminate)

TXT_FILE_PATH = "story.txt"

# autoread(TXT_FILE_PATH, verbose=1)

# tolerance 작을 수록 빨리 점자 나옴 (숙련자용)

# Vectpr 모드
#handread(True, TXT_FILE_PATH, move_tolerance=PRESET['LOW'], verbose=1)
# Scalar 모드
handread(False, TXT_FILE_PATH, move_tolerance=PRESET['LOW'], verbose=1)

## 조이스틱 앞, 뒤 신호를 이용해 Tolerance를 조절하도록 하면 어떨까? ㅋ
