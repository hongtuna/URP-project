import string
import argparse
import keyboard

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import serial
from serial import Serial
from sys import version_info
from time import sleep

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

import hbcvt

PY2 = version_info[0] == 2 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        #print(raw_braille)
        char = raw_braille[i][0]
        brailles = raw_braille[i][1][0][1]
        
        for braille in brailles:
            list_braille.append(braille)
            list_txt.append(char)

    list_braille.append([0,0,0,0,0,0])
    list_txt.append( '' )
    print('Brailles are converted!\n')

    return list_txt, list_braille
                
def clipping_index(max_len, index):
    if max_len-1 < index:
        index = max_len-1
    elif index < 0:
        index = 0
                
    return index

def motor_move(list_braille, list_txt, cur_index):
    for i in range(0,1):
        if(list_braille[cur_index][i] == 1):
            servo.setTarget(i,1500*4)
            print('0번')
        else:
            servo.setTarget(i,1650*4)
    for i in range(1,2):
        if(list_braille[cur_index][i] == 1):
            servo.setTarget(i,1500*4)
            print('1번')
        else:
            servo.setTarget(i,1650*4)
    for i in range(2,3):
        if(list_braille[cur_index][i] == 1):
            servo.setTarget(i,1400*4)
            print('2번')
        else:
            servo.setTarget(i,1700*4)


    for i in range(3,4):
        if(list_braille[cur_index][i] == 1):
            servo.setTarget(i,1650*4)
            print('3번')
        else:
            servo.setTarget(i,1500*4)
    for i in range(4,5):
        if(list_braille[cur_index][i] == 1):
            servo.setTarget(i,1650*4)
            print('4번')
        else:
            servo.setTarget(i,1500*4)

    for i in range(5,6):
        if(list_braille[cur_index][i] == 1):
            servo.setTarget(i,1700*4)
            print('5번')
        else:
            servo.setTarget(i,1500*4)
    print('#{}/{} {} {}'.format(cur_index, len(list_braille), list_txt[cur_index], list_braille[cur_index]))


class Controller:
    # When connected via USB, the Maestro creates two virtual serial ports
    # /dev/ttyACM0 for commands and /dev/ttyACM1 for communications.
    # Be sure the Maestro is configured for "USB Dual Port" serial mode.
    # "USB Chained Mode" may work as well, but hasn't been tested.
    #
    # Pololu protocol allows for multiple Maestros to be connected to a single
    # serial port. Each connected device is then indexed by number.
    # This device number defaults to 0x0C (or 12 in decimal), which this module
    # assumes.  If two or more controllers are connected to different serial
    # ports, or you are using a Windows OS, you can provide the tty port.  For
    # example, '/dev/ttyACM2' or for Windows, something like 'COM3'.
    

    def __init__(self,ttyStr='/dev/ttyACM0',device=0x0c):
        # Open the command port
        self.usb = serial.Serial(ttyStr)
        # Command lead-in and device number are sent for each Pololu serial command.
        self.PololuCmd = chr(0xaa) + chr(device)
        # Track target position for each servo. The function isMoving() will
        # use the Target vs Current servo position to determine if movement is
        # occuring.  Upto 24 servos on a Maestro, (0-23). Targets start at 0.
        self.Targets = [0] * 24
        # Servo minimum and maximum targets can be restricted to protect components.
        self.Mins = [0] * 24
        self.Maxs = [0] * 24
        
    # Cleanup by closing USB serial port
    def close(self):
        self.usb.close()

    # Send a Pololu command out the serial port
    def sendCmd(self, cmd):
        cmdStr = self.PololuCmd + cmd
        if PY2:
            self.usb.write(cmdStr)
        else:
            self.usb.write(bytes(cmdStr,'latin-1'))

    # Set channels min and max value range.  Use this as a safety to protect
    # from accidentally moving outside known safe parameters. A setting of 0
    # allows unrestricted movement.
    #
    # ***Note that the Maestro itself is configured to limit the range of servo travel
    # which has precedence over these values.  Use the Maestro Control Center to configure
    # ranges that are saved to the controller.  Use setRange for software controllable ranges.
    def setRange(self, chan, min, max):
        self.Mins[chan] = min
        self.Maxs[chan] = max

    # Return Minimum channel range value
    def getMin(self, chan):
        return self.Mins[chan]

    # Return Maximum channel range value
    def getMax(self, chan):
        return self.Maxs[chan]
        
    # Set channel to a specified target value.  Servo will begin moving based
    # on Speed and Acceleration parameters previously set.
    # Target values will be constrained within Min and Max range, if set.
    # For servos, target represents the pulse width in of quarter-microseconds
    # Servo center is at 1500 microseconds, or 6000 quarter-microseconds
    # Typcially valid servo range is 3000 to 9000 quarter-microseconds
    # If channel is configured for digital output, values < 6000 = Low ouput
    def setTarget(self, chan, target):
        # if Min is defined and Target is below, force to Min
        if self.Mins[chan] > 0 and target < self.Mins[chan]:
            target = self.Mins[chan]
        # if Max is defined and Target is above, force to Max
        if self.Maxs[chan] > 0 and target > self.Maxs[chan]:
            target = self.Maxs[chan]
        #    
        lsb = target & 0x7f #7 bits for least significant byte
        msb = (target >> 7) & 0x7f #shift 7 and take next 7 bits for msb
        cmd = chr(0x04) + chr(chan) + chr(lsb) + chr(msb)
        self.sendCmd(cmd)
        # Record Target value
        self.Targets[chan] = target
        
    # Set speed of channel
    # Speed is measured as 0.25microseconds/10milliseconds
    # For the standard 1ms pulse width change to move a servo between extremes, a speed
    # of 1 will take 1 minute, and a speed of 60 would take 1 second.
    # Speed of 0 is unrestricted.
    def setSpeed(self, chan, speed):
        lsb = speed & 0x7f #7 bits for least significant byte
        msb = (speed >> 7) & 0x7f #shift 7 and take next 7 bits for msb
        cmd = chr(0x07) + chr(chan) + chr(lsb) + chr(msb)
        self.sendCmd(cmd)

    # Set acceleration of channel
    # This provide soft starts and finishes when servo moves to target position.
    # Valid values are from 0 to 255. 0=unrestricted, 1 is slowest start.
    # A value of 1 will take the servo about 3s to move between 1ms to 2ms range.
    def setAccel(self, chan, accel):
        lsb = accel & 0x7f #7 bits for least significant byte
        msb = (accel >> 7) & 0x7f #shift 7 and take next 7 bits for msb
        cmd = chr(0x09) + chr(chan) + chr(lsb) + chr(msb)
        self.sendCmd(cmd)
    
    # Get the current position of the device on the specified channel
    # The result is returned in a measure of quarter-microseconds, which mirrors
    # the Target parameter of setTarget.
    # This is not reading the true servo position, but the last target position sent
    # to the servo. If the Speed is set to below the top speed of the servo, then
    # the position result will align well with the acutal servo position, assuming
    # it is not stalled or slowed.
    def getPosition(self, chan):
        cmd = chr(0x10) + chr(chan)
        self.sendCmd(cmd)
        lsb = ord(self.usb.read())
        msb = ord(self.usb.read())
        return (msb << 8) + lsb

    # Test to see if a servo has reached the set target position.  This only provides
    # useful results if the Speed parameter is set slower than the maximum speed of
    # the servo.  Servo range must be defined first using setRange. See setRange comment.
    #
    # ***Note if target position goes outside of Maestro's allowable range for the
    # channel, then the target can never be reached, so it will appear to always be
    # moving to the target.  
    def isMoving(self, chan):
        if self.Targets[chan] > 0:
            if self.getPosition(chan) != self.Targets[chan]:
                return True
        return False
    
    # Have all servo outputs reached their targets? This is useful only if Speed and/or
    # Acceleration have been set on one or more of the channels. Returns True or False.
    # Not available with Micro Maestro.
    def getMovingState(self):
        cmd = chr(0x13)
        self.sendCmd(cmd)
        if self.usb.read() == chr(0):
            return False
        else:
            return True

    # Run a Maestro Script subroutine in the currently active script. Scripts can
    # have multiple subroutines, which get numbered sequentially from 0 on up. Code your
    # Maestro subroutine to either infinitely loop, or just end (return is not valid).
    def runScriptSub(self, subNumber):
        cmd = chr(0x27) + chr(subNumber)
        # can pass a param with command 0x28
        # cmd = chr(0x28) + chr(subNumber) + chr(lsb) + chr(msb)
        self.sendCmd(cmd)

    # Stop the current Maestro Script
    def stopScript(self):
        cmd = chr(0x24)
        self.sendCmd(cmd)



CAM_ID = 0
def capture(camid = CAM_ID):
    count = 0
    cam = cv2.VideoCapture(camid)
    cam.set(3,720)#가로
    cam.set(4,1080)#세로
    
    if cam.isOpened() == False:
        print ('cant open the cam (%d)' % camid)
        return None
    
    while True:
        ret, frame = cam.read()
        cv2.imshow('test', frame)
        
        if frame is None:
            print ('frame is not exist')
            return None
        if count < 2:
            # png로 압축 없이 영상 저장 
            cv2.imwrite('./capture_image/test_'+str(count)+'.png',frame,params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            #cv2.imwrite('/Users/won_yeeun/Downloads/vf_test/vf_test'+str(count)+'.png',frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        
        else:
            break
        k = cv2.waitKey(1) 
        if k == 27: 
            break

        count = count+1

    cam.release()


def im_trim (img,x,y,w,h,title):
    img_trim = img[y:h, x:w]
    cv2.imwrite(title+'.jpg',img_trim)
    return img_trim #필요에 따라 결과물을 리턴 

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def demo(opt):
    """ model configuration """
    f = open('text_result.txt', 'a')
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')
                f.write(pred)
                f.write(" ")

            log.close()
            f.close()



def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, opt.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=opt.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if opt.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    capture()

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(opt.test_folder)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)


    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()


    net = CRAFT()     # initialize
    #print(image_list)
    print('Loading weights from checkpoint (' + opt.trained_model + ')')
    if opt.cuda:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model, map_location='cpu')))

    if opt.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if opt.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + opt.refiner_model + ')')
        if opt.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(opt.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(opt.refiner_model, map_location='cpu')))

        refine_net.eval()
        opt.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end = "\r")
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, opt.text_threshold, opt.link_threshold, opt.low_text, opt.cuda, opt.poly, refine_net)
        #print(bboxes)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    demo(opt)

    print("elapsed time : {}s".format(time.time() - t))

    cur_index = 0
    
    textpath = "text_result.txt"
    txt = read_text(textpath)
    list_txt, list_braille = txt_2_braille(txt)
    
    servo = Controller(ttyStr='/dev/cu.usbmodem002388571',device=0x0c)

    pos = 1500

    servo.setSpeed(0, 0)
    servo.setSpeed(1, 0)
    servo.setSpeed(2, 0)
    servo.setSpeed(3, 0)
    servo.setSpeed(4, 0)
    servo.setSpeed(5, 0)
    
    while True:
        # if the `q` key was pressed, break from the loop
        if(keyboard.is_pressed('s') == True):
            time.sleep(0.2)
            cur_index = clipping_index(len(list_braille), cur_index+1)
            motor_move(list_braille,list_txt,cur_index)
        elif(keyboard.is_pressed('a') == True):
            time.sleep(0.2)
            cur_index = clipping_index(len(list_braille), cur_index-1)
            motor_move(list_braille,list_txt,cur_index)



    

    servo.close()

