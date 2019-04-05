# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: config.py
   create time: 2017年06月25日 星期日 10时56分55秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import numpy as np
import argparse
import os
import random
import time
import collections
import logging

batchSize = 64

learningRateBase = 0.001
learningRateDecayStep = 1000
learningRateDecayRate = 0.95

epochNum = 10                    # train epoch
generateNum = 1                   # number of generated poems per time

# type = "poetrySong"                   # dataset to use, shijing, songci, etc
# trainPoems = "./poem_generator_web_project/dataset/" + type + "/" + type + ".txt" # training file location
# checkpointsPath = "./poem_generator_web_project/checkpoints/" + type # checkpoints location
#
# typeReverse = "reversePoem"
# checkpointsPathReverse = "./poem_generator_web_project/checkpoints/" + typeReverse

saveStep = 1000                   # save model every savestep

logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename='server.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )

# evaluate
trainRatio = 0.8                    # train percentage
evaluateCheckpointsPath = "./checkpoints/evaluate"