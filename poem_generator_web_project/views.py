from django.shortcuts import render
import json
from django.http import HttpResponse
import poem_generator_web_project.data as data
import poem_generator_web_project.model as model
from poem_generator_web_project.config import *

# Create your views here.

def getPoem(request, poemType):
    tf.reset_default_graph()
    trainPoems = "./poem_generator_web_project/dataset/" + poemType + ".txt"
    trainData = data.POEMS(trainPoems)
    MCPangHu = model.MODEL(trainData)
    checkpointsPath = "./poem_generator_web_project/checkpoints/" + poemType
    poem = MCPangHu.test(checkpointsPath)
    return HttpResponse(poem, content_type="application/json")

def getHeadPoem(request, poemType, characters):
    tf.reset_default_graph()
    poemType = poemType[:-3]
    trainPoems = "./poem_generator_web_project/dataset/" + poemType + ".txt"
    trainData = data.POEMS(trainPoems)
    MCPangHu = model.MODEL(trainData)
    checkpointsPath = "./poem_generator_web_project/checkpoints/" + poemType
    poem = MCPangHu.testHead(checkpointsPath, characters)
    return HttpResponse(poem, content_type="application/json")

def getTailPoem(request, poemType, characters):
    tf.reset_default_graph()
    poemType = poemType[:-3]
    poemType = "reverse" + poemType[0].upper() + poemType[1:]
    trainPoems = "./poem_generator_web_project/dataset/" + poemType + ".txt"
    trainData = data.POEMS(trainPoems)
    MCPangHu = model.MODEL(trainData)
    checkpointsPath = "./poem_generator_web_project/checkpoints/" + poemType
    poem = MCPangHu.testTail(checkpointsPath, characters)
    return HttpResponse(poem, content_type="application/json")
