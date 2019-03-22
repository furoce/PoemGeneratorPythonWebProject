from django.shortcuts import render
import json
from django.http import HttpResponse
import poem_generator_web_project.data as data
import poem_generator_web_project.model as model
from poem_generator_web_project.config import *

# Create your views here.

trainData = data.POEMS(trainPoems)
MCPangHu = model.MODEL(trainData)

def getTestPoem(request):
    poems = MCPangHu.test()
    tf.reset_default_graph()
    resp = {'poem': poems}
    return HttpResponse(json.dumps(resp), content_type="application/json")

def getHeadPoem(request, characters):
    poem = MCPangHu.testHead(characters)
    tf.reset_default_graph()
    resp = {'poem': poem}
    return HttpResponse(json.dumps(resp), content_type="application/json")

def getTailPoem(request, characters):
    poem = MCPangHu.getTail(characters)
    tf.reset_default_graph()
    resp = {'poem': poem}
    return HttpResponse(json.dumps(resp), content_type="application/json")
