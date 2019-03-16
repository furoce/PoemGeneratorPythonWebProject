from django.shortcuts import render
import json
from django.http import HttpResponse
import poem_generator_web_project.data as data
import poem_generator_web_project.model as model
from poem_generator_web_project.config import *

# Create your views here.

def getPoem(request):
    trainData = data.POEMS(trainPoems)
    MCPangHu = model.MODEL(trainData)
    poems = MCPangHu.test()
    resp = {'code': 100, 'detail': poems}
    return HttpResponse(json.dumps(resp), content_type="application/json")
