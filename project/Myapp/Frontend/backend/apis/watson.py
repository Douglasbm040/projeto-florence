# api para analise dos textos

import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, ConceptsOptions

#from ..interface.screen import *


class watson():


 def analisar(self,texto):
     apikey = open(r'C:\Users\dougl\Documents\GitHub\projeto-florence\project\Myapp\Frontend\backend\apis\credenciais\credencial.txt', 'r')
     url = open(r'C:\Users\dougl\Documents\GitHub\projeto-florence\project\Myapp\Frontend\backend\apis\credenciais\url.txt', 'r')
     authenticator = IAMAuthenticator(str(apikey.read()))
     natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2021-03-25',
        authenticator=authenticator)
     natural_language_understanding.set_service_url(str(url.read()))
     response = natural_language_understanding.analyze(
        text='{0}'.format(texto),
        features=Features(concepts=ConceptsOptions(limit=1))).get_result()
     print(json.dumps(response, indent=2))   

watson=watson()
watson.analisar('helo my names is chris !!')