import PySimpleGUI as sg
from backend.apis.watson import apiWatson
from translate import Translator
from backend.processamento.ia import resultado
#p=ia.produtos
class Myapp :
    
    def __init__(self):
        self.valor=resultado()
        #layout
        layout=[
        [sg.Text('Descreva a ferida :')],
        [sg.Multiline (size=(150,15))],
        [sg.Button('Enviar',)],
        [sg.Text(text=self.valor)]
        ]
        #janela
        janela=sg.Window('Florence',size=(700,500)).layout(layout)
        self.button, self.Values = janela.Read()
        return self.Values
    
    def iniciar(self):
        print(self.Values)

    def clickbutton(self):
        if self.button== 'Enviar':
            return print(self.Values)

#print(produtos)
#watson=apiWatson()
tela=Myapp()
tela.iniciar()
##traduzir = Translator(from_lang="Portuguese", to_lang="English")
##traducao = traduzir.translate("Amo programar")
#json=watson.analisar(tela.clickbutton())




