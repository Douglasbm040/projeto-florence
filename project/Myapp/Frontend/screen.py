import PySimpleGUI as sg
from backend.apis.watson import apiWatson

class Myapp :
    def __init__(self):
        valor=' '
        #layout
        layout=[
        [sg.Text('Descreva a ferida :')],
        [sg.Multiline (size=(150,15))],
        [sg.Button('Enviar',)],
        [sg.Text(text=valor)]
        ]
        #janela
        janela=sg.Window('Florence',size=(700,500)).layout(layout)
        self.button, self.Values = janela.Read()
    
    def iniciar(self):
        print(self.Values)

    def clickbutton(self):
        if self.button== 'Enviar':
            return str(self.Values[0],)#'utf-8')


watson=apiWatson()
tela=Myapp()
tela.iniciar()
watson.analisar(tela.clickbutton())


