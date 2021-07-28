import PySimpleGUI as sg
from backend.apis import watson

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
            print(self.Values[0])


watson=watson()

tela=Myapp()
tela.clickbutton()

