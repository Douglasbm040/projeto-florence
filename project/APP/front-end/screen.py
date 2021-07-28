import PySimpleGUI as sg

class Myapp :
    def __init__(self):
        #layout
        layout=[
        [sg.Text('Descreva a ferida :')],
        [sg.Multiline (size=(150,15))],
        #[sg.Text('Senha :'),sg.Input()],
        
        [sg.Button('Enviar',)],
        ]
        #janela
        janela=sg.Window('Florence').layout(layout)
        self.button, self.Values = janela.Read()
    
    def iniciar(self):
        print(self.Values )

tela=Myapp()
tela.iniciar()
#tela.iniciar()
